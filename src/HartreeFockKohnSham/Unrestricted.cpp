#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <tuple>
#include <deque>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <Maniverse/Manifold/Grassmann.h>
#include <Maniverse/Manifold/RealSymmetric.h>
#include <Maniverse/Optimizer/TrustRegion.h>
#include <libmwfn.h>

#include "../Macro.h"
#include "../Integral/Int2C1E.h"
#include "../Integral/Int4C2E.h"
#include "../Grid/Grid.h"
#include "../ExchangeCorrelation.h"
#include "../DIIS/CDIIS.h"
#include "../DIIS/EDIIS.h"
#include "../DIIS/ADIIS.h"

#define S (int2c1e.Overlap)
#define Hcore (int2c1e.Kinetic + int2c1e.Nuclear )

std::tuple<double, EigenVector, EigenVector, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Fa, EigenMatrix Fb,
		EigenVector Occa, EigenVector Occb,
		EigenMatrix Za, EigenMatrix Zb,
		int output, int nthreads){
	double oldE = 0;
	double E = 0;
	EigenVector epsa = EigenZero(Za.cols(), 1);
	EigenVector epsb = EigenZero(Zb.cols(), 1);
	EigenVector occa = Occa;
	EigenVector occb = Occb;
	EigenMatrix Ca = EigenZero(Za.rows(), Za.cols());
	EigenMatrix Cb = EigenZero(Zb.rows(), Zb.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	std::vector<EigenMatrix> Jas, Jbs, Kas, Kbs;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> es(S);
	EigenMatrix sinvsqrt = es.operatorInverseSqrt();
	std::function<
		std::tuple<
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
		> (std::vector<EigenMatrix>&, std::vector<bool>&)
	> update_func = [&](std::vector<EigenMatrix>& Fs_, std::vector<bool>&){
		assert(Fs_.size() == 1 && "Two Fock matrices packed together in spin-unrestricted SCF!");
		const EigenMatrix Fa_ = Fs_[0].leftCols(Fa.cols());
		const EigenMatrix Fb_ = Fs_[0].rightCols(Fb.cols());
		const EigenMatrix Faprime_ = Za.transpose() * Fa_ * Za;
		const EigenMatrix Fbprime_ = Zb.transpose() * Fb_ * Zb;
		eigensolver.compute(Faprime_);
		epsa = eigensolver.eigenvalues();
		Ca = Za * eigensolver.eigenvectors();
		eigensolver.compute(Fbprime_);
		epsb = eigensolver.eigenvalues();
		Cb = Zb * eigensolver.eigenvectors();
		oldE = E;
		E = 0;
		if ( T ){
			const EigenArray nas = 1. / ( 1. + ( ( epsa.array() - Mu ) / T ).exp() );
			const EigenArray nbs = 1. / ( 1. + ( ( epsb.array() - Mu ) / T ).exp() );
			occa = nas.matrix();
			occb = nbs.matrix();
			E += (
					T * (
						nas.pow(nas).log() + ( 1. - nas ).pow( 1. - nas ).log()
					).sum()
					- Mu * nas.sum()
			);
			E += (
					T * (
						nbs.pow(nbs).log() + ( 1. - nbs ).pow( 1. - nbs ).log()
					).sum()
					- Mu * nbs.sum()
			);
		}
		const EigenMatrix Da_ = Ca * occa.asDiagonal() * Ca.transpose();
		const EigenMatrix Db_ = Cb * occb.asDiagonal() * Cb.transpose();
		auto [Ghfa_, Ghfb_] = int4c2e.ContractInts(Da_, Db_, nthreads, 1);
		const EigenMatrix Fhfa_ = Hcore + Ghfa_;
		const EigenMatrix Fhfb_ = Hcore + Ghfb_;
		const EigenMatrix Fnewa_ = Fhfa_;
		const EigenMatrix Fnewb_ = Fhfb_;
		E += 0.5 * ( Dot(Da_, Hcore + Fnewa_) + Dot(Db_, Hcore + Fnewb_) );
		if (output>0){
			if ( T == 0 ) std::printf("Electronic energy = %.10f\n", E);
			else std::printf("Electronic grand potential = %.10f\n", E);
			std::printf("Changed by %E from the last step\n", E - oldE);
		}
		const EigenMatrix Ga_ = sinvsqrt * (Fnewa_ * Da_ * S - S * Da_ * Fnewa_ ) * sinvsqrt;
		const EigenMatrix Gb_ = sinvsqrt * (Fnewb_ * Db_ * S - S * Db_ * Fnewb_ ) * sinvsqrt;
		EigenMatrix Fnew_ = EigenZero(Fa.rows(), Fa.cols() * 2);
		Fnew_ << Fnewa_, Fnewb_;
		EigenMatrix G_ = EigenZero(Fa.rows(), Fa.cols() * 2);
		G_ << Ga_, Gb_;
		EigenMatrix Aux_ = EigenZero(Fa.rows(), Fa.cols() * 2 + 1);
		Aux_ << Da_, Db_, EigenZero(Fa.rows(), 1);
		Aux_(0, Fa.cols() * 2) = E;
		return std::make_tuple(
				std::vector<EigenMatrix>{Fnew_},
				std::vector<EigenMatrix>{G_},
				std::vector<EigenMatrix>{Aux_}
		);
	};
	EigenMatrix F = EigenZero(Fa.rows(), Fa.cols() * 2);
	F << Fa, Fb;
	std::vector<EigenMatrix> Fs = {F};
	ADIIS adiis(&update_func, 1, 20, 1e-1, 100, output>0 ? 2 : 0);
	if ( T == 0 ) if ( !adiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	CDIIS cdiis(&update_func, 1, 20, 1e-8, 100, output>0 ? 2 : 0);
	if ( T > 0 ) cdiis.Damps.push_back(std::make_tuple(0.1, 100, 0.75));
	cdiis.Steal(adiis);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsa, epsb, occa, occb, Ca, Cb);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix D1prime, EigenMatrix D2prime,
		EigenMatrix Z1, EigenMatrix Z2,
		int output, int nthreads){
	double E = 0;
	EigenVector epsilon1s = EigenZero(Z1.cols(), 1);
	EigenVector epsilon2s = EigenZero(Z2.cols(), 1);
	EigenMatrix C1 = EigenZero(Z1.rows(), Z1.cols());
	EigenMatrix C2 = EigenZero(Z2.rows(), Z2.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	EigenMatrix Vtmp1 = EigenZero(Z1.rows(), Z1.rows());
	EigenMatrix Gtmp2 = EigenZero(Z1.rows(), Z1.rows());
	Maniverse::Iterate M({Maniverse::Grassmann(D1prime).Clone(), Maniverse::Grassmann(D2prime).Clone()}, 1);
	std::function<
		std::tuple<
			double,
			std::vector<EigenMatrix>,
			std::vector<std::function<EigenMatrix (EigenMatrix)>>
		> (std::vector<EigenMatrix>, int)
	> dfunc_newton = [&](std::vector<EigenMatrix> Dprimes_, int order){
		const EigenMatrix D1prime_ = Dprimes_[0];
		const EigenMatrix D2prime_ = Dprimes_[1];
		const EigenMatrix D1_ = Z1 * D1prime_ * Z1.transpose();
		const EigenMatrix D2_ = Z2 * D2prime_ * Z2.transpose();
		auto [Ghf1_, Ghf2_] = int4c2e.ContractInts(D1_, D2_, nthreads, 1);
		const EigenMatrix Fhf1_ = Hcore + Ghf1_;
		const EigenMatrix Fhf2_ = Hcore + Ghf2_;
		const EigenMatrix F1_ = Fhf1_;
		const EigenMatrix F2_ = Fhf2_;
		const EigenMatrix F1prime_ = Z1.transpose() * F1_ * Z1; // Euclidean gradient
		const EigenMatrix F2prime_ = Z2.transpose() * F2_ * Z2; // Euclidean gradient
		eigensolver.compute(F1prime_);
		epsilon1s = eigensolver.eigenvalues();
		C1 = Z1 * eigensolver.eigenvectors();
		eigensolver.compute(F2prime_);
		epsilon2s = eigensolver.eigenvalues();
		C2 = Z2 * eigensolver.eigenvectors();
		const double E_ = 0.5 * ( Dot(D1_, Hcore + F1_) + Dot(D2_, Hcore + F2_) );
		std::vector<std::function<EigenMatrix (EigenMatrix)>> He;
		if ( order == 2 ){
			He.push_back([&Vtmp1](EigenMatrix v1prime){
				Vtmp1 = v1prime;
				return EigenZero(v1prime.rows(), v1prime.cols());
			});
			He.push_back([Z1, Z2, &Vtmp1, &int4c2e, &Gtmp2, nthreads](EigenMatrix v2prime){
				const EigenMatrix v1 = Z1 * Vtmp1 * Z1.transpose();
				const EigenMatrix v2 = Z2 * v2prime * Z2.transpose();
				auto [Gtmp1_, Gtmp2_] = int4c2e.ContractInts(v1, v2, nthreads, 0);
				Gtmp2 = Gtmp2_;
				return (EigenMatrix)(Z2.transpose() * Gtmp1_ * Z2);
			});
			He.push_back([Z1, &Gtmp2](EigenMatrix /*v1prime*/){
				return (EigenMatrix)(Z1.transpose() * Gtmp2 * Z1);
			});
			He.push_back([](EigenMatrix v2prime){
				return EigenZero(v2prime.rows(), v2prime.cols());
			});
		}
		return std::make_tuple(
				E_,
				std::vector<EigenMatrix>{F1prime_, F2prime_},
				He
		);
	};
	Maniverse::TrustRegionSetting tr_setting;
	if ( ! Maniverse::TrustRegion(
				dfunc_newton, tr_setting, {1.e-8, 1.e-5, 1.e-5},
				0.001, 1, 100, E, M, output
	) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilon1s, epsilon2s, C1, C2);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedRiemannARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix D1prime, EigenMatrix D2prime,
		EigenMatrix Z1, EigenMatrix Z2,
		int output, int nthreads){
	double E = 0;
	EigenVector epsilon1s = EigenZero(Z1.cols(), 1);
	EigenVector epsilon2s = EigenZero(Z2.cols(), 1);
	EigenMatrix C1 = EigenZero(Z1.rows(), Z1.cols());
	EigenMatrix C2 = EigenZero(Z2.rows(), Z2.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	// ARH hessian related
	std::deque<EigenMatrix> D1primes;
	std::deque<EigenMatrix> D2primes;
	std::deque<EigenMatrix> F1primes;
	std::deque<EigenMatrix> F2primes;
	std::vector<double> TrDsv;

	Maniverse::Iterate M({Maniverse::Grassmann(D1prime).Clone(), Maniverse::Grassmann(D2prime).Clone()}, 1);
	std::function<
		std::tuple<
			double,
			std::vector<EigenMatrix>,
			std::vector<std::function<EigenMatrix (EigenMatrix)>>
		> (std::vector<EigenMatrix>, int)
	> dfunc_newton = [&](std::vector<EigenMatrix> Dprimes_, int order){
		const EigenMatrix D1prime_ = Dprimes_[0];
		const EigenMatrix D2prime_ = Dprimes_[1];
		const EigenMatrix D1_ = Z1 * D1prime_ * Z1.transpose();
		const EigenMatrix D2_ = Z2 * D2prime_ * Z2.transpose();
		auto [Ghf1_, Ghf2_] = int4c2e.ContractInts(D1_, D2_, nthreads, 1);
		const EigenMatrix Fhf1_ = Hcore + Ghf1_;
		const EigenMatrix Fhf2_ = Hcore + Ghf2_;
		const EigenMatrix F1_ = Fhf1_;
		const EigenMatrix F2_ = Fhf2_;
		const EigenMatrix F1prime_ = Z1.transpose() * F1_ * Z1; // Euclidean gradient
		const EigenMatrix F2prime_ = Z2.transpose() * F2_ * Z2; // Euclidean gradient

		// ARH hessian related
		D1primes.push_back(D1prime_);
		D2primes.push_back(D2prime_);
		F1primes.push_back(F1prime_);
		F2primes.push_back(F2prime_);
		if ( D1primes.size() > 20 ){
			D1primes.pop_front();
			D2primes.pop_front();
			F1primes.pop_front();
			F2primes.pop_front();
		}

		eigensolver.compute(F1prime_);
		epsilon1s = eigensolver.eigenvalues();
		C1 = Z1 * eigensolver.eigenvectors();
		eigensolver.compute(F2prime_);
		epsilon2s = eigensolver.eigenvalues();
		C2 = Z2 * eigensolver.eigenvectors();
		const double E_ = 0.5 * ( Dot(D1_, Hcore + F1_) + Dot(D2_, Hcore + F2_) );
		std::vector<std::function<EigenMatrix (EigenMatrix)>> He;
		if ( order == 2 && D1primes.size() > 1 ){
			if (output>0) std::printf("Using augmented Roothaan-Hall hessian with %d previous iterations\n", (int)D1primes.size());
			const int size = (int)D1primes.size() - 1;
			const int nbasis = Z1.rows();
			std::vector<EigenMatrix> D1diff(size, EigenZero(nbasis, nbasis));
			std::vector<EigenMatrix> D2diff(size, EigenZero(nbasis, nbasis));
			std::vector<EigenMatrix> F1diff(size, EigenZero(nbasis, nbasis));
			std::vector<EigenMatrix> F2diff(size, EigenZero(nbasis, nbasis));
			for ( int i = 0; i < size; i++ ){
				D1diff[i] = D1primes[i] - D1prime_;
				D2diff[i] = D2primes[i] - D2prime_;
				F1diff[i] = F1primes[i] - F1prime_;
				F2diff[i] = F2primes[i] - F2prime_;
			}
			EigenMatrix T = EigenZero(size, size);
			for ( int i = 0; i < size; i++ ) for ( int j = i; j < size; j++ ){
				T(i, j) = T(j, i) = D1diff[i].cwiseProduct(D1diff[j]).sum() + D2diff[i].cwiseProduct(D2diff[j]).sum();
			}
			const EigenMatrix Tinv = T.inverse();
			TrDsv.resize(size);
			He.push_back([nbasis, size, D1diff, &TrDsv](EigenMatrix v1){
					for ( int j = 0; j < size; j++ )
						TrDsv[j] = D1diff[j].cwiseProduct(v1).sum();
					return EigenZero(nbasis, nbasis);
			});
			He.push_back([nbasis, size, F1diff, D2diff, Tinv, &TrDsv](EigenMatrix v2){
					for ( int j = 0; j < size; j++ )
						TrDsv[j] += D2diff[j].cwiseProduct(v2).sum();
					EigenMatrix Hv2 = EigenZero(nbasis, nbasis);
					for ( int i = 0; i < size; i++ ){
						double coeff = 0;
						for ( int j = 0; j < size; j++ ){
							coeff += Tinv(i, j) * TrDsv[j];
						}
						Hv2 += F1diff[i] * coeff;
					}
					return Hv2;
			});
			He.push_back([](EigenMatrix v1){
					return EigenZero(v1.rows(), v1.cols());
			});
			He.push_back([nbasis, size, F2diff, &TrDsv, Tinv](EigenMatrix /*v2*/){
					EigenMatrix Hv2 = EigenZero(nbasis, nbasis);
					for ( int i = 0; i < size; i++ ){
						double coeff = 0;
						for ( int j = 0; j < size; j++ ){
							coeff += Tinv(i, j) * TrDsv[j];
						}
						Hv2 += F2diff[i] * coeff;
					}
					return (EigenMatrix)Hv2;
			});
		}else{
			He.push_back([](EigenMatrix v1){ return v1; });
			He.push_back([](EigenMatrix v2){ return EigenZero(v2.rows(), v2.cols()); });
			He.push_back([](EigenMatrix v1){ return EigenZero(v1.rows(), v1.cols()); });
			He.push_back([](EigenMatrix v2){ return v2; });
		}
		return std::make_tuple(
				E_,
				std::vector<EigenMatrix>{F1prime_, F2prime_},
				He
		);
	};
	Maniverse::TrustRegionSetting tr_setting;
	if ( ! Maniverse::TrustRegion(
				dfunc_newton, tr_setting, {1.e-8, 1.e-5, 1.e-5},
				0.01, 1, 300, E, M, output
	) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilon1s, epsilon2s, C1, C2);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedRiemannARH_villain(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix D1prime, EigenMatrix D2prime,
		EigenMatrix Z1, EigenMatrix Z2,
		int output, int nthreads){
	double E = 0;
	EigenVector epsilon1s = EigenZero(Z1.cols(), 1);
	EigenVector epsilon2s = EigenZero(Z2.cols(), 1);
	EigenMatrix C1 = EigenZero(Z1.rows(), Z1.cols());
	EigenMatrix C2 = EigenZero(Z2.rows(), Z2.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	// ARH hessian related
	std::deque<EigenMatrix> D1primes;
	std::deque<EigenMatrix> D2primes;
	std::deque<EigenMatrix> F1primes;
	std::deque<EigenMatrix> F2primes;
	std::vector<double> TrDsv;

	Maniverse::Iterate M({Maniverse::Grassmann(D1prime).Clone(), Maniverse::Grassmann(D2prime).Clone()}, 1);
	std::function<
		std::tuple<
			double,
			std::vector<EigenMatrix>,
			std::vector<std::function<EigenMatrix (EigenMatrix)>>
		> (std::vector<EigenMatrix>, int)
	> dfunc_newton = [&](std::vector<EigenMatrix> Dprimes_, int order){
		const EigenMatrix D1prime_ = Dprimes_[0];
		const EigenMatrix D2prime_ = Dprimes_[1];
		const EigenMatrix D1_ = Z1 * D1prime_ * Z1.transpose();
		const EigenMatrix D2_ = Z2 * D2prime_ * Z2.transpose();
		auto [Ghf1_, Ghf2_] = int4c2e.ContractInts(D1_, D2_, nthreads, 1);
		const EigenMatrix Fhf1_ = Hcore + Ghf1_;
		const EigenMatrix Fhf2_ = Hcore + Ghf2_;
		const EigenMatrix F1_ = Fhf1_;
		const EigenMatrix F2_ = Fhf2_;
		const EigenMatrix F1prime_ = Z1.transpose() * F1_ * Z1; // Euclidean gradient
		const EigenMatrix F2prime_ = Z2.transpose() * F2_ * Z2; // Euclidean gradient

		// ARH hessian related
		D1primes.push_back(D1prime_);
		D2primes.push_back(D2prime_);
		F1primes.push_back(F1prime_);
		F2primes.push_back(F2prime_);
		if ( D1primes.size() > 20 ){
			D1primes.pop_front();
			D2primes.pop_front();
			F1primes.pop_front();
			F2primes.pop_front();
		}

		eigensolver.compute(F1prime_);
		epsilon1s = eigensolver.eigenvalues();
		C1 = Z1 * eigensolver.eigenvectors();
		eigensolver.compute(F2prime_);
		epsilon2s = eigensolver.eigenvalues();
		C2 = Z2 * eigensolver.eigenvectors();
		const double E_ = 0.5 * ( Dot(D1_, Hcore + F1_) + Dot(D2_, Hcore + F2_) );
		std::vector<std::function<EigenMatrix (EigenMatrix)>> He;
		if ( order == 2 && D1primes.size() > 1 ){
			const int size = (int)D1primes.size() - 1;
			const int nbasis = Z1.rows();
			std::vector<EigenMatrix> D1diff(size, EigenZero(nbasis, nbasis));
			std::vector<EigenMatrix> D2diff(size, EigenZero(nbasis, nbasis));
			std::vector<EigenMatrix> F1diff(size, EigenZero(nbasis, nbasis));
			std::vector<EigenMatrix> F2diff(size, EigenZero(nbasis, nbasis));
			for ( int i = 0; i < size; i++ ){
				D1diff[i] = D1primes[i] - D1prime_;
				D2diff[i] = D2primes[i] - D2prime_;
				F1diff[i] = F1primes[i] - F1prime_;
				F2diff[i] = F2primes[i] - F2prime_;
			}
			EigenMatrix T1 = EigenZero(size, size);
			for ( int i = 0; i < size; i++ ) for ( int j = i; j < size; j++ ){
				T1(i, j) = T1(j, i) = D1diff[i].cwiseProduct(D1diff[j]).sum();
			}
			EigenMatrix T2 = EigenZero(size, size);
			for ( int i = 0; i < size; i++ ) for ( int j = i; j < size; j++ ){
				T2(i, j) = T2(j, i) = D2diff[i].cwiseProduct(D2diff[j]).sum();
			}
			const EigenMatrix T1inv = T1.inverse();
			const EigenMatrix T2inv = T2.inverse();
			TrDsv.resize(size);
			He.push_back([nbasis, size, D1diff, F1diff, &TrDsv, T1inv](EigenMatrix v1){
					for ( int j = 0; j < size; j++ )
						TrDsv[j] = D1diff[j].cwiseProduct(v1).sum();
					EigenMatrix Hv1 = EigenZero(nbasis, nbasis);
					for ( int i = 0; i < size; i++ ){
						double coeff = 0;
						for ( int j = 0; j < size; j++ ){
							coeff += T1inv(i, j) * TrDsv[j];
						}
						Hv1 += F1diff[i] * coeff;
					}
					return Hv1;
			});
			He.push_back([](EigenMatrix v2){
					return EigenZero(v2.rows(), v2.cols());
			});
			He.push_back([](EigenMatrix v1){
					return EigenZero(v1.rows(), v1.cols());
			});
			He.push_back([nbasis, size, D2diff, F2diff, &TrDsv, T2inv](EigenMatrix v2){
					for ( int j = 0; j < size; j++ )
						TrDsv[j] = D2diff[j].cwiseProduct(v2).sum();
					EigenMatrix Hv2 = EigenZero(nbasis, nbasis);
					for ( int i = 0; i < size; i++ ){
						double coeff = 0;
						for ( int j = 0; j < size; j++ ){
							coeff += T2inv(i, j) * TrDsv[j];
						}
						Hv2 += F2diff[i] * coeff;
					}
					return (EigenMatrix)Hv2;
			});
		}else{
			He.push_back([](EigenMatrix v1){ return v1; });
			He.push_back([](EigenMatrix v2){ return EigenZero(v2.rows(), v2.cols()); });
			He.push_back([](EigenMatrix v1){ return EigenZero(v1.rows(), v1.cols()); });
			He.push_back([](EigenMatrix v2){ return v2; });
		}
		return std::make_tuple(
				E_,
				std::vector<EigenMatrix>{F1prime_, F2prime_},
				He
		);
	};
	Maniverse::TrustRegionSetting tr_setting;
	if ( ! Maniverse::TrustRegion(
				dfunc_newton, tr_setting, {1.e-8, 1.e-5, 1.e-5},
				0.01, 1, 300, E, M, output
	) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilon1s, epsilon2s, C1, C2);
}
