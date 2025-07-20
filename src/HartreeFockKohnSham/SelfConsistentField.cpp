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

#include <random>
#include <iostream>

#define S (int2c1e.Overlap)
#define Hcore (int2c1e.Kinetic + int2c1e.Nuclear )

std::function<EigenMatrix (EigenMatrix)> RestrictedHessianARH(std::deque<EigenMatrix> Ds, std::deque<EigenMatrix> Fs){
	assert( Ds.size() == Fs.size() );
	assert( Ds.size() > 1 );
	const int size = (int)Ds.size() - 1;
	const EigenMatrix D = Ds.back();
	const EigenMatrix F = Fs.back();
	const int nbasis = D.cols();
	std::vector<EigenMatrix> Ddiff(size, EigenZero(nbasis, nbasis));
	std::vector<EigenMatrix> Fdiff(size, EigenZero(nbasis, nbasis));
	for ( int i = 0; i < size; i++ ){
		Ddiff[i] = Ds[i] - D;
		Fdiff[i] = Fs[i] - F;
	}
	EigenMatrix T = EigenZero(size, size);
	for ( int i = 0; i < size; i++ ) for ( int j = i; j < size; j++ ){
		T(i, j) = T(j, i) = Ddiff[i].cwiseProduct(Ddiff[j]).sum();
	}
	const EigenMatrix Tinv = T.inverse();
	const std::function<EigenMatrix (EigenMatrix)> H = [size, nbasis, Ddiff, Fdiff, Tinv](EigenMatrix v){
		EigenMatrix Hv = EigenZero(nbasis, nbasis);
		for ( int i = 0; i < size; i++ ) for ( int j = 0; j < size; j++ ){
			Hv += Fdiff[i] * Tinv(i, j) * Ddiff[j].cwiseProduct(v).sum();
		}
		return (EigenMatrix)Hv;
	};
	return H;
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFock(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Fprime, EigenVector Occ, EigenMatrix Z,
		int output, int nthreads){
	double E = 0;
	EigenVector occupations = Occ;
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	Iterate M({RealSymmetric(Fprime).Clone()}, 1);
	std::function<
		std::tuple<
			double,
			std::vector<EigenMatrix>,
			std::vector<std::function<EigenMatrix (EigenMatrix)>>
		> (std::vector<EigenMatrix>, int)
	> ffunc_newton = [&](std::vector<EigenMatrix> Fprimes_, int order){
		const EigenMatrix Fprime_ = Fprimes_[0];
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		const EigenMatrix Cprime = eigensolver.eigenvectors();
		C = Z * Cprime;
		EigenMatrix K = EigenZero(Z.cols(), Z.cols());
		if ( T ){
			const EigenArray ns = 1. / ( 1. + ( ( epsilons.array() - Mu ) / T ).exp() );
			occupations = (EigenVector)ns;
			K.diagonal() = (EigenVector)(ns * ( ns - 1 ) / T);
		}
		for ( int i = 0; i < Z.cols(); i++ ) for ( int j = 0; j < i; j++ ){
			K(i, j) = K(j, i) = ( occupations(i) - occupations(j) ) / ( epsilons(i) - epsilons(j) );
		}
		const EigenMatrix Dprime_ = Cprime * occupations.asDiagonal() * Cprime.transpose();
		const EigenMatrix D_ = C * occupations.asDiagonal() * C.transpose();
		const EigenMatrix Ghf_tilde_ = int4c2e.ContractInts(D_, nthreads, 1);
		double Exc_ = 0;
		EigenMatrix Gxc_tilde_ = EigenZero(Ghf_tilde_.rows(), Ghf_tilde_.cols());
		if (xc){
			grid.getDensity(2 * D_);
			xc.Evaluate("ev", grid);
			xc.Evaluate("f", grid);
			Exc_ = grid.getEnergy();
			Gxc_tilde_ = grid.getFock();
		}
		const EigenMatrix Fhf_tilde_ = Hcore + Ghf_tilde_;
		const EigenMatrix F_tilde_ = Fhf_tilde_ + Gxc_tilde_;
		const EigenMatrix Fprime_tilde_ = Z.transpose() * F_tilde_ * Z;
		double E_ = 0.5 * D_.cwiseProduct( Hcore + Fhf_tilde_ ).sum() + Exc_;
		if ( T ){
			EigenArray ns = occupations.array();
			E_ += (
					T * (
						ns.pow(ns).log() + ( 1. - ns ).pow( 1. - ns ).log()
					).sum()
					- Mu * ns.sum()
			);
		}
		const EigenMatrix Fao_bar = T ? ( Fprime_tilde_ - Fprime_ ) : Fprime_tilde_;
		const EigenMatrix Fmo_bar = Cprime.transpose() * Fao_bar * Cprime;
		const EigenMatrix Ge = Cprime * Fmo_bar.cwiseProduct(K) * Cprime.transpose();
		std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix vprime){ return vprime; };
		if ( order == 2 ) He = [Z, Cprime, K, T, &int4c2e, &grid, &xc, nthreads](EigenMatrix delta){
			const EigenMatrix square = Cprime.transpose() * delta * Cprime;
			const EigenMatrix pentagon = Cprime * K.cwiseProduct(square) * Cprime.transpose();
			const EigenMatrix v = Z * pentagon * Z.transpose();
			const EigenMatrix FhfU = int4c2e.ContractInts(v, nthreads, 0);
			EigenMatrix FxcU = EigenZero(FhfU.rows(), FhfU.cols());
			if (xc){
				std::vector<Eigen::Tensor<double, 1>> RhoUss, SigmaUss;
				std::vector<Eigen::Tensor<double, 2>> Rho1Uss;
				grid.getDensityU({2*v}, RhoUss, Rho1Uss, SigmaUss);
				FxcU = grid.getFockU(RhoUss, Rho1Uss, SigmaUss)[0];
			}
			const EigenMatrix hexagon = Z.transpose() * (FhfU + FxcU) * Z;
			const EigenMatrix octagon = Cprime.transpose() * hexagon * Cprime;
			const EigenMatrix Hdelta = T ? ( Cprime * K.cwiseProduct(octagon - square) * Cprime.transpose() ).eval() : ( Cprime * K.cwiseProduct(octagon) * Cprime.transpose() ).eval();
			return Hdelta;
		};
		return std::make_tuple(
				E_,
				std::vector<EigenMatrix>{Ge},
				std::vector<std::function<EigenMatrix (EigenMatrix)>>{He}
		);
	};
	TrustRegionSetting tr_setting;
	tr_setting.R0 = 1;
	if ( !TrustRegion(
			ffunc_newton, tr_setting, {1.e-8, 1.e-5, 1.e-1},
			0.00001, 1, 100, E, M, output
	) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilons, occupations, C);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedGrassmannARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Dprime, EigenMatrix Z,
		int output, int nthreads){
	double E = 0;
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	std::deque<EigenMatrix> Dprimes = {};
	std::deque<EigenMatrix> Fprimes = {};
	Iterate M({Grassmann(Dprime).Clone()}, 1);
	std::function<
		std::tuple<
			double,
			std::vector<EigenMatrix>,
			std::vector<std::function<EigenMatrix (EigenMatrix)>>
		> (std::vector<EigenMatrix>, int)
	> dfunc_newton = [&](std::vector<EigenMatrix> Dprimes_, int order){
		const EigenMatrix Dprime_ = Dprimes_[0];
		const EigenMatrix D_ = Z * Dprime_ * Z.transpose();
		const EigenMatrix Ghf_ = int4c2e.ContractInts(D_, nthreads, 1);
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(Ghf_.rows(), Ghf_.cols());
		if (xc){
			grid.getDensity(2 * D_);
			xc.Evaluate("ev", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock();
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix F_ = Fhf_+ Gxc_;
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z; // Euclidean gradient
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		Dprimes.push_back(Dprime_);
		Fprimes.push_back(Fprime_);
		if ( Dprimes.size() > 20 ){
			Dprimes.pop_front();
			Fprimes.pop_front();
		}
		const double E_ = 0.5 * (( D_ * ( Hcore + Fhf_ ) ).trace() + Exc_);
		std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix vprime){ return vprime; };
		if ( order == 2 && Dprimes.size() > 1 ){
			He = RestrictedHessianARH(Dprimes, Fprimes);
			if (output>0) std::printf("Augmented Roothaan-Hall hessian with %d previous iterations", (int)Dprimes.size());
		}
		return std::make_tuple(
				E_,
				std::vector<EigenMatrix>{Fprime_},
				std::vector<std::function<EigenMatrix (EigenMatrix)>>{He}
		);
	};
	TrustRegionSetting tr_setting;
	if ( ! TrustRegion(
				dfunc_newton, tr_setting, {1.e-8, 1.e-5, 1.e-5},
				0.00001, 1, 100, E, M, output
	) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilons, C);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedGrassmann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Dprime, EigenMatrix Z,
		int output, int nthreads){
	double E = 0;
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	Iterate M({Grassmann(Dprime).Clone()}, 1);
	std::function<
		std::tuple<
			double,
			std::vector<EigenMatrix>,
			std::vector<std::function<EigenMatrix (EigenMatrix)>>
		> (std::vector<EigenMatrix>, int)
	> dfunc_newton = [&](std::vector<EigenMatrix> Dprimes_, int order){
		const EigenMatrix Dprime_ = Dprimes_[0];
		const EigenMatrix D_ = Z * Dprime_ * Z.transpose();
		const EigenMatrix Ghf_ = int4c2e.ContractInts(D_, nthreads, 1);
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(Ghf_.rows(), Ghf_.cols());
		if (xc){
			grid.getDensity(2 * D_);
			xc.Evaluate("ev", grid);
			xc.Evaluate("f", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock();
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix F_ = Fhf_+ Gxc_;
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z; // Euclidean gradient
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		const double E_ = 0.5 * (( D_ * ( Hcore + Fhf_ ) ).trace() + Exc_);
		std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix vprime){ return vprime; };
		if ( order == 2 ) He = [Z, &int4c2e, &grid, nthreads](EigenMatrix vprime){
			const EigenMatrix v = Z * vprime * Z.transpose();
			const EigenMatrix FhfU = int4c2e.ContractInts(v, nthreads, 0);
			std::vector<Eigen::Tensor<double, 1>> RhoUss, SigmaUss;
			std::vector<Eigen::Tensor<double, 2>> Rho1Uss;
			grid.getDensityU({2*v}, RhoUss, Rho1Uss, SigmaUss);
			const EigenMatrix FxcU = grid.getFockU(RhoUss, Rho1Uss, SigmaUss)[0];
			const EigenMatrix FU = FhfU + FxcU;
			return (EigenMatrix)(Z.transpose() * FU * Z);
		};
		return std::make_tuple(
				E_,
				std::vector<EigenMatrix>{Fprime_},
				std::vector<std::function<EigenMatrix (EigenMatrix)>>{He}
		);
	};
	TrustRegionSetting tr_setting;
	if ( ! TrustRegion(
				dfunc_newton, tr_setting, {1.e-8, 1.e-5, 1.e-5},
				0.00001, 1, 100, E, M, output
	) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilons, C);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedGrassmann(
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
	EigenMatrix Jtmp1 = EigenZero(Z1.rows(), Z1.rows());
	EigenMatrix Ktmp1 = EigenZero(Z1.rows(), Z1.rows());
	EigenMatrix Jtmp2 = EigenZero(Z2.rows(), Z2.rows());
	EigenMatrix Ktmp2 = EigenZero(Z2.rows(), Z2.rows());
	Iterate M({Grassmann(D1prime).Clone(), Grassmann(D2prime).Clone()}, 1);
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
		auto [J1_, K1_] = int4c2e.ContractInts2(D1_, nthreads, 1);
		auto [J2_, K2_] = int4c2e.ContractInts2(D2_, nthreads, 1);
		const EigenMatrix Fhf1_ = Hcore + J1_ + J2_ - K1_;
		const EigenMatrix Fhf2_ = Hcore + J1_ + J2_ - K2_;
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
			He.push_back([Z1, &Jtmp1, &Ktmp1, &int4c2e, nthreads](EigenMatrix v1prime){
				const EigenMatrix v1 = Z1 * v1prime * Z1.transpose();
				std::tie(Jtmp1, Ktmp1) = int4c2e.ContractInts2(v1, nthreads, 0);
				const EigenMatrix F1U = Jtmp1 - Ktmp1;
				return (EigenMatrix)(Z1.transpose() * F1U * Z1);
			});
			He.push_back([Z2, &Jtmp2, &Ktmp2, &int4c2e, nthreads](EigenMatrix v2prime){
				const EigenMatrix v2 = Z2 * v2prime * Z2.transpose();
				std::tie(Jtmp2, Ktmp2) = int4c2e.ContractInts2(v2, nthreads, 0);
				return (EigenMatrix)(Z2.transpose() * Jtmp2 * Z2);
			});
			He.push_back([Z1, &Jtmp1](EigenMatrix /*v1prime*/){
				return (EigenMatrix)(Z1.transpose() * Jtmp1 * Z1);
			});
			He.push_back([Z2, &Jtmp2, &Ktmp2](EigenMatrix /*v2prime*/){
				const EigenMatrix F2U = Jtmp2 - Ktmp2;
				return (EigenMatrix)(Z2.transpose() * F2U * Z2);
			});
		}
		return std::make_tuple(
				E_,
				std::vector<EigenMatrix>{F1prime_, F2prime_},
				He
		);
	};
	TrustRegionSetting tr_setting;
	if ( ! TrustRegion(
				dfunc_newton, tr_setting, {1.e-8, 1.e-5, 1.e-5},
				0.00001, 1, 100, E, M, output
	) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilon1s, epsilon2s, C1, C2);
}

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

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix F, EigenVector Occ, EigenMatrix Z,
		int output, int nthreads){
	double oldE = 0;
	double E = 0;
	const int nbasis = F.cols();
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenVector occupations = Occ;
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	std::function<std::tuple<
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
			>(std::vector<EigenMatrix>&, std::vector<bool>&)
	> update_func = [&](std::vector<EigenMatrix>& Fs_, std::vector<bool>&){
		assert(Fs_.size() == 1 && "Only one Fock matrix should be optimized in spin-restricted SCF!");
		const EigenMatrix F_ = Fs_[0];
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z;
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		oldE = E;
		E = 0;
		if ( T ){
			const EigenArray ns = 1. / ( 1. + ( ( epsilons.array() - Mu ) / T ).exp() );
			occupations = (EigenVector)ns;
			E += 2 * (
					T * (
						ns.pow(ns).log() + ( 1. - ns ).pow( 1. - ns ).log()
					).sum()
					- Mu * ns.sum()
			);
		}
		const EigenMatrix D_ = C * occupations.asDiagonal() * C.transpose();
		const EigenMatrix Ghf_ = int4c2e.ContractInts(D_, nthreads, 1);
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(nbasis, nbasis);
		if (xc){
			grid.getDensity(2 * D_);
			xc.Evaluate("ev", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock();
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix Fnew_ = Fhf_ + Gxc_;
		E += (D_ * ( Hcore + Fhf_ )).trace() + Exc_;
		if (output>0){
			if ( T == 0 ) std::printf("Electronic energy = %.10f\n", E);
			else std::printf("Electronic grand potential = %.10f\n", E);
			std::printf("Changed by %E from the last step\n", E - oldE);
		}
		EigenMatrix G_ = Fnew_ * D_ * S - S * D_ * Fnew_;
		EigenMatrix Aux_ = EigenZero(F.rows(), F.cols() + 1);
		Aux_ << D_, EigenZero(F.rows(), 1);
		Aux_(0, F.cols()) = E;
		return std::make_tuple(
				std::vector<EigenMatrix>{Fnew_},
				std::vector<EigenMatrix>{G_},
				std::vector<EigenMatrix>{Aux_}
		);
	};
	std::vector<EigenMatrix> Fs = {F};
	ADIIS adiis(&update_func, 1, 20, 1e-1, 100, output>0 ? 2 : 0);
	if ( T == 0 ) if ( !adiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	CDIIS cdiis(&update_func, 1, 20, 1e-8, 100, output>0 ? 2 : 0);
	if ( T > 0 ) cdiis.Damps.push_back(std::make_tuple(0.1, 100, 0.75));
	cdiis.Steal(adiis);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilons, occupations, C);
}

double HartreeFockKohnSham(Mwfn& mwfn, Environment& env, Int2C1E& int2c1e, Int4C2E& int4c2e, ExchangeCorrelation& xc, Grid& grid, std::string scf, int output, int nthreads){
	Eigen::setNbThreads(nthreads);

	const double T = env.Temperature;
	const double Mu = env.ChemicalPotential;
	if (output > 0) std::printf("Self-consistent field in %s-canonical ensemble\n", T > 0 ? "grand" : "micro");
	if ( T > 0 && scf != "DIIS" && scf != "FOCK" ) throw std::runtime_error("Only DIIS optimization is supported for finite-temperature DFT!");

	const EigenMatrix Z = mwfn.getCoefficientMatrix(mwfn.Wfntype);
	EigenMatrix Z1 = EigenZero(Z.rows(), Z.cols());
	EigenMatrix Z2 = EigenZero(Z.rows(), Z.cols());
	if ( mwfn.Wfntype == 1 ){
		Z1 = mwfn.getCoefficientMatrix(1);
		Z2 = mwfn.getCoefficientMatrix(2);
	}

	double E_scf = 0;
	if ( scf == "FOCK" ){
		EigenMatrix Fprime = mwfn.getEnergy().asDiagonal();
		EigenVector Occ = mwfn.getOccupation() / 2;
		auto [E, epsilons, occupations, C] = RestrictedFock(T, Mu, int2c1e, int4c2e, xc, grid, Fprime, Occ, Z, output-1, nthreads);
		E_scf = E * 2;
		mwfn.setEnergy(epsilons);
		mwfn.setOccupation(occupations * 2);
		mwfn.setCoefficientMatrix(C);
	}else if ( scf == "GRASSMANN-ARH" ){
		EigenMatrix Dprime = (mwfn.getOccupation() / 2).asDiagonal();
		auto [E, epsilons, C] = RestrictedGrassmannARH(int2c1e, int4c2e, xc, grid, Dprime, Z, output-1, nthreads);
		E_scf = 2 * E;
		mwfn.setEnergy(epsilons);
		mwfn.setCoefficientMatrix(C);
	}else if ( scf == "GRASSMANN" ){
		if ( mwfn.Wfntype == 0 ){
			EigenMatrix Dprime = (mwfn.getOccupation() / 2).asDiagonal();
			auto [E, epsilons, C] = RestrictedGrassmann(int2c1e, int4c2e, xc, grid, Dprime, Z, output-1, nthreads);
			E_scf = 2 * E;
			mwfn.setEnergy(epsilons);
			mwfn.setCoefficientMatrix(C);
		}else if ( mwfn.Wfntype == 1 ){
			EigenMatrix D1prime = mwfn.getOccupation(1).asDiagonal();
			EigenMatrix D2prime = mwfn.getOccupation(2).asDiagonal();
			auto [E, epsilon1s, epsilon2s, C1, C2] = UnrestrictedGrassmann(int2c1e, int4c2e, D1prime, D2prime, Z1, Z2, output-1, nthreads);
			E_scf = E;
			mwfn.setEnergy(epsilon1s, 1);
			mwfn.setEnergy(epsilon2s, 2);
			mwfn.setCoefficientMatrix(C1, 1);
			mwfn.setCoefficientMatrix(C2, 2);
		}
	}else if ( scf == "DIIS" ){
		if ( mwfn.Wfntype == 0 ){
			EigenMatrix F = mwfn.getFock();
			EigenVector Occ = mwfn.getOccupation() / 2;
			auto [E, epsilons, occupations, C] = RestrictedDIIS(
					T, Mu,
					int2c1e, int4c2e,
					xc, grid,
					F, Occ, Z,
					output-1, nthreads
			);
			E_scf = E;
			mwfn.setEnergy(epsilons);
			mwfn.setOccupation(occupations * 2);
			mwfn.setCoefficientMatrix(C);
		}else if ( mwfn.Wfntype == 1 ){
			EigenMatrix Fa = mwfn.getFock(1);
			EigenMatrix Fb = mwfn.getFock(2);
			EigenVector Occa = mwfn.getOccupation(1);
			EigenVector Occb = mwfn.getOccupation(2);
			auto [E, epsa, epsb, occa, occb, Ca, Cb] = UnrestrictedDIIS(
					T, Mu,
					int2c1e, int4c2e,
					Fa, Fb,
					Occa, Occb,
					Z, Z,
					output-1, nthreads
			);
			E_scf = E;
			mwfn.setEnergy(epsa, 1);
			mwfn.setEnergy(epsb, 2);
			mwfn.setOccupation(occa, 1);
			mwfn.setOccupation(occb, 2);
			mwfn.setCoefficientMatrix(Ca, 1);
			mwfn.setCoefficientMatrix(Cb, 2);
		}
	}
	return E_scf;
}
