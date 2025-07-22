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

std::tuple<double, EigenVector, EigenMatrix> RestrictedGrassmannARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Dprime, EigenMatrix Z,
		int output, int nthreads){
	double E = 0;
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	// ARH hessian related
	std::deque<EigenMatrix> Dprimes;
	std::deque<EigenMatrix> Fprimes;

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

		// ARH hessian related
		Dprimes.push_back(Dprime_);
		Fprimes.push_back(Fprime_);
		if ( Dprimes.size() > 20 ){
			Dprimes.pop_front();
			Fprimes.pop_front();
		}

		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		const double E_ = 0.5 * (( D_ * ( Hcore + Fhf_ ) ).trace() + Exc_);
		std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix vprime){ return vprime; };
		if ( order == 2 && Dprimes.size() > 1 ){
			if (output>0) std::printf("Using augmented Roothaan-Hall hessian with %d previous iterations\n", (int)Dprimes.size());
			const int size = (int)Dprimes.size() - 1;
			const int nbasis = Z.rows();
			std::vector<EigenMatrix> Ddiff(size, EigenZero(nbasis, nbasis));
			std::vector<EigenMatrix> Fdiff(size, EigenZero(nbasis, nbasis));
			for ( int i = 0; i < size; i++ ){
				Ddiff[i] = Dprimes[i] - Dprime_;
				Fdiff[i] = Fprimes[i] - Fprime_;
			}
			EigenMatrix T = EigenZero(size, size);
			for ( int i = 0; i < size; i++ ) for ( int j = i; j < size; j++ ){
				T(i, j) = T(j, i) = Ddiff[i].cwiseProduct(Ddiff[j]).sum();
			}
			const EigenMatrix Tinv = T.inverse();
			He = [nbasis, size, Ddiff, Fdiff, Tinv](EigenMatrix vprime){
				std::vector<double> TrDsv(size);
				for ( int j = 0; j < size; j++ )
					TrDsv[j] = Ddiff[j].cwiseProduct(vprime).sum();
				EigenMatrix Hv = EigenZero(nbasis, nbasis);
				for ( int i = 0; i < size; i++ ){
					double coeff = 0;
					for ( int j = 0; j < size; j++ ){
						coeff += Tinv(i, j) * TrDsv[j];
					}
					Hv += Fdiff[i] * coeff;
				}
				return (EigenMatrix)Hv;
			};
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
				0.01, 1, 100, E, M, output
	) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilons, C);
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


