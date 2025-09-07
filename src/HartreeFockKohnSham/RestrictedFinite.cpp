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
#include <Maniverse/Manifold/RealSymmetric.h>
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/Manifold/Euclidean.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/TrustRegion.h>
#include <libmwfn.h>

#include "../Macro.h"
#include "../Integral/Int2C1E.h"
#include "../Integral/Int4C2E.h"
#include "../Grid/Grid.h"
#include "../ExchangeCorrelation.h"
#include "../DIIS/CDIIS.h"
#include "AugmentedRoothaanHall.h"

#define S (int2c1e.Overlap)
#define Hcore (int2c1e.Kinetic + int2c1e.Nuclear )

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteDIIS(
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
	EigenMatrix C = EigenOne(Z.rows(), Z.cols());
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
		const EigenArray ns = 1. / ( 1. + ( ( epsilons.array() - Mu ) / T ).exp() );
		occupations = (EigenVector)ns;
		E = 2 * (
				T * (
					ns.pow(ns).log() + ( 1. - ns ).pow( 1. - ns ).log()
				).sum()
				- Mu * ns.sum()
		);
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
			std::printf("Electronic grand potential = %.10f\n", E);
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
	CDIIS cdiis(&update_func, 1, 20, 1e-8, 100, output>0 ? 2 : 0);
	cdiis.Damps.push_back(std::make_tuple(0.1, 100, 0.75));
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilons, occupations, C);
}

EigenVector FermiDirac(EigenVector epsilons, double T, double Mu, int order){
	EigenArray ns = 1. / ( 1. + ( ( epsilons.array() - Mu ) / T ).exp() );
	EigenArray res = ns / std::pow( - T, order );
	for ( int i = 1; i <= order; i++ ) res *= 1. - i * ns;
	return res.matrix();
}

#define IdentityFunc [](EigenMatrix v){ return v; }
#define DummyFunc(_size_) [_size_](EigenMatrix /*v*/){ return EigenZero(_size_, _size_); }

#define lbfgs_t 114514
#define newton_t 1919
#define arh_t 810
template <int scf_t>
std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteRiemann(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenVector occ_guess, EigenMatrix Z,
		int output, int nthreads){
	double E = 0;
	const int nbasis = Z.cols();
	EigenMatrix C = EigenZero(nbasis, nbasis);
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	// ARH hessian related
	AugmentedRoothaanHall arh;
	if constexpr ( scf_t == arh_t ) arh.Init(20, 1);

	// Exact hessian related
	EigenMatrix FoverC = EigenZero(1, 1);
	EigenMatrix FoverOcc = EigenZero(1, 1);
	if constexpr ( scf_t == newton_t ){
		FoverC.resize(nbasis, nbasis);
		FoverOcc.resize(nbasis, nbasis);
	}

	EigenMatrix all_occ = occ_guess;
	all_occ.resize(nbasis, 1);
	int ni = 0; int na = 0;
	for ( int i = 0; i < all_occ.size(); i++ ){
		if ( all_occ(i, 0) > 1. - 1e-6 ) ni++;
		else if ( all_occ(i, 0) > 1e-6 ) na++;
	}
	int nv = nbasis - ni - na;

	EigenVector epsilons = EigenZero(nbasis, 1);

	Maniverse::PreconFunc cfunc_newton = [&](std::vector<EigenMatrix> Xs, int /*order*/){
		const EigenMatrix Cprime_ = Xs[0];
		const EigenMatrix Cprime_thin = Cprime_(Eigen::all, Eigen::seqN(ni, na));
		C = Z * Cprime_;
		const EigenMatrix Occ = Xs[1];
		if (output>0){
			std::printf("Fractional occupation:");
			for ( int i = 0; i < Occ.size(); i++ ) std::printf(" %f", Occ(i, 0));
			std::printf("\n");
		}

		all_occ(Eigen::seqN(ni, na), 0) = Occ;
		const EigenMatrix Dprime_ = Cprime_ * all_occ.asDiagonal() * Cprime_.transpose();
		const EigenMatrix D_ = Z * Dprime_ * Z.transpose();
		const EigenMatrix Ghf_ = int4c2e.ContractInts(D_, nthreads, 1);
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(D_.rows(), D_.cols());
		if (xc){
			grid.getDensity(2 * D_);
			xc.Evaluate("ev", grid);
			if constexpr ( scf_t == newton_t ) xc.Evaluate("f", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock();
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix F_ = Fhf_ + Gxc_;
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z;
		
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();

		// ARH hessian related
		if constexpr ( scf_t == arh_t ) arh.Append(Dprime_, Fprime_);

		const EigenArray ns = Occ.array();
		const double E_ =
			( D_ * ( Hcore + Fhf_ ) ).trace() + Exc_
			+ 2 * T * ( ns.pow(ns).log() + ( 1. - ns ).pow( 1. - ns ).log() ).sum()
			- 2 * Mu * ( ns.sum() + ni );

		const EigenMatrix GradC = 4 * Fprime_ * Cprime_ * all_occ.asDiagonal();
		const EigenVector CtFC = ( Cprime_.transpose() * Fprime_ * Cprime_ ).diagonal();
		const EigenMatrix GradOcc1 = 2 * CtFC(Eigen::seqN(ni, na));
		const EigenMatrix GradOcc2 = ( - 2 * T * ( 1. / ns - 1. ).log() - 2 * Mu ).matrix();

		EigenMatrix A = EigenMatrix::Ones(nbasis, nbasis);
		for ( int i = 0; i < nbasis; i++ ){
			for ( int j = i; j < nbasis; j++ ){
				const double occ_diff = all_occ(i, 0) - all_occ(j, 0);
				if ( std::abs(occ_diff) > 1e-6 ) A(i, j) = A(j, i) = - 2 * ( epsilons(i) - epsilons(j) ) * occ_diff;
			}
		}
		std::function<EigenMatrix (EigenMatrix)> DummyC = [nbasis](EigenMatrix /*Z*/){ return EigenZero(nbasis, nbasis); };
		std::function<EigenMatrix (EigenMatrix)> DummyOcc = [na](EigenMatrix /*Z*/){ return EigenZero(na, 1); };
		if constexpr ( scf_t == lbfgs_t ){
			const EigenMatrix Asqrt = A.cwiseSqrt();
			const EigenMatrix Asqrtinv = Asqrt.cwiseInverse();
			std::function<EigenMatrix (EigenMatrix)> Psqrt = [Cprime_, Asqrtinv](EigenMatrix Z){
				EigenMatrix Omega = Cprime_.transpose() * Z;
				Omega = Omega.cwiseProduct(Asqrtinv);
				return ( Cprime_ * Omega ).eval();
			};
			std::function<EigenMatrix (EigenMatrix)> Psqrtinv = [Cprime_, Asqrt](EigenMatrix Z){
				EigenMatrix Omega = Cprime_.transpose() * Z;
				Omega = Omega.cwiseProduct(Asqrt);
				return ( Cprime_ * Omega ).eval();
			};
			return std::make_tuple(
				E_,
				std::vector<EigenMatrix>{GradC, GradOcc1 + GradOcc2},
				std::vector<std::function<EigenMatrix (EigenMatrix)>>{Psqrt, DummyC, DummyOcc, IdentityFunc},
				std::vector<std::function<EigenMatrix (EigenMatrix)>>{Psqrtinv, DummyC, DummyOcc, IdentityFunc}
			);
		}else if constexpr ( scf_t == newton_t || scf_t == arh_t){
			std::vector<std::function<EigenMatrix (EigenMatrix)>> He;
			He.push_back([Z, all_occ, Fprime_, Cprime_, &int4c2e, &xc, &grid, nthreads, &arh, &FoverC](EigenMatrix vprime){
					EigenMatrix Dprime = Cprime_ * all_occ.asDiagonal() * vprime.transpose();
					Dprime += Dprime.transpose().eval();
					if constexpr ( scf_t == newton_t ){
						const EigenMatrix D = Z * Dprime * Z.transpose();
						const EigenMatrix FhfU = int4c2e.ContractInts(D, nthreads, 0);
						EigenMatrix FxcU = EigenZero(D.rows(), D.cols());
						if (xc){
							std::vector<Eigen::Tensor<double, 1>> RhoUss, SigmaUss;
							std::vector<Eigen::Tensor<double, 2>> Rho1Uss;
							grid.getDensityU({2*D}, RhoUss, Rho1Uss, SigmaUss);
							FxcU = grid.getFockU(RhoUss, Rho1Uss, SigmaUss)[0];
						}
						const EigenMatrix FU = FhfU + FxcU;
						FoverC = Z.transpose() * FU * Z;
					}else{
						FoverC = arh.Hessian(Dprime);
					}
					const EigenMatrix res = 4 * ( 
							FoverC * Cprime_
							+ Fprime_ * vprime
					) * all_occ.asDiagonal();
					return res;
			});
			He.push_back([ni, na, Z, Cprime_, Cprime_thin, all_occ, Fprime_, &int4c2e, &xc, &grid, nthreads, &arh, &FoverOcc](EigenMatrix gamma){
					const EigenMatrix Dprime = Cprime_thin * gamma.asDiagonal() * Cprime_thin.transpose();
					if constexpr ( scf_t == newton_t ){
						const EigenMatrix D = Z * Dprime * Z.transpose();
						const EigenMatrix FhfU = int4c2e.ContractInts(D, nthreads, 0);
						EigenMatrix FxcU = EigenZero(D.rows(), D.cols());
						if (xc){
							std::vector<Eigen::Tensor<double, 1>> RhoUss, SigmaUss;
							std::vector<Eigen::Tensor<double, 2>> Rho1Uss;
							grid.getDensityU({2*D}, RhoUss, Rho1Uss, SigmaUss);
							FxcU = grid.getFockU(RhoUss, Rho1Uss, SigmaUss)[0];
						}
						const EigenMatrix FU = FhfU + FxcU;
						FoverOcc = Z.transpose() * FU * Z;
					}else{
						FoverOcc = arh.Hessian(Dprime);
					}
					EigenMatrix all_gamma = EigenZero(Dprime.rows(), 1);
					all_gamma(Eigen::seqN(ni, na), 0) = gamma;
					const EigenMatrix res = 4 * (
							Fprime_ * Cprime_ * all_gamma.asDiagonal()
							+ FoverOcc * Cprime_ * all_occ.asDiagonal()
					);
					return res;
			});
			He.push_back([ni, na, Cprime_, Fprime_, &FoverC](EigenMatrix Delta){
					const EigenVector res = 2 * (
							2 * Delta.transpose() * Fprime_ * Cprime_
							+ Cprime_.transpose() * FoverC * Cprime_
					).diagonal();
					return res(Eigen::seqN(ni, na)).eval();
			});
			He.push_back([T, ni, na, Cprime_thin, ns, Fprime_, &FoverOcc](EigenMatrix gamma){
					const EigenVector res = 2 * (
							( Cprime_thin.transpose() * FoverOcc * Cprime_thin ).diagonal()
							- ( T / ns / ( ns - 1. ) ).matrix().cwiseProduct(gamma)
					);
					return res;
			});
			const EigenMatrix Ainv = A.cwiseInverse();
			std::function<EigenMatrix (EigenMatrix)> Pr = [Cprime_, Ainv](EigenMatrix Z){
				EigenMatrix Omega = Cprime_.transpose() * Z;
				Omega = Omega.cwiseProduct(Ainv);
				return ( Cprime_ * Omega ).eval();
			};
			return std::make_tuple(
					E_,
					std::vector<EigenMatrix>{GradC, GradOcc1 + GradOcc2},
					He,
					std::vector<std::function<EigenMatrix (EigenMatrix)>>{Pr, DummyC, DummyOcc, IdentityFunc}
			);
		}
	};

	Maniverse::Flag flag(Cprime);
	std::vector<int> spaces = {ni};
	for ( int i = 0; i < na; i++ ) spaces.push_back(1);
	spaces.push_back(nv);
	flag.setBlockParameters(spaces);
	Maniverse::Euclidean euclidean(all_occ(Eigen::seqN(ni, na), 0));
	Maniverse::Iterate M({flag.Clone(), euclidean.Clone()}, 1);

	const std::tuple<double, double, double> tol = {1.e-8, 1.e-5, 1.e-5};
	if constexpr ( scf_t == lbfgs_t ){
		if ( ! Maniverse::LBFGS(
					cfunc_newton, tol,
					20, 300, E, M, output
		) ) throw std::runtime_error("Convergence failed!");
	}else{
		double ratio = 0;
		if constexpr ( scf_t == newton_t ) ratio = 0.001;
		else ratio = 0.01;
		Maniverse::TrustRegionSetting tr_setting;
		if ( ! Maniverse::TrustRegion(
					cfunc_newton, tr_setting, tol,
					ratio, 1, 300, E, M, output
		) ) throw std::runtime_error("Convergence failed!");
	}
	return std::make_tuple(E, epsilons, all_occ, C);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteLBFGS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenVector epsilons, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedFiniteRiemann<lbfgs_t>(T, Mu, int2c1e, int4c2e, xc, grid, Cprime, epsilons, Z, output, nthreads);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteNewton(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenVector epsilons, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedFiniteRiemann<newton_t>(T, Mu, int2c1e, int4c2e, xc, grid, Cprime, epsilons, Z, output, nthreads);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteARH(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenVector epsilons, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedFiniteRiemann<arh_t>(T, Mu, int2c1e, int4c2e, xc, grid, Cprime, epsilons, Z, output, nthreads);
}
