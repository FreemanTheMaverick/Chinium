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
#include <Maniverse/Optimizer/LBFGS.h>
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
#include "AugmentedRoothaanHall.h"

#define S (int2c1e.Overlap)
#define Hcore (int2c1e.Kinetic + int2c1e.Nuclear )

std::tuple<double, EigenVector, EigenMatrix> RestrictedDIIS(
		int nocc,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix F, EigenMatrix Z,
		int output, int nthreads){
	double oldE = 0;
	double E = 0;
	const int nbasis = F.cols();
	EigenVector epsilons = EigenZero(Z.cols(), 1);
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
		const EigenMatrix D_ = C.leftCols(nocc) * C.leftCols(nocc).transpose();
		const auto [Ghf_, _, __] = int4c2e.ContractInts(D_, EigenZero(0, 0), EigenZero(0, 0), nthreads, 1);
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(nbasis, nbasis);
		if (xc){
			grid.getDensity({2 * D_});
			xc.Evaluate("ev", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock()[0];
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix Fnew_ = Fhf_ + Gxc_;
		E += (D_ * ( Hcore + Fhf_ )).trace() + Exc_;
		if (output>0){
			std::printf("Electronic energy = %.10f\n", E);
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
	ADIIS adiis(&update_func, 1, 20, 1e-1, 300, output>0 ? 2 : 0);
	if ( !adiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	CDIIS cdiis(&update_func, 1, 20, 1e-6, 300, output>0 ? 2 : 0);
	cdiis.Steal(adiis);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilons, C);
}

static std::function<EigenMatrix (EigenMatrix)> Preconditioner(EigenMatrix D, EigenMatrix A){
	return [D, A](EigenMatrix Z){
		EigenMatrix W = D * Z - Z * D;
		W = W.cwiseProduct(A);
		return ( D * W - W * D ).eval();
	};
}

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenMatrix> RestrictedRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Dprime, EigenMatrix Z,
		int output, int nthreads){
	double E = 0;
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	// ARH hessian related
	AugmentedRoothaanHall arh;
	if constexpr ( scf_t == arh_t ) arh.Init(20, 1);

	Maniverse::Grassmann grassmann(Dprime);
	Maniverse::Iterate M({grassmann.Clone()}, 1);
	Maniverse::PreconFunc dfunc_newton = [&](std::vector<EigenMatrix> Dprimes_, int /*order*/){
		const EigenMatrix Dprime_ = Dprimes_[0];
		const EigenMatrix D_ = Z * Dprime_ * Z.transpose();
		const auto [Ghf_, _, __] = int4c2e.ContractInts(D_, EigenZero(0, 0), EigenZero(0, 0), nthreads, 1);
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(Ghf_.rows(), Ghf_.cols());
		if (xc){
			grid.getDensity({2 * D_});
			xc.Evaluate("ev", grid);
			if constexpr ( scf_t == newton_t ) xc.Evaluate("f", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock()[0];
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix F_ = Fhf_ + Gxc_;
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z; // Euclidean gradient

		// ARH hessian related
		if constexpr ( scf_t == arh_t ) arh.Append(Dprime_, Fprime_);

		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		const double E_ = 0.5 * ( ( D_ * ( Hcore + Fhf_ ) ).trace() + Exc_);

		const int nocc = std::lround(Dprime.diagonal().sum());
		const int nbasis = Dprime.rows();
		EigenMatrix A = EigenMatrix::Ones(nbasis, nbasis);
		for ( int o = 0; o < nocc; o++ ){
			for ( int v = nocc; v < nbasis; v++ ){
				A(o, v) = A(v, o) = 2 * ( epsilons(v) - epsilons(o) );
			}
		}
		if constexpr ( scf_t == lbfgs_t ){
			const EigenMatrix Asqrt = A.cwiseSqrt();
			const EigenMatrix Asqrtinv = Asqrt.cwiseInverse();
			const std::function<EigenMatrix (EigenMatrix)> Psqrt = Preconditioner(Dprime_, Asqrtinv);
			const std::function<EigenMatrix (EigenMatrix)> Psqrtinv = Preconditioner(Dprime_, Asqrt);
			return std::make_tuple(
				E_,
				std::vector<EigenMatrix>{Fprime_},
				std::vector<std::function<EigenMatrix (EigenMatrix)>>{Psqrt},
				std::vector<std::function<EigenMatrix (EigenMatrix)>>{Psqrtinv}
			);
		}else if constexpr ( scf_t == newton_t || scf_t == arh_t){
			std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix vprime){ return vprime; };
			if constexpr ( scf_t == newton_t ) He = [Z, &int4c2e, &grid, &xc, nthreads](EigenMatrix vprime){
				const EigenMatrix v = Z * vprime * Z.transpose();
				const auto [FhfU, _, __] = int4c2e.ContractInts(v, EigenZero(0, 0), EigenZero(0, 0), nthreads, 0);
				EigenMatrix FxcU = EigenZero(vprime.rows(), vprime.cols());
				if (xc){
					grid.getDensityU({{2*v}});
					FxcU = grid.getFockU<u_t>()[0][0];
				}
				const EigenMatrix FU = FhfU + FxcU;
				return (Z.transpose() * FU * Z).eval();
			};
			else He = [&arh](EigenMatrix vprime){ return arh.Hessian(vprime); };
			const EigenMatrix Ainv = A.cwiseInverse();
			const std::function<EigenMatrix (EigenMatrix)> Pr = Preconditioner(Dprime_, Ainv);
			return std::make_tuple(
					E_,
					std::vector<EigenMatrix>{Fprime_},
					std::vector<std::function<EigenMatrix (EigenMatrix)>>{He},
					std::vector<std::function<EigenMatrix (EigenMatrix)>>{Pr}
			);
		}
	};

	const std::tuple<double, double, double> tol = {1.e-8, 1.e-5, 1.e-5};
	if constexpr ( scf_t == lbfgs_t ){
		if ( ! Maniverse::LBFGS(
					dfunc_newton, tol,
					20, 300, 0.1, 0.75, 100,
					E, M, output
		) ) throw std::runtime_error("Convergence failed!");
	}else{
		double ratio = 0;
		if constexpr ( scf_t == newton_t ) ratio = 0.001;
		else ratio = 0.01;
		Maniverse::TrustRegionSetting tr_setting;
		if ( ! Maniverse::TrustRegion(
					dfunc_newton, tr_setting, tol,
					ratio, 1, 300, E, M, output
		) ) throw std::runtime_error("Convergence failed!");
	}
	return std::make_tuple(E, epsilons, C);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedLBFGS(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Dprime, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedRiemann<lbfgs_t>(int2c1e, int4c2e, xc, grid, Dprime, Z, output, nthreads);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedNewton(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Dprime, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedRiemann<newton_t>(int2c1e, int4c2e, xc, grid, Dprime, Z, output, nthreads);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Dprime, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedRiemann<arh_t>(int2c1e, int4c2e, xc, grid, Dprime, Z, output, nthreads);
}
