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
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
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

#undef S
#undef Hcore
#define S (int2c1e->Overlap)
#define Hcore (int2c1e->Kinetic + int2c1e->Nuclear )

namespace{

class ObjBase: public Maniverse::Objective{ public:
	Int2C1E* int2c1e;
	Int4C2E* int4c2e;
	ExchangeCorrelation* xc;
	Grid* grid;
	int nocc;
	EigenMatrix Z;
	int nthreads;

	int nbasis;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	EigenMatrix Dprime;
	EigenMatrix Fprime;
	EigenVector epsilons;
	EigenMatrix C;
	EigenMatrix A;

	ObjBase(
			Int2C1E& int2c1e, Int4C2E& int4c2e,
			ExchangeCorrelation& xc, Grid& grid,
			int nocc, EigenMatrix Z,
			int nthreads
	): int2c1e(&int2c1e), int4c2e(&int4c2e), xc(&xc), grid(&grid), nocc(nocc), Z(Z), nthreads(nthreads){
		nbasis = Z.rows();
		epsilons = EigenZero(nbasis, 1);
		C = A = EigenZero(nbasis, nbasis);
	};

	virtual void Calculate(std::vector<EigenMatrix> Dprimes, std::vector<int> derivatives) override{
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			Dprime = Dprimes[0];
			const EigenMatrix D = Z * Dprime * Z.transpose();
			const auto [Ghf, _, __] = int4c2e->ContractInts(D, EigenZero(0, 0), EigenZero(0, 0), nthreads, 1);
			double Exc = 0;
			EigenMatrix Gxc = EigenZero(nbasis, nbasis);
			if (*xc){
				grid->getDensity({ 2 * D });
				xc->Evaluate("ev", *grid);
				Exc = grid->getEnergy();
				Gxc = grid->getFock()[0];
			}
			const EigenMatrix Fhf = Hcore + Ghf;
			const EigenMatrix F = Fhf + Gxc;
			Value = 0.5 * ( ( D * ( Hcore + Fhf ) ).trace() + Exc );
			Fprime = Z.transpose() * F * Z; // Euclidean gradient
		}
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Gradient = { Fprime };
			eigensolver.compute(Fprime);
			epsilons = eigensolver.eigenvalues();
			C = Z * eigensolver.eigenvectors();
			A = EigenMatrix::Ones(nbasis, nbasis);
			for ( int o = 0; o < nocc; o++ ){
				for ( int v = nocc; v < nbasis; v++ ){
					A(o, v) = A(v, o) = 2 * ( epsilons(v) - epsilons(o) );
				}
			}
		}
	};
};

EigenMatrix Preconditioner(EigenMatrix D, EigenMatrix A, EigenMatrix V){
	EigenMatrix W = D * V - V * D;
	W = W.cwiseProduct(A);
	return D * W - W * D;
}

class ObjLBFGS: public ObjBase{ public:
	EigenMatrix Asqrt;
	EigenMatrix Asqrtinv;

	using ObjBase::ObjBase;

	void Calculate(std::vector<EigenMatrix> Dprimes, std::vector<int> derivatives) override{
		ObjBase::Calculate(Dprimes, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Asqrt = A.cwiseSqrt();
			Asqrtinv = Asqrt.cwiseInverse();
		}
	};

	std::vector<EigenMatrix> PreconditionerSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{ ::Preconditioner(Dprime, Asqrtinv, Vs[0]) };
	};

	std::vector<EigenMatrix> PreconditionerInvSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{ ::Preconditioner(Dprime, Asqrt, Vs[0]) };
	};
};

class ObjNewtonBase: public ObjBase{ public:
	EigenMatrix Ainv;

	using ObjBase::ObjBase;

	virtual void Calculate(std::vector<EigenMatrix> Dprimes, std::vector<int> derivatives) override{
		ObjBase::Calculate(Dprimes, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) ){
			Ainv = A.cwiseInverse();
		}
	};

	std::vector<EigenMatrix> Preconditioner(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{ ::Preconditioner(Dprime, Ainv, Vs[0]) };
	};
};

class ObjNewton: public ObjNewtonBase{ public:
	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> Dprimes, std::vector<int> derivatives) override{
		ObjNewtonBase::Calculate(Dprimes, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) && *xc ){
			xc->Evaluate("f", *grid);
		}
	};

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> Vprimes) const override{
		const EigenMatrix Vprime = Vprimes[0];
		const EigenMatrix V = Z * Vprime * Z.transpose();
		const auto [FhfU, _, __] = int4c2e->ContractInts(V, EigenZero(0, 0), EigenZero(0, 0), nthreads, 0);
		EigenMatrix FxcU = EigenZero(nbasis, nbasis);
		if (*xc){
			grid->getDensityU({{ 2 * V }});
			FxcU = grid->getFockU<u_t>()[0][0];
		}
		const EigenMatrix FU = FhfU + FxcU;
		return std::vector<EigenMatrix>{ Z.transpose() * FU * Z };
	};
};

class ObjARH: public ObjNewtonBase{ public:
	AugmentedRoothaanHall arh = AugmentedRoothaanHall(20, 1);

	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> Dprimes, std::vector<int> derivatives) override{
		ObjNewtonBase::Calculate(Dprimes, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			arh.Append(Dprime, Fprime);
		}
	};

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> Vprimes) const override{
		return std::vector<EigenMatrix>{ arh.Hessian(Vprimes[0]) };
	};
};

} // namespace

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenMatrix> RestrictedRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc, EigenMatrix Z,
		int nthreads, int output){
	std::conditional_t< scf_t == lbfgs_t,
				ObjLBFGS,
				std::conditional_t< scf_t == newton_t,
							ObjNewton,
							ObjARH
				>
	> obj(int2c1e, int4c2e, xc, grid, nocc, Z, nthreads);
	EigenMatrix Dprime = EigenZero(Z.cols(), Z.cols());
	for ( int i = 0; i < nocc; i++ ) Dprime(i, i) = 1;
	Maniverse::Grassmann grassmann(Dprime);
	Maniverse::Iterate M(obj, {grassmann.Share()}, 1);
	std::tuple<double, double, double> tol = {1.e-8, 1.e-5, 1.e-5};
	if constexpr ( scf_t == lbfgs_t ){
		if ( ! Maniverse::LBFGS(
					M, tol,
					20, 300, 0.1, 0.75, 100, output
		) ) throw std::runtime_error("Convergence failed!");
	}else{
		Maniverse::TrustRegion tr;
		if constexpr ( scf_t == newton_t ){
			if ( ! Maniverse::TruncatedNewton(
						M, tr, tol,
						0.001, 300, output
			) ) throw std::runtime_error("Convergence failed!");
		}else{
			if ( ! Maniverse::TruncatedNewton(
						M, tr, tol,
						0.01, 300, output
			) ) throw std::runtime_error("Convergence failed!");
		}
	}
	return std::make_tuple(obj.Value, obj.epsilons, obj.C);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedLBFGS(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc, EigenMatrix Z,
		int nthreads, int output){
	return RestrictedRiemann<lbfgs_t>(int2c1e, int4c2e, xc, grid, nocc, Z, nthreads, output);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedNewton(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc, EigenMatrix Z,
		int nthreads, int output){
	return RestrictedRiemann<newton_t>(int2c1e, int4c2e, xc, grid, nocc, Z, nthreads, output);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc, EigenMatrix Z,
		int nthreads, int output){
	return RestrictedRiemann<arh_t>(int2c1e, int4c2e, xc, grid, nocc, Z, nthreads, output);
}
