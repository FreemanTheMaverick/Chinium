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

std::tuple<double, EigenVector, EigenVector, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Fa, EigenMatrix Fb,
		EigenVector Occa, EigenVector Occb,
		EigenMatrix Za, EigenMatrix Zb,
		int output, int nthreads){
	double oldE = 0;
	double E = 0;
	const int nbasis = Fa.rows();
	EigenVector epsa = EigenZero(Za.cols(), 1);
	EigenVector epsb = EigenZero(Zb.cols(), 1);
	EigenVector occa = Occa;
	EigenVector occb = Occb;
	EigenMatrix Ca = EigenZero(Za.rows(), Za.cols());
	EigenMatrix Cb = EigenZero(Zb.rows(), Zb.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
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
		const auto [_, Ghfa_, Ghfb_] = int4c2e.ContractInts(EigenZero(0, 0), Da_, Db_, nthreads, 1);
		const EigenMatrix Fhfa_ = Hcore + Ghfa_;
		const EigenMatrix Fhfb_ = Hcore + Ghfb_;
		double Exc_ = 0;
		EigenMatrix Gxca_ = EigenZero(nbasis, nbasis);
		EigenMatrix Gxcb_ = EigenZero(nbasis, nbasis);
		if (xc){
			grid.getDensity({Da_, Db_});
			xc.Evaluate("ev", grid);
			Exc_ = grid.getEnergy();
			std::vector<EigenMatrix> Gxc_ = grid.getFock();
			Gxca_ = Gxc_[0];
			Gxcb_ = Gxc_[1];
		}
		const EigenMatrix Fnewa_ = Fhfa_ + Gxca_;
		const EigenMatrix Fnewb_ = Fhfb_ + Gxcb_;

		E += 0.5 * ( Dot(Da_, Hcore + Fhfa_) + Dot(Db_, Hcore + Fhfb_) ) + Exc_;
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
	ADIIS adiis(&update_func, 1, 20, 1e-1, 300, output>0 ? 2 : 0);
	if ( T == 0 ) if ( !adiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	CDIIS cdiis(&update_func, 1, 20, 1e-6, 300, output>0 ? 2 : 0);
	if ( T > 0 ) cdiis.Damps.push_back(std::make_tuple(0.1, 100, 0.75));
	cdiis.Steal(adiis);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsa, epsb, occa, occb, Ca, Cb);
}

namespace{

#undef S
#undef Hcore
#define S (int2c1e->Overlap)
#define Hcore (int2c1e->Kinetic + int2c1e->Nuclear )

class ObjBase: public Maniverse::Objective{ public:
	Int2C1E* int2c1e;
	Int4C2E* int4c2e;
	ExchangeCorrelation* xc;
	Grid* grid;
	int nocc1;
	int nocc2;
	EigenMatrix Z1;
	EigenMatrix Z2;
	int nthreads;

	int nbasis;

	EigenMatrix D1prime;
	EigenMatrix D2prime;
	EigenMatrix F1prime;
	EigenMatrix F2prime;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	EigenVector epsilon1s;
	EigenVector epsilon2s;
	EigenMatrix C1;
	EigenMatrix C2;
	EigenMatrix A1;
	EigenMatrix A2;

	ObjBase(
			Int2C1E& int2c1e, Int4C2E& int4c2e,
			ExchangeCorrelation& xc, Grid& grid,
			int nocc1, int nocc2, EigenMatrix Z1, EigenMatrix Z2,
			int nthreads
	): int2c1e(&int2c1e), int4c2e(&int4c2e), xc(&xc), grid(&grid), nocc1(nocc1), nocc2(nocc2), Z1(Z1), Z2(Z2), nthreads(nthreads){
		nbasis = Z1.rows();
		epsilon1s = epsilon2s = EigenZero(nbasis, 1);
		C1 = C2 = A1 = A2 = EigenZero(nbasis, nbasis);
	};

	virtual void Calculate(std::vector<EigenMatrix> Dprimes, int /*derivative*/) override{
		D1prime = Dprimes[0];
		D2prime = Dprimes[1];
		const EigenMatrix D1 = Z1 * D1prime * Z1.transpose();
		const EigenMatrix D2 = Z2 * D2prime * Z2.transpose();
		const auto [_, Ghf1, Ghf2] = int4c2e->ContractInts(EigenZero(0, 0), D1, D2, nthreads, 1);
		double Exc = 0;
		EigenMatrix Gxc1 = EigenZero(Z1.rows(), Z1.rows());
		EigenMatrix Gxc2 = EigenZero(Z1.rows(), Z1.rows());
		if (*xc){
			grid->getDensity({ D1, D2 });
			xc->Evaluate("ev", *grid);
			Exc = grid->getEnergy();
			const std::vector<EigenMatrix> Gxc = grid->getFock();
			Gxc1 = Gxc[0];
			Gxc2 = Gxc[1];
		}
		const EigenMatrix Fhf1 = Hcore + Ghf1;
		const EigenMatrix Fhf2 = Hcore + Ghf2;
		const EigenMatrix F1 = Fhf1 + Gxc1;
		const EigenMatrix F2 = Fhf2 + Gxc2;
		F1prime = Z1.transpose() * F1 * Z1; // Euclidean gradient
		F2prime = Z2.transpose() * F2 * Z2;

		Value = 0.5 * ( Dot(D1, Hcore + Fhf1) + Dot(D2, Hcore + Fhf2) ) + Exc;
		Gradient = { F1prime, F2prime };

		eigensolver.compute(F1prime);
		epsilon1s = eigensolver.eigenvalues();
		C1 = Z1 * eigensolver.eigenvectors();
		eigensolver.compute(F2prime);
		epsilon2s = eigensolver.eigenvalues();
		C2 = Z2 * eigensolver.eigenvectors();

		A1 = EigenMatrix::Ones(nbasis, nbasis);
		for ( int o = 0; o < nocc1; o++ ){
			for ( int v = nocc1; v < nbasis; v++ ){
				A1(o, v) = A1(v, o) = 2 * ( epsilon1s(v) - epsilon1s(o) );
			}
		}
		A2 = EigenMatrix::Ones(nbasis, nbasis);
		for ( int o = 0; o < nocc2; o++ ){
			for ( int v = nocc2; v < nbasis; v++ ){
				A2(o, v) = A2(v, o) = 2 * ( epsilon2s(v) - epsilon2s(o) );
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
	EigenMatrix A1sqrt;
	EigenMatrix A2sqrt;
	EigenMatrix A1sqrtinv;
	EigenMatrix A2sqrtinv;

	using ObjBase::ObjBase;

	void Calculate(std::vector<EigenMatrix> Dprimes, int /*derivative*/) override{
		ObjBase::Calculate(Dprimes, 2);
		A1sqrt = A1.cwiseSqrt();
		A2sqrt = A2.cwiseSqrt();
		A1sqrtinv = A1sqrt.cwiseInverse();
		A2sqrtinv = A2sqrt.cwiseInverse();
	};

	std::vector<std::vector<EigenMatrix>> PreconditionerSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<std::vector<EigenMatrix>>{
			{ ::Preconditioner(D1prime, A1sqrtinv, Vs[0]), EigenZero(nbasis, nbasis) },
			{ EigenZero(nbasis, nbasis), ::Preconditioner(D2prime, A2sqrtinv, Vs[1]) },
		};
	}

	std::vector<std::vector<EigenMatrix>> PreconditionerInvSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<std::vector<EigenMatrix>>{
			{ ::Preconditioner(D1prime, A1sqrt, Vs[0]), EigenZero(nbasis, nbasis) },
			{ EigenZero(nbasis, nbasis), ::Preconditioner(D2prime, A2sqrt, Vs[1]) },
		};
	};
};

class ObjNewtonBase: public ObjBase{ public:
	EigenMatrix A1inv;
	EigenMatrix A2inv;

	using ObjBase::ObjBase;

	virtual void Calculate(std::vector<EigenMatrix> Dprimes, int /*derivative*/) override{
		ObjBase::Calculate(Dprimes, 2);
		A1inv = A1.cwiseInverse();
		A2inv = A2.cwiseInverse();
	};

	std::vector<std::vector<EigenMatrix>> Preconditioner(std::vector<EigenMatrix> Vs) const override{
		return std::vector<std::vector<EigenMatrix>>{
			{ ::Preconditioner(D1prime, A1inv, Vs[0]), EigenZero(nbasis, nbasis) },
			{ EigenZero(nbasis, nbasis), ::Preconditioner(D2prime, A2inv, Vs[1]) },
		};
	};
};

class ObjNewton: public ObjNewtonBase{ public:
	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> Dprimes, int /*derivative*/) override{
		ObjNewtonBase::Calculate(Dprimes, 2);
		if (*xc) xc->Evaluate("f", *grid);
	};

	std::vector<std::vector<EigenMatrix>> Hessian(std::vector<EigenMatrix> Vprimes) const override{
		const EigenMatrix V1prime = Vprimes[0];
		const EigenMatrix V2prime = Vprimes[1];
		const EigenMatrix V1 = Z1 * V1prime * Z1.transpose();
		const EigenMatrix V2 = Z2 * V2prime * Z2.transpose();
		auto [_, Gtmp1, Gtmp2] = int4c2e->ContractInts(EigenZero(0, 0), V1, V2, nthreads, 0);
		if (*xc){
			grid->getDensityU({ {V1}, {V2} });
			const std::vector<std::vector<EigenMatrix>> Gtmpxc = grid->getFockU<u_t>();
			Gtmp1 += Gtmpxc[0][0];
			Gtmp2 += Gtmpxc[1][0];
		}
		return std::vector<std::vector<EigenMatrix>>{
			{ EigenZero(nbasis, nbasis), Z2.transpose() * Gtmp1 * Z2 },
			{ Z1.transpose() * Gtmp2 * Z1, EigenZero(nbasis, nbasis) }
		};
	};
};

class ObjARH: public ObjNewtonBase{ public:
	AugmentedRoothaanHall arh = AugmentedRoothaanHall(20, 1);

	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> Dprimes, int /*derivative*/) override{
		ObjNewtonBase::Calculate(Dprimes, 2);
		EigenMatrix Dprime = EigenZero(nbasis, 2 * nbasis);
		Dprime << D1prime, D2prime;
		EigenMatrix Fprime = EigenZero(nbasis, 2 * nbasis);
		Fprime << F1prime, F2prime;
		arh.Append(Dprime, Fprime);
	};

	std::vector<std::vector<EigenMatrix>> Hessian(std::vector<EigenMatrix> Vprimes) const override{
		EigenMatrix Vprime = EigenZero(nbasis, 2 * nbasis);
		Vprime << Vprimes[0], Vprimes[1];
		const EigenMatrix HVprime = arh.Hessian(Vprime);
		return std::vector<std::vector<EigenMatrix>>{
			{ EigenZero(nbasis, nbasis), HVprime.leftCols(nbasis) },
			{ HVprime.rightCols(nbasis), EigenZero(nbasis, nbasis) }
		};
	};
};

} // namespace

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc1, int nocc2,
		EigenMatrix Z1, EigenMatrix Z2,
		int nthreads, int output){
	std::conditional_t< scf_t == lbfgs_t,
				ObjLBFGS,
				std::conditional_t< scf_t == newton_t,
							ObjNewton,
							ObjARH
				>
	> obj(int2c1e, int4c2e, xc, grid, nocc1, nocc2, Z1, Z2, nthreads);
	EigenMatrix D1prime = EigenZero(Z1.cols(), Z1.cols());
	for ( int i = 0; i < nocc1; i++ ) D1prime(i, i) = 1;
	Maniverse::Grassmann grassmann1(D1prime);
	EigenMatrix D2prime = EigenZero(Z1.cols(), Z1.cols());
	for ( int i = 0; i < nocc2; i++ ) D2prime(i, i) = 1;
	Maniverse::Grassmann grassmann2(D2prime);
	Maniverse::Iterate M(obj, {grassmann1.Share(), grassmann2.Share()}, 1);
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
	return std::make_tuple(obj.Value, obj.epsilon1s, obj.epsilon2s, obj.C1, obj.C2);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedLBFGS(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc1, int nocc2,
		EigenMatrix Z1, EigenMatrix Z2,
		int nthreads, int output){
	return UnrestrictedRiemann<lbfgs_t>(int2c1e, int4c2e, xc, grid, nocc1, nocc2, Z1, Z2, nthreads, output);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedNewton(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc1, int nocc2,
		EigenMatrix Z1, EigenMatrix Z2,
		int nthreads, int output){
	return UnrestrictedRiemann<newton_t>(int2c1e, int4c2e, xc, grid, nocc1, nocc2, Z1, Z2, nthreads, output);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc1, int nocc2,
		EigenMatrix Z1, EigenMatrix Z2,
		int nthreads, int output){
	return UnrestrictedRiemann<arh_t>(int2c1e, int4c2e, xc, grid, nocc1, nocc2, Z1, Z2, nthreads, output);
}
