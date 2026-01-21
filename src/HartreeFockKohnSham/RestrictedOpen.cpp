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
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <libmwfn.h>

#include "../Macro.h"
#include "../Integral/Int2C1E.h"
#include "../Integral/Int4C2E.h"
#include "../Grid/Grid.h"
#include "../ExchangeCorrelation.h"
#include "AugmentedRoothaanHall.h"

#define S (int2c1e->Overlap)
#define Hcore (int2c1e->Kinetic + int2c1e->Nuclear )

namespace{

class ObjBase: public Maniverse::Objective{ public:
	Int2C1E* int2c1e;
	Int4C2E* int4c2e;
	ExchangeCorrelation* xc;
	Grid* grid;
	int nd;
	int na;
	int nb;
	EigenMatrix Z;
	int nthreads;

	int nbasis;
	std::vector<int> types;
	std::vector<int> space_sizes;

	EigenMatrix Cprime;
	EigenMatrix Cprime_perp;
	std::vector<EigenMatrix> Cprimes;
	std::vector<EigenMatrix> Dprimes;
	std::vector<EigenMatrix> Fprimes;
	EigenVector epsilons;
	EigenMatrix C;
	EigenMatrix K;
	EigenMatrix L;

	#define ntypes (int)types.size()
	#define type types[itype]
	ObjBase(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nd, int na, int nb, EigenMatrix Z,
		int nthreads
	): int2c1e(&int2c1e), int4c2e(&int4c2e), xc(&xc), grid(&grid), nd(nd), na(na), nb(nb), Z(Z), nthreads(nthreads){
		nbasis = Z.rows();
		space_sizes = {nd, na, nb};
		types = {};
		if (nd) types.push_back(0);
		if (na) types.push_back(1);
		if (nb) types.push_back(2);
		Cprimes.resize(ntypes);
		Dprimes.resize(ntypes);
		Fprimes.resize(ntypes);
		epsilons = EigenZero(nbasis, 1);
	};

	virtual void Calculate(std::vector<EigenMatrix> Cprimes_, int /*derivative*/) override{
		Cprime = Cprimes_[0];
		std::vector<EigenMatrix> Ds(3, EigenZero(0, 0));
		int ncols = 0;
		for ( int itype = 0; itype < ntypes; itype++ ){
			Cprimes[itype] = Cprime(Eigen::placeholders::all, Eigen::seqN(ncols, space_sizes[type]));
			ncols += space_sizes[type];
			Dprimes[itype] = Cprimes[itype] * Cprimes[itype].transpose();
			Ds[type].resize(nbasis, nbasis);
			Ds[type] = Z * Dprimes[itype] * Z.transpose();
		}
		std::vector<EigenMatrix> Ghfs(3, EigenZero(nbasis, nbasis));
		std::tie(Ghfs[0], Ghfs[1], Ghfs[2]) = int4c2e->ContractInts(Ds[0], Ds[1], Ds[2], nthreads, 1);
		std::vector<EigenMatrix> Fhfs(ntypes, EigenZero(nbasis, nbasis));
		for ( int itype = 0; itype < ntypes; itype++ ){
			Fhfs[itype] = Hcore + Ghfs[type];
		}

		Ds[0] *= 2;
		Ds.erase(
				std::remove_if(
					Ds.begin(), Ds.end(),
					[](const EigenMatrix& D){ return D.size() == 0; }
				), Ds.end()
		);
		double Exc = 0;
		std::vector<EigenMatrix> Gxcs(ntypes, EigenZero(nbasis, nbasis));
		if (*xc){
			grid->getDensity(Ds);
			xc->Evaluate("ev", *grid);
			Exc = grid->getEnergy();
			Gxcs = grid->getFock();
		}

		std::vector<EigenMatrix> Fs(ntypes, EigenZero(nbasis, nbasis));
		Value = Exc;
		for ( int itype = 0; itype < ntypes; itype++ ){
			Fs[itype] = Fhfs[itype] + Gxcs[itype];
			Fprimes[itype] = Z.transpose() * Fs[itype] * Z;
			Value += 0.5 * Ds[itype].cwiseProduct( Hcore + Fhfs[itype] ).sum();
		}
		Gradient = { EigenZero(nbasis, nd + na + nb) };
		int itype = 0;
		if (nd){
			Gradient[0](Eigen::placeholders::all, Eigen::seqN(0, nd)) = 4 * Fprimes[itype] * Cprimes[itype];
			itype++;
		}
		if (na){
			Gradient[0](Eigen::placeholders::all, Eigen::seqN(nd, na)) = 2 * Fprimes[itype] * Cprimes[itype];
			itype++;
		}
		if (nb){
			Gradient[0](Eigen::placeholders::all, Eigen::seqN(nd + na, nb)) = 2 * Fprimes[itype] * Cprimes[itype];
			itype++;
		}

		Eigen::HouseholderQR<EigenMatrix> qr(Cprime);
		const EigenMatrix Call = qr.householderQ();
		C = Z * Call;
		Cprime_perp = Call.rightCols(nbasis - nd - na - nb);
		std::vector<EigenMatrix> Fmos(4, EigenZero(0, 0));
		itype = 0;
		if (nd) Fmos[0] = Call.transpose() * Fprimes[itype++] * Call;
		if (na) Fmos[1] = Call.transpose() * Fprimes[itype++] * Call;
		if (nb) Fmos[2] = Call.transpose() * Fprimes[itype++] * Call;
		Fmos[3] = EigenZero(nbasis, nbasis);
		EigenMatrix A = EigenMatrix::Ones(nbasis, nbasis);
		for ( int i = 0; i < nbasis; i++ ){
			int I = 0; int Iscale = 4;
			if ( i < nd ){ I = 0; Iscale = 4; }
			else if ( i < nd + na ){ I = 1; Iscale = 2; }
			else if ( i < nd + na + nb ) { I = 2; Iscale = 2; }
			else { I = 3; Iscale = 0; };
			for ( int j = 0; j < nbasis; j++ ){
				int J = 0; int Jscale = 4;
				if ( j < nd ){ J = 0; Jscale = 4; }
				else if ( j < nd + na ){ J = 1; Jscale = 2; }
				else if ( j < nd + na + nb ){ J = 2; Jscale = 2; }
				else{ J = 3; Jscale = 0; }
				const double FIi = Fmos[I](i, i) * Iscale;
				const double FIj = Fmos[I](j, j) * Iscale;
				const double FJi = Fmos[J](i, i) * Jscale;
				const double FJj = Fmos[J](j, j) * Jscale;
				A(i, j) = std::abs(FIj + FJi - FIi - FJj);
				if ( A(i, j) < 0.1 ) A(i, j) = 0.1;
			}
		}
		K = A.block(0, 0, nd + na + nb, nd + na + nb);
		L = A.block(nd + na + nb, 0, nbasis - nd - na - nb, nd + na + nb);
	};
};

EigenMatrix Preconditioner(EigenMatrix U, EigenMatrix Uperp, EigenMatrix K, EigenMatrix L, EigenMatrix V){
	EigenMatrix Omega = U.transpose() * V;
	Omega = Omega.cwiseProduct(K);
	EigenMatrix Kappa = Uperp.transpose() * V;
	Kappa = Kappa.cwiseProduct(L);
	return ( U * Omega + Uperp * Kappa ).eval();
}

class ObjLBFGS: public ObjBase{ public:
	EigenMatrix Ksqrt, Ksqrtinv, Lsqrt, Lsqrtinv;

	using ObjBase::ObjBase;

	void Calculate(std::vector<EigenMatrix> Cprimes_, int /*derivative*/) override{
		ObjBase::Calculate(Cprimes_, 2);
		Ksqrt = K.cwiseSqrt();
		Ksqrtinv = Ksqrt.cwiseInverse();
		Lsqrt = L.cwiseSqrt();
		Lsqrtinv = Lsqrt.cwiseInverse();
	};

	std::vector<std::vector<EigenMatrix>> PreconditionerSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<std::vector<EigenMatrix>>{{ ::Preconditioner(Cprime, Cprime_perp, Ksqrtinv, Lsqrtinv, Vs[0]) }};
	};

	std::vector<std::vector<EigenMatrix>> PreconditionerInvSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<std::vector<EigenMatrix>>{{ ::Preconditioner(Cprime, Cprime_perp, Ksqrt, Lsqrt, Vs[0]) }};
	};
};

class ObjNewtonBase: public ObjBase{ public:
	EigenMatrix Kinv, Linv;

	using ObjBase::ObjBase;

	void Calculate(std::vector<EigenMatrix> Cprimes_, int /*derivative*/) override{
		ObjBase::Calculate(Cprimes_, 2);
		Kinv = K.cwiseInverse();
		Linv = L.cwiseInverse();
	};

	virtual std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDprimes) const = 0;

	std::vector<std::vector<EigenMatrix>> Hessian(std::vector<EigenMatrix> Vprimes) const override{
		std::vector<EigenMatrix> dCprimes = Cprimes;
		std::vector<EigenMatrix> dDprimes = Dprimes;
		int ncols = 0;
		for ( int itype = 0; itype < ntypes; itype++ ){
			dCprimes[itype] = Vprimes[0](Eigen::placeholders::all, Eigen::seqN(ncols, space_sizes[type]));
			ncols += space_sizes[type];
			dDprimes[itype] = Cprimes[itype] * dCprimes[itype].transpose();
			dDprimes[itype] += dDprimes[itype].transpose().eval();
		}

		const std::vector<EigenMatrix> HdDprimes = DensityHessian(dDprimes);

		std::vector<EigenMatrix> HdCprimes = Cprimes;
		for ( int itype = 0; itype < ntypes; itype++ ){
			HdCprimes[itype] = HdDprimes[itype] * Cprimes[itype] + Fprimes[itype] * dCprimes[itype];
		}
		int itype = 0;
		EigenMatrix HdCprime = EigenZero(nbasis, nd + na + nb);
		if (nd) HdCprime(Eigen::placeholders::all, Eigen::seqN(0, nd)) = 4 * HdCprimes[itype++];
		if (na) HdCprime(Eigen::placeholders::all, Eigen::seqN(nd, na)) = 2 * HdCprimes[itype++];
		if (nb) HdCprime(Eigen::placeholders::all, Eigen::seqN(nd + na, nb)) = 2 * HdCprimes[itype++];
		return std::vector<std::vector<EigenMatrix>>{{ HdCprime }};
	};

	std::vector<std::vector<EigenMatrix>> Preconditioner(std::vector<EigenMatrix> Vs) const override{
		return std::vector<std::vector<EigenMatrix>>{{ ::Preconditioner(Cprime, Cprime_perp, Kinv, Linv, Vs[0]) }};
	};
};

class ObjNewton: public ObjNewtonBase{ public:
	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> Cprimes_, int /*derivative*/) override{
		ObjNewtonBase::Calculate(Cprimes_, 2);
		if (*xc) xc->Evaluate("f", *grid);
	};

	std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDprimes) const override{
		std::vector<std::vector<EigenMatrix>> dDs(3, {EigenZero(0, 0)});
		for ( int itype = 0; itype < ntypes; itype++ ){
			dDs[type][0] = Z * dDprimes[itype] * Z.transpose();
		}
		std::vector<EigenMatrix> dGs(3, EigenZero(nbasis, nbasis));
		std::tie(dGs[0], dGs[1], dGs[2]) = int4c2e->ContractInts(dDs[0][0], dDs[1][0], dDs[2][0], nthreads, 0);

		dDs[0][0] *= 2;
		dDs.erase(
				std::remove_if(
					dDs.begin(), dDs.end(),
					[](const std::vector<EigenMatrix>& dD){ return dD[0].size() == 0; }
				), dDs.end()
		);
		if (*xc){
			grid->getDensityU(dDs);
			const std::vector<std::vector<EigenMatrix>> dGxcs = grid->getFockU<u_t>();
			for ( int itype = 0; itype < ntypes; itype++ ){
				dGs[type] += dGxcs[itype][0];
			}
		}
		std::vector<EigenMatrix> HdDprimes(ntypes, EigenZero(nbasis, nbasis));
		for ( int itype = 0; itype < ntypes; itype++ ){
			HdDprimes[itype] = Z.transpose() * dGs[type] * Z;
		}
		return HdDprimes;
	};
};

class ObjARH: public ObjNewtonBase{ public:
	AugmentedRoothaanHall arh = AugmentedRoothaanHall(20, 1);

	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> Cprimes_, int /*derivative*/) override{
		ObjNewtonBase::Calculate(Cprimes_, 2);
		EigenMatrix Dprime = EigenZero(nbasis, nbasis * ntypes);
		EigenMatrix Fprime = EigenZero(nbasis, nbasis * ntypes);
		for ( int itype = 0; itype < ntypes; itype++ ){
			Dprime(Eigen::placeholders::all, Eigen::seqN(itype * nbasis, nbasis)) = Dprimes[itype];
			Fprime(Eigen::placeholders::all, Eigen::seqN(itype * nbasis, nbasis)) = Fprimes[itype] / 2;
		}
		if (nd) Fprime.leftCols(nbasis) *= 2;
		arh.Append(Dprime, Fprime);
	};

	std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDprimes) const override{
		EigenMatrix dDprime = EigenZero(nbasis, nbasis * ntypes);
		for ( int itype = 0; itype < ntypes; itype++ ){
			dDprime(Eigen::placeholders::all, Eigen::seqN(itype * nbasis, nbasis)) = dDprimes[itype];
		}
		const EigenMatrix HdDprime = arh.Hessian(dDprime);
		std::vector<EigenMatrix> HdDprimes = dDprimes;
		for ( int itype = 0; itype < ntypes; itype++ ){
			HdDprimes[itype] = HdDprime(Eigen::placeholders::all, Eigen::seqN(itype * nbasis, nbasis));
		}
		return HdDprimes;
	};
};

} // namespace

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nd, int na, int nb, EigenMatrix Z,
		int nthreads, int output){
	std::conditional_t< scf_t == lbfgs_t,
				ObjLBFGS,
				std::conditional_t< scf_t == newton_t,
							ObjNewton,
							ObjARH
				>
	> obj(int2c1e, int4c2e, xc, grid, nd, na, nb, Z, nthreads);
	std::vector<int> space = {};
	if (nd) space.push_back(nd);
	if (na) space.push_back(na);
	if (nb) space.push_back(nb);
	Maniverse::Flag flag(EigenOne(Z.rows(), nd + na + nb)); flag.setBlockParameters(space);
	Maniverse::Iterate M(obj, {flag.Share()}, 1);
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

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenLBFGS(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nd, int na, int nb, EigenMatrix Z,
		int nthreads, int output){
	return RestrictedOpenRiemann<lbfgs_t>(int2c1e, int4c2e, xc, grid, nd, na, nb, Z, nthreads, output);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenNewton(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nd, int na, int nb, EigenMatrix Z,
		int nthreads, int output){
	return RestrictedOpenRiemann<newton_t>(int2c1e, int4c2e, xc, grid, nd, na, nb, Z, nthreads, output);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nd, int na, int nb, EigenMatrix Z,
		int nthreads, int output){
	return RestrictedOpenRiemann<arh_t>(int2c1e, int4c2e, xc, grid, nd, na, nb, Z, nthreads, output);
}
