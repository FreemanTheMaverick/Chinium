#include<iostream>
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <tuple>
#include <cstdio>
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <libmwfn.h>

#include "../../Macro.h"
#include "../../Integral.h"
#include "../../Grid.h"
#include "../../ExchangeCorrelation.h"

#include "../AugmentedRoothaanHall.h"
#include "../TwoDeterminant.h"

#define S ( int2c1e->Overlap )
#define Hcore ( int2c1e->Kinetic + int2c1e->Nuclear )

namespace{

#define One ( grid2->SubGridBatches.size() > 0 )
#define __Make_Block_View__(Mat, Mats) Mats = {Mat.middleCols(0, Np), Mat.middleCols(Np, 1), Mat.middleCols(Np + 1, 1)};

class ObjBase: public Maniverse::Objective{ public:
	Int2C1E* int2c1e;
	Int4C2E* int4c2e;
	ExchangeCorrelation* xc;
	Grid* grid;
	Grid* grid2;
	int Np;
	EigenMatrix Z;
	int nthreads;

	int nbasis;

	EigenMatrix Cprime;
	EigenMatrix Cprime_perp;
	std::vector<int> occ = {2, 1, 1};
	std::vector<Eigen::Ref<EigenMatrix>> Cprimes;
	std::vector<Eigen::Ref<EigenMatrix>> Gradients;
	std::vector<EigenMatrix> Dprimes;
	std::vector<EigenMatrix> FprimeMs;
	std::vector<EigenMatrix> FprimeTs;
	EigenMatrix C;
	EigenMatrix K;
	EigenMatrix L;

	ObjBase(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid, Grid& grid2,
		int Np, EigenMatrix Z,
		int nthreads
	): int2c1e(&int2c1e), int4c2e(&int4c2e), xc(&xc), grid(&grid), grid2(&grid2), Np(Np), Z(Z), nthreads(nthreads){
		nbasis = Z.rows();
		Cprime = EigenZero(nbasis, Np + 2);
		Gradient = {Cprime};
		__Make_Block_View__(Cprime, Cprimes);
		__Make_Block_View__(Gradient[0], Gradients);
		Dprimes = FprimeMs = FprimeTs = { EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis) };
	};

	virtual void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			Cprime = Cprimes_[0];
			std::vector<EigenMatrix> Ds(3, EigenZero(nbasis, nbasis));
			for ( int type = 0; type < 3; type++ ){
				Dprimes[type] = Cprimes[type] * Cprimes[type].transpose();
				Ds[type] = Z * Dprimes[type] * Z.transpose();
			}
			const auto [J, Kd, Ka, Kb] = int4c2e->ContractInts(Ds[0], Ds[1], Ds[2], nthreads, 1);
			const std::vector<EigenMatrix> FhfMs = {
				Hcore + J - Kd - 0.5 * ( Ka + Kb ),
				Hcore + J - Kd - Ka + Kb,
				Hcore + J - Kd + Ka - Kb
			};
			const std::vector<EigenMatrix> FhfTs = {
				Hcore + J - Kd - 0.5 * ( Ka + Kb ),
				Hcore + J - Kd - Ka - Kb,
				Hcore + J - Kd - Ka - Kb
			};
			Value = 0;
			for ( int type = 0; type < 3; type++ ){
				Value += 0.5 * occ[type] * Ds[type].cwiseProduct( Hcore + 2 * FhfMs[type] - FhfTs[type] ).sum();
				FprimeMs[type] = Z.transpose() * FhfMs[type] * Z;
				FprimeTs[type] = Z.transpose() * FhfTs[type] * Z;
			}

			if (*xc){
				if (One){
					grid->getDensity({ 2 * Ds[0], Ds[1], Ds[2] });
					grid2->getDensity({ 2 * Ds[0], Ds[1] + Ds[2] });
					xc->Evaluate("ev", *grid);
					xc->Evaluate("ev", *grid2);
					Value += 2 * grid->getEnergy() - grid2->getEnergy();
					const std::vector<EigenMatrix> GxcMs = grid->getFock();
					std::vector<EigenMatrix> GxcTs = grid2->getFock();
					GxcTs.push_back(GxcTs[1]);
					for ( int type = 0; type < 3; type++ ){
						FprimeMs[type] += Z.transpose() * GxcMs[type] * Z;
						FprimeTs[type] += Z.transpose() * GxcTs[type] * Z;
					}
				}else{
					grid->getDensity({ 2 * Ds[0] + Ds[1] + Ds[2] });
					xc->Evaluate("ev", *grid);
					Value += grid->getEnergy();
					const EigenMatrix Gxc = grid->getFock()[0];
					for ( int type = 0; type < 3; type++ ){
						const EigenMatrix tmp = Z.transpose() * Gxc * Z;
						FprimeMs[type] += tmp;
						FprimeTs[type] += tmp;
					}
				}
			}
		}

		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			for ( int type = 0; type < 3; type++ ){
				Gradients[type] = 2 * occ[type] * ( 2 * FprimeMs[type] - FprimeTs[type] ) * Cprimes[type];
			}

			Eigen::HouseholderQR<EigenMatrix> qr(Cprime);
			const EigenMatrix Call = qr.householderQ();
			C = Z * Call;
			Cprime_perp = Call.rightCols(nbasis - Np - 2);
			std::vector<EigenMatrix> Fmos(4, EigenZero(nbasis, nbasis));
			for ( int type = 0; type < 3; type++ ){
				Fmos[type] = Call.transpose() * ( 2 * FprimeMs[type] - FprimeTs[type] ) * Call;
			}
			EigenMatrix A = EigenMatrix::Ones(nbasis, nbasis);
			for ( int i = 0; i < nbasis; i++ ){
				int I = 0; int Iscale = 4;
				if ( i < Np ){ I = 0; Iscale = 4; }
				else if ( i < Np + 1 ){ I = 1; Iscale = 2; }
				else if ( i < Np + 2 ) { I = 2; Iscale = 2; }
				else { I = 3; Iscale = 0; };
				for ( int j = 0; j < nbasis; j++ ){
					int J = 0; int Jscale = 4;
					if ( j < Np ){ J = 0; Jscale = 4; }
					else if ( j < Np + 1 ){ J = 1; Jscale = 2; }
					else if ( j < Np + 2 ){ J = 2; Jscale = 2; }
					else{ J = 3; Jscale = 0; }
					const double FIi = Fmos[I](i, i) * Iscale;
					const double FIj = Fmos[I](j, j) * Iscale;
					const double FJi = Fmos[J](i, i) * Jscale;
					const double FJj = Fmos[J](j, j) * Jscale;
					A(i, j) = std::abs(FIj + FJi - FIi - FJj);
					if ( A(i, j) < 0.1 ) A(i, j) = 0.1;
				}
			}
			K = A.block(0, 0, Np + 2, Np + 2);
			L = A.block(Np + 2, 0, nbasis - Np - 2, Np + 2);
		}
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

	void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		ObjBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Ksqrt = K.cwiseSqrt();
			Ksqrtinv = Ksqrt.cwiseInverse();
			Lsqrt = L.cwiseSqrt();
			Lsqrtinv = Lsqrt.cwiseInverse();
		}
	};

	std::vector<EigenMatrix> PreconditionerSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{ ::Preconditioner(Cprime, Cprime_perp, Ksqrtinv, Lsqrtinv, Vs[0]) };
	};

	std::vector<EigenMatrix> PreconditionerInvSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{ ::Preconditioner(Cprime, Cprime_perp, Ksqrt, Lsqrt, Vs[0]) };
	};
};

class ObjNewtonBase: public ObjBase{ public:
	EigenMatrix Kinv, Linv;

	using ObjBase::ObjBase;

	void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		ObjBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) ){
			Kinv = K.cwiseInverse();
			Linv = L.cwiseInverse();
		}
	};

	virtual std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDprimes) const = 0;

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> Vprimes) const override{
		std::vector<Eigen::Ref<EigenMatrix>> dCprimes;
		__Make_Block_View__(Vprimes[0], dCprimes);
		std::vector<EigenMatrix> dDprimes(3, EigenZero(nbasis, nbasis));
		for ( int type = 0; type < 3; type++ ){
			dDprimes[type] = Cprimes[type] * dCprimes[type].transpose();
			dDprimes[type] += dDprimes[type].transpose().eval();
		}

		const std::vector<EigenMatrix> HdDprimes = DensityHessian(dDprimes);

		EigenMatrix HdCprime = EigenZero(nbasis, Np + 2);
		std::vector<Eigen::Ref<EigenMatrix>> HdCprimes;
		__Make_Block_View__(HdCprime, HdCprimes);
		for ( int type = 0; type < 3; type++ ){
			HdCprimes[type] = 2 * occ[type] * ( HdDprimes[type] * Cprimes[type] + ( 2 * FprimeMs[type] - FprimeTs[type] ) * dCprimes[type] );
		}
		return std::vector<EigenMatrix>{ HdCprime };
	};

	std::vector<EigenMatrix> Preconditioner(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{ ::Preconditioner(Cprime, Cprime_perp, Kinv, Linv, Vs[0]) };
	};
};

class ObjNewton: public ObjNewtonBase{ public:
	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		ObjNewtonBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) && *xc ){
			xc->Evaluate("f", *grid);
			if (One) xc->Evaluate("f", *grid2);
		}
	};

	std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDprimes) const override{
		std::vector<std::vector<EigenMatrix>> dDs(3, {EigenZero(0, 0)});
		for ( int type = 0; type < 3; type++ ){
			dDs[type][0] = Z * dDprimes[type] * Z.transpose();
		}
		const auto [J, Kd, Ka, Kb] = int4c2e->ContractInts(dDs[0][0], dDs[1][0], dDs[2][0], nthreads, 0);
		std::vector<EigenMatrix> dFMs = {
			J - Kd - 0.5 * ( Ka + Kb ),
			J - Kd - Ka + Kb,
			J - Kd + Ka - Kb
		};
		std::vector<EigenMatrix> dFTs = {
			J - Kd - 0.5 * ( Ka + Kb ),
			J - Kd - Ka - Kb,
			J - Kd - Ka - Kb
		};

		if (*xc){
			if (One){
				grid->getDensityU({ { 2 * dDs[0][0] }, { dDs[1][0] }, { dDs[2][0] } });
				grid2->getDensityU({ { 2 * dDs[0][0] }, { dDs[1][0] + dDs[2][0] } });
				const std::vector<std::vector<EigenMatrix>> dGxcMs = grid->getFockU<u_t>();
				std::vector<std::vector<EigenMatrix>> dGxcTs = grid2->getFockU<u_t>();
				dGxcTs.push_back(dGxcTs.back());
				for ( int type = 0; type < 3; type++ ){
					dFMs[type] += dGxcMs[type][0];
					dFTs[type] += dGxcTs[type][0];
				}
			}else{
				grid->getDensityU({{ 2 * dDs[0][0] + dDs[1][0] + dDs[2][0] }});
				const EigenMatrix dGxcs = grid->getFockU<u_t>()[0][0];
				for ( int type = 0; type < 3; type++ ){
					dFMs[type] += dGxcs;
					dFTs[type] += dGxcs;
				}
			}
		}
		std::vector<EigenMatrix> HdDprimes(3, EigenZero(nbasis, nbasis));
		for ( int type = 0; type < 3; type++ ){
			HdDprimes[type] = Z.transpose() * ( 2 * dFMs[type] - dFTs[type] ) * Z;
		}
		return HdDprimes;
	};
};

class ObjARH: public ObjNewtonBase{ public:
	AugmentedRoothaanHall arh = AugmentedRoothaanHall(20, 1);

	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		ObjNewtonBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			EigenMatrix Dprime = EigenZero(nbasis, nbasis * 3);
			EigenMatrix Fprime = EigenZero(nbasis, nbasis * 3);
			for ( int type = 0; type < 3; type++ ){
				Dprime.middleCols(type * nbasis, nbasis) = Dprimes[type];
				Fprime.middleCols(type * nbasis, nbasis) = 0.5 * occ[type] * ( 2 * FprimeMs[type] - FprimeTs[type] );
			}
			arh.Append(Dprime, Fprime);
		}
	};

	std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDprimes) const override{
		EigenMatrix dDprime = EigenZero(nbasis, nbasis * 3);
		for ( int type = 0; type < 3; type++ ){
			dDprime.middleCols(type * nbasis, nbasis) = dDprimes[type];
		}
		const EigenMatrix HdDprime = arh.Hessian(dDprime);
		std::vector<EigenMatrix> HdDprimes = dDprimes;
		for ( int type = 0; type < 3; type++ ){
			HdDprimes[type] = HdDprime.middleCols(type * nbasis, nbasis);
		}
		return HdDprimes;
	};
};

} // namespace

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenMatrix> TwoDeterminantRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid, Grid& grid2,
		int Np, EigenMatrix Z,
		int nthreads, int output){
	std::conditional_t< scf_t == lbfgs_t,
				ObjLBFGS,
				std::conditional_t< scf_t == newton_t,
							ObjNewton,
							ObjARH
				>
	> obj(int2c1e, int4c2e, xc, grid, grid2, Np, Z, nthreads);
	Maniverse::Flag flag(EigenOne(Z.rows(), Np + 2)); flag.setBlockParameters({Np, 1, 1});
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

	return std::make_tuple(obj.Value, obj.C);
}

void TwoDet::Calculate0(){
	const EigenMatrix Z = mwfn.getCoefficientMatrix(1);
	auto [E, C] =
		scftype == "LBFGS" ? TwoDeterminantRiemann<lbfgs_t>(int2c1e, int4c2e, xc, grid, grid2, Np, Z, nthreads, 1) :
		scftype == "ARH" ? TwoDeterminantRiemann<arh_t>(int2c1e, int4c2e, xc, grid, grid2, Np, Z, nthreads, 1) :
		/* scftype == "NEWTON" ? */ TwoDeterminantRiemann<newton_t>(int2c1e, int4c2e, xc, grid, grid2, Np, Z, nthreads, 1);
	Energy += E;
	mwfn.setCoefficientMatrix(C, 1);
}
