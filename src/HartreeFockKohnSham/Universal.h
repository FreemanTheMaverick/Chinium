#pragma once

#include <Eigen/Dense>
#include <vector>
#include <Maniverse/Manifold/Manifold.h>

#include "../Macro.h"
#include "../Integral.h"
#include "../Grid.h"
#include "../ExchangeCorrelation.h"

#include "AugmentedRoothaanHall.h"

class UniversalObjBase: public Maniverse::Objective{ public:
	Int2C1E* int2c1e;
	Int4C2E* int4c2e;
	ExchangeCorrelation* xc;
	Grid* grid;
	std::vector<int> Norbs;
	double Coupling;

	std::vector<EigenMatrix> Zs;
	int nthreads;

	int nbasis;

	std::vector<int> occ = { 2, 1, 1 };
	std::vector<EigenMatrix> Cprimes;
	std::vector<EigenMatrix> Gradients;
	std::vector<EigenMatrix> Dprimes;
	std::vector<EigenMatrix> Fprimes;

	UniversalObjBase(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		std::vector<int> Norbs, double Coupling,
		std::vector<EigenMatrix> Zs, int nthreads
	);

	virtual void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override;
};

EigenMatrix UniversalPreconditioner(EigenMatrix U, EigenMatrix Uperp, EigenMatrix K, EigenMatrix L, EigenMatrix V);

#define __Base_Members__(basename)\
	using basename::basename;\
	using basename::int2c1e;\
	using basename::int4c2e;\
	using basename::xc;\
	using basename::grid;\
	using basename::Norbs;\
	using basename::Coupling;\
	using basename::Zs;\
	using basename::nthreads;\
	using basename::nbasis;\
	using basename::occ;\
	using basename::Cprimes;\
	using basename::Gradients;\
	using basename::Dprimes;\
	using basename::Fprimes;

template<typename ObjBase>
class UniversalObjNewtonBase: public ObjBase{ public:
	__Base_Members__(ObjBase)

	virtual std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDprimes) const = 0;

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> dCprimes) const override{
		std::vector<EigenMatrix> dDprimes(3, EigenZero(nbasis, nbasis));
		for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
			dDprimes[type] = Cprimes[type] * dCprimes[type].transpose();
			dDprimes[type] += dDprimes[type].transpose().eval();
		}

		const std::vector<EigenMatrix> HdDprimes = DensityHessian(dDprimes);

		std::vector<EigenMatrix> HdCprimes = dCprimes;
		for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
			HdCprimes[type] = 2 * occ[type] * ( HdDprimes[type] * Cprimes[type] + Fprimes[type] * dCprimes[type] );
		}
		return HdCprimes;
	};
};

template<typename ObjNewtonBase>
class UniversalObjNewton: public ObjNewtonBase{ public:
	__Base_Members__(ObjNewtonBase)

	void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		ObjNewtonBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) && *xc ){
			xc->Evaluate("f", *grid);
		}
	};

	std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDprimes) const override{
		std::vector<std::vector<EigenMatrix>> dDs(3, {EigenZero(0, 0)});
		for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
			dDs[type][0] = Zs[type] * dDprimes[type] * Zs[type].transpose();
		}
		const auto [J, Kd, Ka, Kb] = int4c2e->ContractInts(dDs[0][0], dDs[1][0], dDs[2][0], nthreads, 0);
		std::vector<EigenMatrix> dFs(3, EigenZero(nbasis, nbasis));
		for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
			dFs[type] =
				type == 0 ? ( J - Kd - 0.5 * Ka - 0.5 * Kb ).eval() :
				type == 1 ? ( J - Kd - Ka + Coupling * Kb ).eval() :
				/* type == 2 ? */ ( J - Kd - Kb + Coupling * Ka ).eval();
		}

		if (*xc){
			std::vector<std::vector<EigenMatrix>> dDab(2, {EigenZero(nbasis, nbasis)});
			for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
				if ( type == 0 || type == 1 ) dDab[0][0] += dDs[type][0];
				if ( type == 0 || type == 2 ) dDab[1][0] += dDs[type][0];
			}
			if ( Norbs[0] && !Norbs[1] && !Norbs[2] ) dDab = {{ 2 * dDab[0][0] }};
			grid->getDensityU(dDab);
			const std::vector<std::vector<EigenMatrix>> dFxcab = grid->template getFockU<u_t>();
			if ( Norbs[0] && !Norbs[1] && !Norbs[2] ){
				dFs[0] += dFxcab[0][0];
			}else for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
				dFs[type] +=
					type == 0 ? ( ( dFxcab[0][0] + dFxcab[1][0] ) / 2 ).eval() :
					type == 1 ? dFxcab[0][0] :
					/* type == 2 ? */ dFxcab[1][0];
			}
		}
		std::vector<EigenMatrix> HdDprimes = dDprimes;
		for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
			HdDprimes[type] = Zs[type].transpose() * dFs[type] * Zs[type];
		}
		return HdDprimes;
	};
};

template<typename ObjNewtonBase>
class UniversalObjARH: public ObjNewtonBase{ public:
	__Base_Members__(ObjNewtonBase)

	AugmentedRoothaanHall arh = AugmentedRoothaanHall(20, 1);

	#define ntypes ( ( Norbs[0] > 0 ) + ( Norbs[1] > 0 ) + ( Norbs[2] > 0 ) )
	void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		ObjNewtonBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			EigenMatrix Dprime = EigenZero(nbasis, nbasis * ntypes);
			EigenMatrix Fprime = EigenZero(nbasis, nbasis * ntypes);
			for ( int type = 0, itype = 0; type < 3; type++ ) if ( Norbs[type] ){
				Dprime.middleCols(itype * nbasis, nbasis) = Dprimes[type];
				Fprime.middleCols(itype * nbasis, nbasis) = Fprimes[type];
				itype++;
			}
			arh.Append(Dprime, Fprime);
		}
	};

	std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDprimes) const override{
		EigenMatrix dDprime = EigenZero(nbasis, nbasis * ntypes);
		for ( int type = 0, itype = 0; type < 3; type++ ) if ( Norbs[type] ){
			dDprime.middleCols((itype++) * nbasis, nbasis) = dDprimes[type];
		}
		const EigenMatrix HdDprime = arh.Hessian(dDprime);
		std::vector<EigenMatrix> HdDprimes = dDprimes;
		for ( int type = 0, itype = 0; type < 3; type++ ) if ( Norbs[type] ){
			HdDprimes[type] = HdDprime.middleCols((itype++) * nbasis, nbasis);
		}
		return HdDprimes;
	};
};
