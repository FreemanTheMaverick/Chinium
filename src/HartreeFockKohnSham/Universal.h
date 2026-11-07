#pragma once

#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <Maniverse/Manifold/Manifold.h>

#include "../Macro.h"
#include "../Integral.h"
#include "../Grid.h"
#include "../ExchangeCorrelation.h"

#include "AugmentedRoothaanHall.h"

class UniversalObjBase: public Maniverse::Function{ public:
	Int2C1E* int2c1e;
	Int4C2E* int4c2e;
	EigenMatrix Ssqrt;
	EigenMatrix Ssqrtinv;
	ExchangeCorrelation* xc;
	Grid* grid;
	std::vector<int> Norbs;
	double Coupling;

	int nthreads;

	int nbasis;

	std::vector<int> occ = { 2, 1, 1 };
	std::vector<EigenMatrix> Cs;
	std::vector<EigenMatrix> Gradients;
	std::vector<EigenMatrix> Ds;
	std::vector<EigenMatrix> Fs;

	UniversalObjBase(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		std::vector<int> Norbs, double Coupling,
		int nthreads
	);

	virtual void Calculate(std::vector<EigenMatrix> Cs_, std::vector<int> derivatives) override;
};

template<typename ObjBase>
class UniversalObjNewtonBase: public ObjBase{ public:
	using ObjBase::ObjBase;
	using ObjBase::Norbs;
	using ObjBase::nbasis;
	using ObjBase::Cs;
	using ObjBase::occ;
	using ObjBase::Fs;

	virtual std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDs) const = 0;

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> dCs) const override{
		std::vector<EigenMatrix> dDs(3, EigenZero(nbasis, nbasis));
		for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
			dDs[type] = Cs[type] * dCs[type].transpose();
			dDs[type] += dDs[type].transpose().eval();
		}

		const std::vector<EigenMatrix> HdDs = DensityHessian(dDs);

		std::vector<EigenMatrix> HdCs = dCs;
		for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
			HdCs[type] = 2 * occ[type] * ( HdDs[type] * Cs[type] + Fs[type] * dCs[type] );
		}
		return HdCs;
	};
};

template<typename ObjNewtonBase>
class UniversalObjNewton: public ObjNewtonBase{ public:
	using ObjNewtonBase::ObjNewtonBase;
	using ObjNewtonBase::int4c2e;
	using ObjNewtonBase::xc;
	using ObjNewtonBase::grid;
	using ObjNewtonBase::Norbs;
	using ObjNewtonBase::Coupling;
	using ObjNewtonBase::nthreads;
	using ObjNewtonBase::nbasis;

	void Calculate(std::vector<EigenMatrix> Cs_, std::vector<int> derivatives) override{
		ObjNewtonBase::Calculate(Cs_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) && *xc ){
			xc->Evaluate("f", *grid);
		}
	};

	std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDs) const override{
		const auto [J, Kd, Ka, Kb] = int4c2e->ContractInts(dDs[0], dDs[1], dDs[2], nthreads, 0);
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
				if ( type == 0 || type == 1 ) dDab[0][0] += dDs[type];
				if ( type == 0 || type == 2 ) dDab[1][0] += dDs[type];
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
		return dFs;
	};
};

template<typename ObjNewtonBase>
class UniversalObjARH: public ObjNewtonBase{ public:
    using ObjNewtonBase::ObjNewtonBase;
    using ObjNewtonBase::Norbs;
    using ObjNewtonBase::nbasis;
    using ObjNewtonBase::Ssqrt;
    using ObjNewtonBase::Ssqrtinv;
    using ObjNewtonBase::Ds;
    using ObjNewtonBase::Fs;

	AugmentedRoothaanHall arh = AugmentedRoothaanHall(20, 1);

	#define ntypes ( ( Norbs[0] > 0 ) + ( Norbs[1] > 0 ) + ( Norbs[2] > 0 ) )
	void Calculate(std::vector<EigenMatrix> Cs_, std::vector<int> derivatives) override{
		ObjNewtonBase::Calculate(Cs_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			EigenMatrix Dprime = EigenZero(nbasis, nbasis * ntypes);
			EigenMatrix Fprime = EigenZero(nbasis, nbasis * ntypes);
			for ( int type = 0, itype = 0; type < 3; type++ ) if ( Norbs[type] ){
				Dprime.middleCols(itype * nbasis, nbasis) = Ssqrt * Ds[type] * Ssqrt;
				Fprime.middleCols(itype * nbasis, nbasis) = Ssqrtinv * Fs[type] * Ssqrtinv;
				itype++;
			}
			arh.Append(Dprime, Fprime);
		}
	};

	std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDs) const override{
		EigenMatrix dDprime = EigenZero(nbasis, nbasis * ntypes);
		for ( int type = 0, itype = 0; type < 3; type++ ) if ( Norbs[type] ){
			dDprime.middleCols((itype++) * nbasis, nbasis) = Ssqrt * dDs[type] * Ssqrt;
		}
		const EigenMatrix HdDprime = arh.Hessian(dDprime);
		std::vector<EigenMatrix> HdDs = dDs;
		for ( int type = 0, itype = 0; type < 3; type++ ) if ( Norbs[type] ){
			HdDs[type] = Ssqrt * HdDprime.middleCols((itype++) * nbasis, nbasis) * Ssqrt;
		}
		return HdDs;
	};
};

extern std::function<double (double)> SafeInverse;

EigenMatrix UniversalPreconditioner(EigenMatrix U, EigenMatrix Uperp, EigenMatrix K, EigenMatrix L, EigenMatrix V);