#include <Eigen/Dense>
#include <vector>
#include <Maniverse/Manifold/Manifold.h>

#include "../Macro.h"
#include "../Integral.h"
#include "../Grid.h"
#include "../ExchangeCorrelation.h"

#include "Universal.h"

#define Hcore ( int2c1e->Kinetic + int2c1e->Nuclear )

UniversalObjBase::UniversalObjBase(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		std::vector<int> Norbs, double Coupling,
		std::vector<EigenMatrix> Zs, int nthreads
): int2c1e(&int2c1e), int4c2e(&int4c2e), xc(&xc), grid(&grid), Norbs(Norbs), Coupling(Coupling), Zs(Zs), nthreads(nthreads){
	nbasis = Zs.back().rows();
	for ( int type = 0; type < 3; type++ ){
		Cprimes.push_back(EigenZero(nbasis, Norbs[type]));
		Gradients.push_back(EigenZero(nbasis, Norbs[type]));
		Dprimes.push_back(EigenZero(nbasis, nbasis));
		Fprimes.push_back(EigenZero(nbasis, nbasis));
	}
}

void UniversalObjBase::Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives){
	Cprimes = Cprimes_;
	if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
		std::vector<EigenMatrix> Ds(3, EigenZero(0, 0));
		for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
			Dprimes[type] = Cprimes[type] * Cprimes[type].transpose();
			Ds[type].resize(nbasis, nbasis);
			Ds[type] = Zs[type] * Dprimes[type] * Zs[type].transpose();
		}
		const auto [J, Kd, Ka, Kb] = int4c2e->ContractInts(Ds[0], Ds[1], Ds[2], nthreads, 1);
		Value = 0;
		for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
			const EigenMatrix Fhf =
				type == 0 ? ( Hcore + J - Kd - 0.5 * Ka - 0.5 * Kb ).eval() :
				type == 1 ? ( Hcore + J - Kd - Ka + Coupling * Kb ).eval() :
				/* type == 2 ? */ ( Hcore + J - Kd - Kb + Coupling * Ka ).eval();
			Value += 0.5 * occ[type] * Ds[type].cwiseProduct( Hcore + Fhf ).sum();
			Fprimes[type] = Zs[type].transpose() * Fhf * Zs[type];
		}

		if (*xc){
			std::vector<EigenMatrix> Dab(2, EigenZero(nbasis, nbasis));
			for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
				if ( type == 0 || type == 1 ) Dab[0] += Ds[type];
				if ( type == 0 || type == 2 ) Dab[1] += Ds[type];
			}
			if ( Norbs[0] && !Norbs[1] && !Norbs[2] ) Dab = { 2 * Dab[0] };
			grid->getDensity(Dab);
			xc->Evaluate("ev", *grid);
			Value += grid->getEnergy();
		}
	}

	if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
		if (*xc){
			const std::vector<EigenMatrix> Fxcab = grid->getFock();
			if ( Norbs[0] && !Norbs[1] && !Norbs[2] ){
				Fprimes[0] += Zs[0].transpose() * Fxcab[0] * Zs[0];
			}else for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
				Fprimes[type] += Zs[type].transpose() * (
					type == 0 ? ( ( Fxcab[0] + Fxcab[1] ) / 2 ).eval() :
					type == 1 ? Fxcab[0] :
					/* type == 2 ? */ Fxcab[1]
				) * Zs[type];
			}
		}
		for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
			Gradients[type] = 2 * occ[type] * Fprimes[type] * Cprimes[type];
		}
	}
}

EigenMatrix UniversalPreconditioner(EigenMatrix U, EigenMatrix Uperp, EigenMatrix K, EigenMatrix L, EigenMatrix V){
	EigenMatrix Omega = U.transpose() * V;
	Omega = Omega.cwiseProduct(K);
	EigenMatrix Kappa = Uperp.transpose() * V;
	Kappa = Kappa.cwiseProduct(L);
	return ( U * Omega + Uperp * Kappa ).eval();
}
