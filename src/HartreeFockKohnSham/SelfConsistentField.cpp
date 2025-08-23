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

#include "Restricted.h"
#include "Unrestricted.h"
#include "RestrictedOpen.h"

double HartreeFockKohnSham(Mwfn& mwfn, Environment& env, Int2C1E& int2c1e, Int4C2E& int4c2e, ExchangeCorrelation& xc, Grid& grid, std::string scf, int output, int nthreads){
	Eigen::setNbThreads(nthreads);

	const double T = env.Temperature;
	const double Mu = env.ChemicalPotential;
	if (output > 0) std::printf("Self-consistent field in %s-canonical ensemble\n", T > 0 ? "grand" : "micro");
	if ( T > 0 && scf != "DIIS" && scf != "FOCK" ) throw std::runtime_error("Only DIIS optimization is supported for finite-temperature DFT!");

	const EigenMatrix Z = mwfn.getCoefficientMatrix(1);
	EigenMatrix Z1 = EigenZero(Z.rows(), Z.cols());
	EigenMatrix Z2 = EigenZero(Z.rows(), Z.cols());
	if ( mwfn.Wfntype == 1 ){
		Z1 = mwfn.getCoefficientMatrix(1);
		Z2 = mwfn.getCoefficientMatrix(2);
	}

	double E_scf = 0;
	if ( scf == "FOCK" ){
		EigenMatrix Fprime = mwfn.getEnergy().asDiagonal();
		EigenVector Occ = mwfn.getOccupation() / 2;
		auto [E, epsilons, occupations, C] = RestrictedFock(T, Mu, int2c1e, int4c2e, xc, grid, Fprime, Occ, Z, output-1, nthreads);
		E_scf = E * 2;
		mwfn.setEnergy(epsilons);
		mwfn.setOccupation(occupations * 2);
		mwfn.setCoefficientMatrix(C);
	}else if ( scf == "RIEMANN-ARH" ){
		if ( mwfn.Wfntype == 0 ){
			EigenMatrix Dprime = (mwfn.getOccupation(1) / 2).asDiagonal();
			auto [E, epsilons, C] = RestrictedRiemannARH(int2c1e, int4c2e, xc, grid, Dprime, Z, output-1, nthreads);
			E_scf = 2 * E;
			mwfn.setEnergy(epsilons, 1);
			mwfn.setCoefficientMatrix(C, 1);
		}else if ( mwfn.Wfntype == 1 ){
			EigenMatrix D1prime = mwfn.getOccupation(1).asDiagonal();
			EigenMatrix D2prime = mwfn.getOccupation(2).asDiagonal();
			auto [E, epsilon1s, epsilon2s, C1, C2] = UnrestrictedRiemannARH(int2c1e, int4c2e, D1prime, D2prime, Z1, Z2, output-1, nthreads);
			E_scf = E;
			mwfn.setEnergy(epsilon1s, 1);
			mwfn.setEnergy(epsilon2s, 2);
			mwfn.setCoefficientMatrix(C1, 1);
			mwfn.setCoefficientMatrix(C2, 2);
		}else if ( mwfn.Wfntype == 2 ){
			int nd = mwfn.getNumElec(2);
			int ns = mwfn.getNumElec(1) - nd;
			EigenMatrix Cprime = EigenOne(Z.rows(), ns + nd);
			auto [E, epsilons, C] = RestrictedOpenRiemannARH(nd, ns, int2c1e, int4c2e, Cprime, Z, output-1, nthreads);
			E_scf = E;
		}
	}else if ( scf == "RIEMANN" ){
		if ( mwfn.Wfntype == 0 ){
			EigenMatrix Dprime = (mwfn.getOccupation(1)).asDiagonal();
			auto [E, epsilons, C] = RestrictedRiemann(int2c1e, int4c2e, xc, grid, Dprime, Z, output-1, nthreads);
			E_scf = 2 * E;
			mwfn.setEnergy(epsilons, 1);
			mwfn.setCoefficientMatrix(C, 1);
		}else if ( mwfn.Wfntype == 1 ){
			EigenMatrix D1prime = mwfn.getOccupation(1).asDiagonal();
			EigenMatrix D2prime = mwfn.getOccupation(2).asDiagonal();
			auto [E, epsilon1s, epsilon2s, C1, C2] = UnrestrictedRiemann(int2c1e, int4c2e, D1prime, D2prime, Z1, Z2, output-1, nthreads);
			E_scf = E;
			mwfn.setEnergy(epsilon1s, 1);
			mwfn.setEnergy(epsilon2s, 2);
			mwfn.setCoefficientMatrix(C1, 1);
			mwfn.setCoefficientMatrix(C2, 2);
		}else if ( mwfn.Wfntype == 2 ){
			int nd = mwfn.getNumElec(2);
			int ns = mwfn.getNumElec(1) - nd;
			EigenMatrix Cprime = EigenOne(Z.rows(), ns + nd);
			auto [E, epsilons, C] = RestrictedOpenRiemann(nd, ns, int2c1e, int4c2e, Cprime, Z, output-1, nthreads);
			E_scf = E;
		}
	}else if ( scf == "DIIS" ){
		if ( mwfn.Wfntype == 0 ){
			EigenMatrix F = mwfn.getFock(1);
			EigenVector Occ = mwfn.getOccupation(1) / 2;
			auto [E, epsilons, occupations, C] = RestrictedDIIS(
					T, Mu,
					int2c1e, int4c2e,
					xc, grid,
					F, Occ, Z,
					output-1, nthreads
			);
			E_scf = E;
			mwfn.setEnergy(epsilons, 1);
			mwfn.setOccupation(occupations * 2, 1);
			mwfn.setCoefficientMatrix(C, 1);
		}else if ( mwfn.Wfntype == 1 ){
			EigenMatrix Fa = mwfn.getFock(1);
			EigenMatrix Fb = mwfn.getFock(2);
			EigenVector Occa = mwfn.getOccupation(1);
			EigenVector Occb = mwfn.getOccupation(2);
			auto [E, epsa, epsb, occa, occb, Ca, Cb] = UnrestrictedDIIS(
					T, Mu,
					int2c1e, int4c2e,
					Fa, Fb,
					Occa, Occb,
					Z, Z,
					output-1, nthreads
			);
			E_scf = E;
			mwfn.setEnergy(epsa, 1);
			mwfn.setEnergy(epsb, 2);
			mwfn.setOccupation(occa, 1);
			mwfn.setOccupation(occb, 2);
			mwfn.setCoefficientMatrix(Ca, 1);
			mwfn.setCoefficientMatrix(Cb, 2);
		}
	}
	return E_scf;
}
