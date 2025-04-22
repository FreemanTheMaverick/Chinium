#include <Eigen/Dense>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <functional>
#include <tuple>
#include <cassert>
#include <chrono>
#include <cstdio>

#include "../Macro.h"
#include "../Multiwfn/Multiwfn.h"
#include "../Integral/Int2C1E.h"

#include "Fock.h"
#include "FosterBoys.h"
#include "PipekMezey.h"

#include <iostream>

void Localize(Multiwfn& mwfn, Int2C1E& int2c1e, std::string scheme, std::string range, int output){

	std::transform(scheme.begin(), scheme.end(), scheme.begin(), ::toupper);
	std::transform(range.begin(), range.end(), range.begin(), ::toupper);

	std::vector<EigenMatrix> Crefs = {};
	int nocc = (int)mwfn.getNumElec(0) / 2;
	int nvir = mwfn.getNumIndBasis() - nocc;
	EigenMatrix Call = mwfn.getCoefficientMatrix();
	if ( range.compare("OCC") == 0 )
		Crefs = {Call.leftCols(nocc)};
	else if ( range.compare("VIR") == 0 )
		Crefs = {Call.rightCols(nvir)};
	else if ( range.compare("ALL") == 0 )
		Crefs = {Call};
	else if ( range.compare("BOTH") == 0 )
		Crefs = {Call.leftCols(nocc), Call.rightCols(nvir)};

	std::vector<EigenMatrix> Us = {};
	if ( scheme.compare("FOCK") == 0 ){
		if (output > 0) std::printf("Fock localization:\n");
		const EigenMatrix Fao = mwfn.getFock();
		const EigenVector E = mwfn.getEnergy();
		const EigenVector E2 = E.cwiseProduct(E);
		mwfn.setEnergy(E2);
		const EigenMatrix F2ao = mwfn.getFock();
		mwfn.setEnergy(E);
		for ( EigenMatrix Cref : Crefs ){
			const EigenMatrix Fref = Cref.transpose() * Fao * Cref;
			const EigenMatrix F2ref = Cref.transpose() * F2ao * Cref;
			Us.push_back(Fock(Fref, F2ref, output-1));
		}
	} else if ( scheme.compare("FOSTER") == 0 ){
		if (output > 0) std::printf("Foster-Boys localization:\n");
		const EigenMatrix Wxao = int2c1e.DipoleX;
		const EigenMatrix Wyao = int2c1e.DipoleY;
		const EigenMatrix Wzao = int2c1e.DipoleZ;
		const EigenMatrix W2aoSum = int2c1e.QuadrapoleXX + int2c1e.QuadrapoleYY + int2c1e.QuadrapoleZZ;
		for ( EigenMatrix Cref : Crefs )
			Us.push_back(FosterBoys(Cref, Wxao, Wyao, Wzao, W2aoSum, output-1));
	} else if ( scheme.compare("PIPEK") == 0){
		if (output > 0) std::printf("Pipek-Mezey localization:\n");
		Eigen::SelfAdjointEigenSolver<EigenMatrix> solver(int2c1e.Overlap);
		const EigenMatrix S12 = solver.operatorSqrt();
		for ( EigenMatrix Cref : Crefs ){
			const EigenMatrix S12Cref = S12 * Cref;
			int nbasis = 0;
			std::vector<EigenMatrix> Qrefs = {};
			for ( MwfnCenter& center : mwfn.Centers ){
				const EigenMatrix tmp = S12Cref.middleRows(nbasis, center.getNumBasis());
				Qrefs.push_back(tmp.transpose() * tmp);
				nbasis += center.getNumBasis();
			}
			Us.push_back(PipekMezey(Qrefs, output-1));
		}
	}


	if ( range.compare("OCC") == 0 )
		Call.leftCols(nocc) = Crefs[0] * Us[0];
	else if ( range.compare("VIR") == 0 )
		Call.rightCols(nvir) = Crefs[0] * Us[0];
	else if ( range.compare("ALL") == 0 )
		Call = Crefs[0] * Us[0];
	else if ( range.compare("BOTH") == 0 ){
		Call.leftCols(nocc) = Crefs[0] * Us[0];
		Call.rightCols(nvir) = Crefs[1] * Us[1];
	}
	mwfn.setCoefficientMatrix(Call);

}
