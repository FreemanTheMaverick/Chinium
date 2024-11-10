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
#include "../Multiwfn.h"

#include "Fock.h"
#include "FosterBoys.h"
#include "PipekMezey.h"

#include <iostream>

void Multiwfn::Localize(std::string scheme, std::string range, int output){

	std::transform(scheme.begin(), scheme.end(), scheme.begin(), ::toupper);
	std::transform(range.begin(), range.end(), range.begin(), ::toupper);

	std::vector<EigenMatrix> Crefs = {};
	int nocc = (int)this->getNumElec(0) / 2;
	int nvir = this->getNumIndBasis() - nocc;
	EigenMatrix Call = this->getCoefficientMatrix();
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
		const EigenMatrix Fao = this->getFock();
		const EigenVector E = this->getEnergy();
		const EigenVector E2 = E.cwiseProduct(E);
		this->setEnergy(E2);
		const EigenMatrix F2ao = this->getFock();
		this->setEnergy(E);
		for ( EigenMatrix Cref : Crefs ){
			const EigenMatrix Fref = Cref.transpose() * Fao * Cref;
			const EigenMatrix F2ref = Cref.transpose() * F2ao * Cref;
			Us.push_back(Fock(Fref, F2ref, output-1));
		}
	} else if ( scheme.compare("FOSTER") == 0 ){
		if (output > 0) std::printf("Foster-Boys localization:\n");
		const EigenMatrix Wxao = this->DipoleX;
		const EigenMatrix Wyao = this->DipoleY;
		const EigenMatrix Wzao = this->DipoleZ;
		const EigenMatrix W2aoSum = this->QuadrapoleXX + this->QuadrapoleYY + this-> QuadrapoleZZ;
		for ( EigenMatrix Cref : Crefs )
			Us.push_back(FosterBoys(Cref, Wxao, Wyao, Wzao, W2aoSum, output-1));
	} else if ( scheme.compare("PIPEK") == 0){
		if (output > 0) std::printf("Pipek-Mezey localization:\n");
		Eigen::SelfAdjointEigenSolver<EigenMatrix> solver(this->Overlap);
		const EigenMatrix S12 = solver.operatorSqrt();
		for ( EigenMatrix Cref : Crefs ){
			const EigenMatrix S12Cref = S12 * Cref;
			int nbasis = 0;
			std::vector<EigenMatrix> Qrefs = {};
			for ( MwfnCenter& center : this->Centers ){
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
	this->setCoefficientMatrix(Call);

}
