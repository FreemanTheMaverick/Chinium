#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <omp.h>

#include "../Macro.h"
#include "../Multiwfn.h"
#include "../Grid/GridDensity.h"
#include "../ExchangeCorrelation/MwfnXC1.h"
#include "../Grid/GridPotential.h"

#include <iostream>

EigenMatrix HFGradient(
		EigenMatrix D, std::vector<std::vector<EigenMatrix>>& Hgrads,
		EigenMatrix W, std::vector<std::vector<EigenMatrix>>& Sgrads,
		std::vector<std::vector<EigenMatrix>>& Ggrads){
	EigenMatrix g = EigenZero(Hgrads.size(), 3);
	for ( int iatom = 0; iatom < (int)Hgrads.size(); iatom++ )
		for ( int xyz = 0; xyz < 3; xyz++ )
			g(iatom, xyz) = ( D * Hgrads[iatom][xyz] - W * Sgrads[iatom][xyz] + 0.5 * D * Ggrads[iatom][xyz] ).trace();
	return g;
}

void Multiwfn::HFKSGradient(){
	assert((int)this->OverlapGrads.size() == this->getNumCenters() && "Overlap matrix gradient does not exist!");
	assert((int)this->KineticGrads.size() == this->getNumCenters() && "Kinetic matrix gradient does not exist!");
	assert((int)this->NuclearGrads.size() == this->getNumCenters() && "Nuclear matrix gradient does not exist!");
	assert((int)this->GGrads.size() == this->getNumCenters() && "Two-electron matrix gradient does not exist!");
	std::vector<std::vector<EigenMatrix>> Hgrads = {};
	for ( int iatom = 0; iatom < this->getNumCenters(); iatom++ )
		Hgrads.push_back({
			this->KineticGrads[iatom][0] + this->NuclearGrads[iatom][0],
			this->KineticGrads[iatom][1] + this->NuclearGrads[iatom][1],
			this->KineticGrads[iatom][2] + this->NuclearGrads[iatom][2]
		});
	const EigenMatrix hfg = HFGradient(this->getDensity(), Hgrads, this->getEnergyDensity(), this->OverlapGrads, this->GGrads);
	const EigenMatrix xcg = EigenZero(this->getNumCenters(), 3);
	this->Gradient += hfg + xcg;
}


