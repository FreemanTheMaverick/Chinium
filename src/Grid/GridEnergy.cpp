#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <string>
#include <vector>
#include <libmwfn.h>

#include "../Macro.h"
#include "Tensor.h"
#include "Grid.h"

double Grid::getEnergy(){
	assert( this->NumGrids == Weights.dimension(0) );
	assert( this->Rhos.size() ? this->NumGrids == Rhos.dimension(0) : this->NumGrids == Rhos_Cache.dimension(0) );
	assert( this->NumGrids == Es.dimension(0) );
	Eigen::Tensor<double, 1>& Rhos = this->Rhos.size() ? this->Rhos : this->Rhos_Cache;
	const Eigen::Tensor<double, 0> e = (Weights * Rhos * Es).sum();
	return e();
}

std::vector<double> Grid::getEnergyGrad(){
	const int natoms = this->MWFN->getNumCenters();
	Eigen::Tensor<double, 1>& Ws = Weights;
	Eigen::Tensor<double, 2> G(3, natoms); G.setZero();
	if ( this->Type >= 0 ){
		#include "EnergyEinSum/Ws_g...E1Rhos_g...RhoGrads_g,t,a---G_t,a.hpp"
	}
	if ( this->Type >= 1 ){
		#include "EnergyEinSum/Ws_g...E1Sigmas_g...SigmaGrads_g,t,a---G_t,a.hpp"
	}
	std::vector<double> Gvec(G.data(), G.data() + 3 * natoms);
	return Gvec;
}

std::vector<std::vector<double>> Grid::getEnergyHess(){
	const int natoms = this->MWFN->getNumCenters();
	Eigen::Tensor<double, 1>& Ws = Weights;
	Eigen::Tensor<double, 4> H(3, natoms, 3, natoms); H.setZero();
	if ( this->Type >= 0 ){
		#include "EnergyEinSum/Ws_g...E2Rho2s_g...RhoGrads_g,t,a...RhoGrads_g,s,b---H_t,a,s,b.hpp"
		#include "EnergyEinSum/Ws_g...E1Rhos_g...RhoHesss_g,t,a,s,b---H_t,a,s,b.hpp"
	}
	if ( this->Type >= 1 ){
		Eigen::Tensor<double, 4> H1(3, natoms, 3, natoms); H1.setZero();
		#include "EnergyEinSum/Ws_g...E2RhoSigmas_g...RhoGrads_g,t,a...SigmaGrads_g,s,b---H1_t,a,s,b.hpp"
		H += H1 + H1.shuffle(Eigen::array<int, 4>{2, 3, 0, 1});
		#include "EnergyEinSum/Ws_g...E2Sigma2s_g...SigmaGrads_g,t,a...SigmaGrads_g,s,b---H_t,a,s,b.hpp"
		#include "EnergyEinSum/Ws_g...E1Sigmas_g...SigmaHesss_g,t,a,s,b---H_t,a,s,b.hpp"
	}
	std::vector<std::vector<double>> Hmat(3 * natoms, std::vector<double>(3 * natoms));
	for ( int i = 0; i < 3 * natoms; i++ )
		Hmat[i].assign(H.data() + i * 3 * natoms, H.data() + ( i + 1 ) * 3 * natoms);
	return Hmat;
}
