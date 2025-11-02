#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <string>
#include <vector>
#include <libmwfn.h>

#include "../Macro.h"
#include "Tensor.h"
#include "Grid.h"

double SubGrid::getEnergy(){
	assert( this->NumGrids == W.dimension(0) );
	assert( this->NumGrids == Rho.dimension(0) );
	assert( this->NumGrids == E.dimension(0) );
	const EigenTensor<0> e = (W * Rho * E).sum();
	return e();
}

void SubGrid::getEnergyGrad(std::vector<double>& e){
	const int natoms = this->getNumAtoms();
	EigenTensor<2> G(3, natoms); G.setZero();
	if ( this->Type >= 0 ){
		#include "EnergyEinSum/W_g...E1Rho_g...RhoGrad_g,t,a---G_t,a.hpp"
	}
	if ( this->Type >= 1 ){
		#include "EnergyEinSum/W_g...E1Sigma_g...SigmaGrad_g,t,a---G_t,a.hpp"
	}
	for ( int a = 0; a < natoms; a++ ){
		e[this->AtomList[a] * 3 + 0] += G(0, a);
		e[this->AtomList[a] * 3 + 1] += G(1, a);
		e[this->AtomList[a] * 3 + 2] += G(2, a);
	}
}

void Grid::getEnergyHess(std::vector<std::vector<double>>& e){
	const int natoms = this->getNumAtoms();
	EigenTensor<4> H(3, natoms, 3, natoms); H.setZero();
	if ( this->Type >= 0 ){
		#include "EnergyEinSum/W_g...E2Rho2_g...RhoGrad_g,t,a...RhoGrad_g,s,b---H_t,a,s,b.hpp"
		#include "EnergyEinSum/W_g...E1Rho_g...RhoHess_g,t,a,s,b---H_t,a,s,b.hpp"
	}
	if ( this->Type >= 1 ){
		EigenTensor<4> H1(3, natoms, 3, natoms); H1.setZero();
		#include "EnergyEinSum/W_g...E2RhoSigma_g...RhoGrad_g,t,a...SigmaGrad_g,s,b---H1_t,a,s,b.hpp"
		H += H1 + H1.shuffle(Eigen::array<int, 4>{2, 3, 0, 1});
		#include "EnergyEinSum/W_g...E2Sigma2_g...SigmaGrad_g,t,a...SigmaGrad_g,s,b---H_t,a,s,b.hpp"
		#include "EnergyEinSum/W_g...E1Sigma_g...SigmaHess_g,t,a,s,b---H_t,a,s,b.hpp"
	}
	for ( int a = 0; a < natoms; a++ ){
		for ( int b = 0; b < natoms; b++ ){
			e[0 + this->AtomList[a] * 3][0 + this->AtomList[b] * 3] += H(0, a, 0, b);
			e[0 + this->AtomList[a] * 3][1 + this->AtomList[b] * 3] += H(0, a, 1, b);
			e[0 + this->AtomList[a] * 3][2 + this->AtomList[b] * 3] += H(0, a, 2, b);
			e[1 + this->AtomList[a] * 3][0 + this->AtomList[b] * 3] += H(1, a, 0, b);
			e[1 + this->AtomList[a] * 3][1 + this->AtomList[b] * 3] += H(1, a, 1, b);
			e[1 + this->AtomList[a] * 3][2 + this->AtomList[b] * 3] += H(1, a, 2, b);
			e[2 + this->AtomList[a] * 3][0 + this->AtomList[b] * 3] += H(2, a, 0, b);
			e[2 + this->AtomList[a] * 3][1 + this->AtomList[b] * 3] += H(2, a, 1, b);
			e[2 + this->AtomList[a] * 3][2 + this->AtomList[b] * 3] += H(2, a, 2, b);
		}
	}
}
