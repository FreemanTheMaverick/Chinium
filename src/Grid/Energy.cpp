#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <string>
#include <vector>
#include <libmwfn.h>

#include "../Macro.h"
#include "Grid.h"

void SubGrid::getEnergy(EigenTensor<0>& E){
	#include "EnergyEinSum/W_g...Rho_g,w...Eps_g---E_.hpp"
}

void SubGrid::getEnergyGrad(EigenTensor<2>& E){
	const int ngrids = this->NumGrids;
	const int nspins = this->Spin;
	if ( this->Type >= 0 ){
		#include "EnergyEinSum/W_g...Eps1Rho_g,w...RhoGrad_g,t,a,w---E_t,a.hpp"
	}
	if ( this->Type >= 1 ){
		EigenTensor<2> S(nspins, nspins); // Scaling factors
		S.setConstant(0.5);
		for ( int w = 0; w < nspins; w++ ) S(w, w) = 1;
		EigenTensor<3> ScaledEps1Sigma(ngrids, nspins, nspins);
		ScaledEps1Sigma.setZero();
		#include "EnergyEinSum/S_u,v...Eps1Sigma_g,u+v---ScaledEps1Sigma_g,u,v.hpp"
		#include "EnergyEinSum/W_g...ScaledEps1Sigma_g,u,v...SigmaGrad_g,t,a,v+v---E_t,a.hpp"
	}
}

void SubGrid::getEnergyHess(EigenTensor<4>& E){
	const int natoms = this->getNumAtoms();
	if ( this->Type >= 0 ){
		#include "EnergyEinSum/W_g...Eps2Rho2_g,u,v...RhoGrad_g,t,a,u...RhoGrad_g,s,b,v---E_t,a,s,b.hpp"
		#include "EnergyEinSum/W_g...Eps1Rho_g,w...RhoHess_g,t,a,s,b,w---E_t,a,s,b.hpp"
	}
	if ( this->Type >= 1 ){ // To be corrected for U and RO
		EigenTensor<4> E1(3, natoms, 3, natoms); E1.setZero();
		#include "EnergyEinSum/W_g...Eps2RhoSigma_g,w,v...RhoGrad_g,t,a,w...SigmaGrad_g,s,b,v---E1_t,a,s,b.hpp"
		E += E1 + E1.shuffle(Eigen::array<int, 4>{2, 3, 0, 1});
		#include "EnergyEinSum/W_g...Eps2Sigma2_g,u,v...SigmaGrad_g,t,a,u...SigmaGrad_g,s,b,v---E_t,a,s,b.hpp"
		#include "EnergyEinSum/W_g...Eps1Sigma_g,u...SigmaHess_g,t,a,s,b,u---E_t,a,s,b.hpp"
	}
}
