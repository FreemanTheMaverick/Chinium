#include <Eigen/Dense>
#include <string>
#include <vector>
#include <cstdio>

#include "../Macro.h"
#include "../Multiwfn.h"

#include <iostream>

#define __Check_and_Allocate__(array, multiple)\
	array = new double[this->NumGrids * ( this->XC.Spin == 1 ? 1 : multiple )]();\
	if (output) std::printf("Allocating memory for array %s ...\n", #array);

void Multiwfn::PrepareXC(std::string order, int output){
	const std::string family = this->XC.XCcode == 0 ? this->XC.Xfamily : this->XC.XCfamily;
	if ( this->XC.XCcode == 0 && this->XC.Xcode == 0 ) return;

	// Preparing density
	if ( family.compare("LDA") == 0 ){
		__Check_and_Allocate__(this->Rhos, 2);
	}else if ( family.compare("GGA") == 0 ){
		__Check_and_Allocate__(this->Rhos, 2);
		__Check_and_Allocate__(this->Rho1Xs, 2);
		__Check_and_Allocate__(this->Rho1Ys, 2);
		__Check_and_Allocate__(this->Rho1Zs, 2);
		__Check_and_Allocate__(this->Sigmas, 3);
	}else if ( family.compare("mGGA") == 0 ){
		__Check_and_Allocate__(this->Rhos, 2);
		__Check_and_Allocate__(this->Rho1Xs, 2);
		__Check_and_Allocate__(this->Rho1Ys, 2);
		__Check_and_Allocate__(this->Rho1Zs, 2);
		__Check_and_Allocate__(this->Sigmas, 3);
		__Check_and_Allocate__(this->Lapls, 2);
		__Check_and_Allocate__(this->Taus, 2);
	}

	// Preparing XC potentials
	if ( family.compare("LDA") == 0 ){
		if ( order.compare("e") == 0 ){
			__Check_and_Allocate__(this->Es, 1);
		}else if ( order.compare("v") == 0 ){
			__Check_and_Allocate__(this->E1Rhos, 2);
		}else if ( order.compare("ev") == 0 ){
			__Check_and_Allocate__(this->Es, 1);
			__Check_and_Allocate__(this->E1Rhos, 2);
		}else if ( order.compare("f") == 0 ){
			__Check_and_Allocate__(this->E2Rho2s, 3);
		}else if ( order.compare("k") == 0 ){
			__Check_and_Allocate__(this->E3Rho3s, 4);
		}else if ( order.compare("l") == 0 ){
			__Check_and_Allocate__(this->E4Rho4s, 5);
		}
	}else if ( family.compare("GGA") == 0 ){
		if ( order.compare("e") == 0 ){
			__Check_and_Allocate__(this->Es, 1);
		}else if ( order.compare("v") == 0 ){
			__Check_and_Allocate__(this->E1Rhos, 2);
			__Check_and_Allocate__(this->E1Sigmas, 3);
		}else if ( order.compare("ev") == 0 ){
			__Check_and_Allocate__(this->Es, 1);
			__Check_and_Allocate__(this->E1Rhos, 2);
			__Check_and_Allocate__(this->E1Sigmas, 3);
		}else if ( order.compare("f") == 0 ){
			__Check_and_Allocate__(this->E2Rho2s, 3);
			__Check_and_Allocate__(this->E2RhoSigmas, 6);
			__Check_and_Allocate__(this->E2Sigma2s, 6);
		}else if ( order.compare("k") == 0 ){
			__Check_and_Allocate__(this->E3Rho3s, 4);
			__Check_and_Allocate__(this->E3Rho2Sigmas, 9);
			__Check_and_Allocate__(this->E3RhoSigma2s, 12);
			__Check_and_Allocate__(this->E3Sigma3s, 10);
		}else if ( order.compare("l") == 0 ){
			__Check_and_Allocate__(this->E4Rho4s, 5);
			__Check_and_Allocate__(this->E4Rho3Sigmas, 12);
			__Check_and_Allocate__(this->E4Rho2Sigma2s, 15);
			__Check_and_Allocate__(this->E4RhoSigma3s, 20);
			__Check_and_Allocate__(this->E4Sigma4s, 15);
		}
	}else if ( family.compare("mGGA") == 0 ){
		if ( order.compare("e") == 0 ){
			__Check_and_Allocate__(this->Es, 1);
		}else if ( order.compare("v") == 0 ){
			__Check_and_Allocate__(this->E1Rhos, 2);
			__Check_and_Allocate__(this->E1Sigmas, 3);
			__Check_and_Allocate__(this->E1Lapls, 2);
			__Check_and_Allocate__(this->E1Taus, 2);
		}else if ( order.compare("ev") == 0 ){
			__Check_and_Allocate__(this->Es, 1);
			__Check_and_Allocate__(this->E1Rhos, 2);
			__Check_and_Allocate__(this->E1Sigmas, 3);
			__Check_and_Allocate__(this->E1Lapls, 2);
			__Check_and_Allocate__(this->E1Taus, 2);
		}
	}
}
