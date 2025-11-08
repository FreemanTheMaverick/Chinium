#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
extern "C"{
	#include <xc.h>
}
#include <sstream>
#include <fstream>
#include <cstdio>
#include <cassert>
#include <string>
#include <vector>
#include <libmwfn.h>

#include "Macro.h"
#include "Grid/Grid.h"
#include "ExchangeCorrelation.h"

void ExchangeCorrelation::Read(std::string df, bool output){

	// Parsing df file.
	std::string filename = (std::string)std::getenv("CHINIUM_PATH") + "/DensityFunctionals/" + df + ".df";
	std::ifstream file(filename);
	if (!file.good()) throw std::runtime_error("Density functional file is missing!");
	if (output) std::printf("Reading density functional file %s ...\n", filename.c_str());
	std::string thisline;

	std::getline(file, thisline);
	std::stringstream ss(thisline);
	int ifunc = 0;
	int type = 0;
	int code = 0;
	this->EXX = 0;
	if (output) std::printf("Density functional information:\n");
	while (ss >> code){
		if (ss.fail()) throw std::runtime_error("Corrupted density functional file!");
		xc_func_type func;
		xc_func_init(&func, code, this->Spin);
		std::string name = func.info->name;
		std::string kind;
		switch (func.info->kind){
			case (XC_EXCHANGE):
				kind = "Exchange"; break;
			case (XC_CORRELATION):
				kind = "Correlation"; break;
			case (XC_EXCHANGE_CORRELATION):
				kind = "Exchange-correlation"; break;
			case (XC_KINETIC):
				kind = "Kinetic"; break;
		}
		std::string family;
		switch (func.info->family){
			case (XC_FAMILY_LDA):
			case (XC_FAMILY_HYB_LDA):
				family = "LDA"; type = type > 0 ? type : 0; break;
			case (XC_FAMILY_GGA):
			case (XC_FAMILY_HYB_GGA):
				family = "GGA"; type = type > 1 ? type : 1; break;
			case (XC_FAMILY_MGGA):
			case (XC_FAMILY_HYB_MGGA):
				family = "mGGA"; type = type > 2 ? type : 2; break;
		}
		this->Codes.push_back(code);
		const double exx = xc_hyb_exx_coef(&func);
		if ( this->EXX < exx ) this->EXX = exx;
		if (output){
			std::printf("| Functional %d:\n", ifunc++);
			std::printf("| | Code: %d\n", code);
			std::printf("| | Name: %s\n", name.c_str());
			std::printf("| | Kind: %s\n", kind.c_str());
			std::printf("| | Family: %s\n", family.c_str());
			if ( func.info->kind == XC_EXCHANGE || func.info->kind == XC_EXCHANGE_CORRELATION )
				std::printf("| | Exact exchange component: %f\n", exx);
		}
		xc_func_end(&func);
	}
	switch (type){
		case (0):
			this->Family = "LDA"; break;
		case (1):
			this->Family = "GGA"; break;
		case (2):
			this->Family = "mGGA"; break;
	}
}

#define __Allocate_Temporary_2__(tensor)\
	Eigen::Tensor<double, 2> tensor(ngrids, 1);\
	tensor.setZero();

#define __Allocate_Temporary_3__(tensor)\
	Eigen::Tensor<double, 3> tensor(ngrids, 1, 1);\
	tensor.setZero();

#define __Append_XC__(tensor, tmp)\
	if ( subgrid->tensor.size() == 0 ) subgrid->tensor = tmp;\
	else subgrid->tensor += tmp;

#define __XC_LDA__\
	assert( rho.size() == ngrids );\
	if ( order == "e" ){\
		__Allocate_Temporary_2__(eps)\
		xc_lda_exc(&func, ngrids, rho.data(), eps.data());\
		__Append_XC__(Eps, eps)\
	}else if ( order == "v" ){\
		__Allocate_Temporary_2__(eps1rho)\
		xc_lda_vxc(&func, ngrids, rho.data(), eps1rho.data());\
		__Append_XC__(Eps1Rho, eps1rho)\
	}else if ( order == "ev" ){\
		__Allocate_Temporary_2__(eps)\
		__Allocate_Temporary_2__(eps1rho)\
		xc_lda_exc_vxc(&func, ngrids, rho.data(), eps.data(), eps1rho.data());\
		__Append_XC__(Eps, eps)\
		__Append_XC__(Eps1Rho, eps1rho)\
	}else if ( order == "f" ){\
		__Allocate_Temporary_3__(eps2rho2)\
		xc_lda_fxc(&func, ngrids, rho.data(), eps2rho2.data());\
		__Append_XC__(Eps2Rho2, eps2rho2)\
	}/*else if ( order == "k" ){\
		__Allocate_Temporary__(eps3rho3)\
		xc_lda_kxc(&func, ngrids, rho.data(), eps3rho3.data());\
		__Append_XC__(Eps3Rho3, eps3rho3)\
	}else if ( order == "l" ){\
		__Allocate_Temporary__(eps4rho4)\
		xc_lda_lxc(&func, ngrids, rho.data(), eps4rho4.data());\
		__Append_XC__(Eps4Rho4, eps4rho4)\
	}*/

#define __XC_GGA__\
	assert( rho.size() == ngrids );\
	assert( sigma.size() == ngrids );\
	if ( order == "e" ){\
		__Allocate_Temporary_2__(eps)\
		xc_gga_exc(&func, ngrids, rho.data(), sigma.data(), eps.data());\
		__Append_XC__(Eps, eps)\
	}else if ( order == "v" ){\
		__Allocate_Temporary_2__(eps1rho)\
		__Allocate_Temporary_2__(eps1sigma)\
		xc_gga_vxc(&func, ngrids, rho.data(), sigma.data(), eps1rho.data(), eps1sigma.data());\
		__Append_XC__(Eps1Rho, eps1rho)\
		__Append_XC__(Eps1Sigma, eps1sigma)\
	}else if ( order == "ev" ){\
		__Allocate_Temporary_2__(eps)\
		__Allocate_Temporary_2__(eps1rho)\
		__Allocate_Temporary_2__(eps1sigma)\
		xc_gga_exc_vxc(&func, ngrids, rho.data(), sigma.data(), eps.data(), eps1rho.data(), eps1sigma.data());\
		__Append_XC__(Eps, eps)\
		__Append_XC__(Eps1Rho, eps1rho)\
		__Append_XC__(Eps1Sigma, eps1sigma)\
	}else if ( order == "f" ){\
		__Allocate_Temporary_3__(eps2rho2)\
		__Allocate_Temporary_3__(eps2rhosigma)\
		__Allocate_Temporary_3__(eps2sigma2)\
		xc_gga_fxc(&func, ngrids, rho.data(), sigma.data(), eps2rho2.data(), eps2rhosigma.data(), eps2sigma2.data());\
		__Append_XC__(Eps2Rho2, eps2rho2)\
		__Append_XC__(Eps2RhoSigma, eps2rhosigma)\
		__Append_XC__(Eps2Sigma2, eps2sigma2)\
	}/*else if ( order == "k" ){\
		__Allocate_Temporary__(eps3rho3)\
		__Allocate_Temporary__(eps3rho2sigma)\
		__Allocate_Temporary__(eps3rhosigma2)\
		__Allocate_Temporary__(eps3sigma3)\
		xc_gga_kxc(&func, ngrids, rho.data(), sigma.data(), eps3rho3.data(), eps3rho2sigma.data(), eps3rhosigma2.data(), eps3sigma3.data());\
		__Append_XC__(Eps3Rho3, eps3rho3)\
		__Append_XC__(Eps3Rho2Sigma, eps3rho2sigma)\
		__Append_XC__(Eps3RhoSigma2, eps3rhosigma2)\
		__Append_XC__(Eps3Sigma3, eps3sigma3)\
	}else if ( order == "l" ){\
		__Allocate_Temporary__(eps4rho4)\
		__Allocate_Temporary__(eps4rho3sigma)\
		__Allocate_Temporary__(eps4rho2sigma2)\
		__Allocate_Temporary__(eps4rhosigma3)\
		__Allocate_Temporary__(eps4sigma4)\
		xc_gga_lxc(&func, ngrids, rho.data(), sigma.data(), eps4rho4.data(), eps4rho3sigma.data(), eps4rho2sigma2.data(), eps4rhosigma3.data(), eps4sigma4.data());\
		__Append_XC__(Eps4Rho4, eps4rho4)\
		__Append_XC__(Eps4Rho3Sigma, eps4rho3sigma)\
		__Append_XC__(Eps4Rho2Sigma2, eps4rho2sigma2)\
		__Append_XC__(Eps4RhoSigma3, eps4rhosigma3)\
		__Append_XC__(Eps4Sigma4, eps4sigma4)\
	}*/

#define __XC_MGGA__\
	assert( rho.size() == ngrids );\
	assert( sigma.size() == ngrids );\
	assert( lapl.size() == ngrids );\
	assert( tau.size() == ngrids );\
	if ( order == "e" ){\
		__Allocate_Temporary_2__(eps)\
		xc_mgga_exc(&func, ngrids, rho.data(), sigma.data(), lapl.data(), tau.data(), eps.data());\
		__Append_XC__(Eps, eps)\
	}else if ( order == "v" ){\
		__Allocate_Temporary_2__(eps1rho)\
		__Allocate_Temporary_2__(eps1sigma)\
		__Allocate_Temporary_2__(eps1lapl)\
		__Allocate_Temporary_2__(eps1tau)\
		xc_mgga_vxc(&func, ngrids, rho.data(), sigma.data(), lapl.data(), tau.data(), eps1rho.data(), eps1sigma.data(), eps1lapl.data(), eps1tau.data());\
		__Append_XC__(Eps1Rho, eps1rho)\
		__Append_XC__(Eps1Sigma, eps1sigma)\
		__Append_XC__(Eps1Lapl, eps1lapl)\
		__Append_XC__(Eps1Tau, eps1tau)\
	}else if ( order == "ev" ){\
		__Allocate_Temporary_2__(eps)\
		__Allocate_Temporary_2__(eps1rho)\
		__Allocate_Temporary_2__(eps1sigma)\
		__Allocate_Temporary_2__(eps1lapl)\
		__Allocate_Temporary_2__(eps1tau)\
		xc_mgga_exc_vxc(&func, ngrids, rho.data(), sigma.data(), lapl.data(), tau.data(), eps.data(), eps1rho.data(), eps1sigma.data(), eps1lapl.data(), eps1tau.data());\
		__Append_XC__(Eps, eps)\
		__Append_XC__(Eps1Rho, eps1rho)\
		__Append_XC__(Eps1Sigma, eps1sigma)\
		__Append_XC__(Eps1Lapl, eps1lapl)\
		__Append_XC__(Eps1Tau, eps1tau)\
	}

void ExchangeCorrelation::Evaluate(std::string order, Grid& grid){
	#pragma omp parallel for schedule(static) num_threads(grid.getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : grid.SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){		
			const int ngrids = subgrid->NumGrids;
			Eigen::Tensor<double, 2>& rho = subgrid->Rho;
			Eigen::Tensor<double, 2>& sigma = subgrid->Sigma;
			Eigen::Tensor<double, 2>& lapl = subgrid->Lapl;
			Eigen::Tensor<double, 2>& tau = subgrid->Tau;
			if ( order == "e" ){
				subgrid->Eps.setZero();
			}else if ( order == "v" ){
				subgrid->Eps1Rho.setZero(); subgrid->Eps1Sigma.setZero(); subgrid->Eps1Lapl.setZero(); subgrid->Eps1Tau.setZero();
			}else if ( order == "ev" ){
				subgrid->Eps.setZero();
				subgrid->Eps1Rho.setZero(); subgrid->Eps1Sigma.setZero(); subgrid->Eps1Lapl.setZero(); subgrid->Eps1Tau.setZero();
			}else if ( order == "f" ){
				subgrid->Eps2Rho2.setZero(); subgrid->Eps2RhoSigma.setZero(); subgrid->Eps2Sigma2.setZero();
			}/*else if ( order == "k" ){
				subgrid->Eps3Rho3.setZero(); subgrid->Eps3Rho2Sigma.setZero(); subgrid->Eps3RhoSigma2.setZero(); subgrid->Eps3Sigma3.setZero();
			}else if ( order == "l" ){
				subgrid->Eps4Rho4.setZero(); subgrid->Eps4Rho3Sigma.setZero(); subgrid->Eps4Rho2Sigma2.setZero(); subgrid->Eps4RhoSigma3.setZero(); subgrid->Eps4Sigma4.setZero();
			}*/
			for ( int code : this->Codes ){
				xc_func_type func;
				xc_func_init(&func, code, this->Spin);
				switch (func.info->family){
					case (XC_FAMILY_LDA):
					case (XC_FAMILY_HYB_LDA):
						__XC_LDA__ break;
					case (XC_FAMILY_GGA):
					case (XC_FAMILY_HYB_GGA):
						__XC_GGA__ break;
					case (XC_FAMILY_MGGA):
					case (XC_FAMILY_HYB_MGGA):
						__XC_MGGA__ break;
				}
				xc_func_end(&func);
			}
		}
	}
}
