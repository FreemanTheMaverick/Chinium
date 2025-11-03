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

#define __Allocate_Temporary__(tensor)\
	Eigen::Tensor<double, 1> tensor(ngrids);\
	tensor.setZero();

#define __Append_XC__(tensor, tmp)\
	if ( subgrid->tensor.size() == 0 ) subgrid->tensor = tmp;\
	else subgrid->tensor += tmp;

#define __XC_LDA__\
	assert( rho.size() == ngrids );\
	if ( order == "e" ){\
		__Allocate_Temporary__(e)\
		xc_lda_exc(&func, ngrids, rho.data(), e.data());\
		__Append_XC__(E, e)\
	}else if ( order == "v" ){\
		__Allocate_Temporary__(e1rho)\
		xc_lda_vxc(&func, ngrids, rho.data(), e1rho.data());\
		__Append_XC__(E1Rho, e1rho)\
	}else if ( order == "ev" ){\
		__Allocate_Temporary__(e)\
		__Allocate_Temporary__(e1rho)\
		xc_lda_exc_vxc(&func, ngrids, rho.data(), e.data(), e1rho.data());\
		__Append_XC__(E, e)\
		__Append_XC__(E1Rho, e1rho)\
	}else if ( order == "f" ){\
		__Allocate_Temporary__(e2rho2)\
		xc_lda_fxc(&func, ngrids, rho.data(), e2rho2.data());\
		__Append_XC__(E2Rho2, e2rho2)\
	}/*else if ( order == "k" ){\
		__Allocate_Temporary__(e3rho3)\
		xc_lda_kxc(&func, ngrids, rho.data(), e3rho3.data());\
		__Append_XC__(E3Rho3, e3rho3)\
	}else if ( order == "l" ){\
		__Allocate_Temporary__(e4rho4)\
		xc_lda_lxc(&func, ngrids, rho.data(), e4rho4.data());\
		__Append_XC__(E4Rho4, e4rho4)\
	}*/

#define __XC_GGA__\
	assert( rho.size() == ngrids );\
	assert( sigma.size() == ngrids );\
	if ( order == "e" ){\
		__Allocate_Temporary__(e)\
		xc_gga_exc(&func, ngrids, rho.data(), sigma.data(), e.data());\
		__Append_XC__(E, e)\
	}else if ( order == "v" ){\
		__Allocate_Temporary__(e1rho)\
		__Allocate_Temporary__(e1sigma)\
		xc_gga_vxc(&func, ngrids, rho.data(), sigma.data(), e1rho.data(), e1sigma.data());\
		__Append_XC__(E1Rho, e1rho)\
		__Append_XC__(E1Sigma, e1sigma)\
	}else if ( order == "ev" ){\
		__Allocate_Temporary__(e)\
		__Allocate_Temporary__(e1rho)\
		__Allocate_Temporary__(e1sigma)\
		xc_gga_exc_vxc(&func, ngrids, rho.data(), sigma.data(), e.data(), e1rho.data(), e1sigma.data());\
		__Append_XC__(E, e)\
		__Append_XC__(E1Rho, e1rho)\
		__Append_XC__(E1Sigma, e1sigma)\
	}else if ( order == "f" ){\
		__Allocate_Temporary__(e2rho2)\
		__Allocate_Temporary__(e2rhosigma)\
		__Allocate_Temporary__(e2sigma2)\
		xc_gga_fxc(&func, ngrids, rho.data(), sigma.data(), e2rho2.data(), e2rhosigma.data(), e2sigma2.data());\
		__Append_XC__(E2Rho2, e2rho2)\
		__Append_XC__(E2RhoSigma, e2rhosigma)\
		__Append_XC__(E2Sigma2, e2sigma2)\
	}/*else if ( order == "k" ){\
		__Allocate_Temporary__(e3rho3)\
		__Allocate_Temporary__(e3rho2sigma)\
		__Allocate_Temporary__(e3rhosigma2)\
		__Allocate_Temporary__(e3sigma3)\
		xc_gga_kxc(&func, ngrids, rho.data(), sigma.data(), e3rho3.data(), e3rho2sigma.data(), e3rhosigma2.data(), e3sigma3.data());\
		__Append_XC__(E3Rho3, e3rho3)\
		__Append_XC__(E3Rho2Sigma, e3rho2sigma)\
		__Append_XC__(E3RhoSigma2, e3rhosigma2)\
		__Append_XC__(E3Sigma3, e3sigma3)\
	}else if ( order == "l" ){\
		__Allocate_Temporary__(e4rho4)\
		__Allocate_Temporary__(e4rho3sigma)\
		__Allocate_Temporary__(e4rho2sigma2)\
		__Allocate_Temporary__(e4rhosigma3)\
		__Allocate_Temporary__(e4sigma4)\
		xc_gga_lxc(&func, ngrids, rho.data(), sigma.data(), e4rho4.data(), e4rho3sigma.data(), e4rho2sigma2.data(), e4rhosigma3.data(), e4sigma4.data());\
		__Append_XC__(E4Rho4, e4rho4)\
		__Append_XC__(E4Rho3Sigma, e4rho3sigma)\
		__Append_XC__(E4Rho2Sigma2, e4rho2sigma2)\
		__Append_XC__(E4RhoSigma3, e4rhosigma3)\
		__Append_XC__(E4Sigma4, e4sigma4)\
	}*/

#define __XC_MGGA__\
	assert( rho.size() == ngrids );\
	assert( sigma.size() == ngrids );\
	assert( lapl.size() == ngrids );\
	assert( tau.size() == ngrids );\
	if ( order == "e" ){\
		__Allocate_Temporary__(e)\
		xc_mgga_exc(&func, ngrids, rho.data(), sigma.data(), lapl.data(), tau.data(), e.data());\
		__Append_XC__(E, e)\
	}else if ( order == "v" ){\
		__Allocate_Temporary__(e1rho)\
		__Allocate_Temporary__(e1sigma)\
		__Allocate_Temporary__(e1lapl)\
		__Allocate_Temporary__(e1tau)\
		xc_mgga_vxc(&func, ngrids, rho.data(), sigma.data(), lapl.data(), tau.data(), e1rho.data(), e1sigma.data(), e1lapl.data(), e1tau.data());\
		__Append_XC__(E1Rho, e1rho)\
		__Append_XC__(E1Sigma, e1sigma)\
		__Append_XC__(E1Lapl, e1lapl)\
		__Append_XC__(E1Tau, e1tau)\
	}else if ( order == "ev" ){\
		__Allocate_Temporary__(e)\
		__Allocate_Temporary__(e1rho)\
		__Allocate_Temporary__(e1sigma)\
		__Allocate_Temporary__(e1lapl)\
		__Allocate_Temporary__(e1tau)\
		xc_mgga_exc_vxc(&func, ngrids, rho.data(), sigma.data(), lapl.data(), tau.data(), e.data(), e1rho.data(), e1sigma.data(), e1lapl.data(), e1tau.data());\
		__Append_XC__(E, e)\
		__Append_XC__(E1Rho, e1rho)\
		__Append_XC__(E1Sigma, e1sigma)\
		__Append_XC__(E1Lapl, e1lapl)\
		__Append_XC__(E1Tau, e1tau)\
	}

void ExchangeCorrelation::Evaluate(std::string order, Grid& grid){
	#pragma omp parallel for schedule(static) num_threads(grid.getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : grid.SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){		
			const int ngrids = subgrid->NumGrids;
			Eigen::Tensor<double, 1>& rho = subgrid->Rho;
			Eigen::Tensor<double, 1>& sigma = subgrid->Sigma;
			Eigen::Tensor<double, 1>& lapl = subgrid->Lapl;
			Eigen::Tensor<double, 1>& tau = subgrid->Tau;
			if ( order == "e" ){
				subgrid->E.setZero();
			}else if ( order == "v" ){
				subgrid->E1Rho.setZero(); subgrid->E1Sigma.setZero(); subgrid->E1Lapl.setZero(); subgrid->E1Tau.setZero();
			}else if ( order == "ev" ){
				subgrid->E.setZero();
				subgrid->E1Rho.setZero(); subgrid->E1Sigma.setZero(); subgrid->E1Lapl.setZero(); subgrid->E1Tau.setZero();
			}else if ( order == "f" ){
				subgrid->E2Rho2.setZero(); subgrid->E2RhoSigma.setZero(); subgrid->E2Sigma2.setZero();
			}/*else if ( order == "k" ){
				subgrid->E3Rho3.setZero(); subgrid->E3Rho2Sigma.setZero(); subgrid->E3RhoSigma2.setZero(); subgrid->E3Sigma3.setZero();
			}else if ( order == "l" ){
				subgrid->E4Rho4.setZero(); subgrid->E4Rho3Sigma.setZero(); subgrid->E4Rho2Sigma2.setZero(); subgrid->E4RhoSigma3.setZero(); subgrid->E4Sigma4.setZero();
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
