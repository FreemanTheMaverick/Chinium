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
	if ( grid.tensor.size() == 0 ) grid.tensor = tmp;\
	else grid.tensor += tmp;

#define __XC_LDA__\
	assert( rhos.size() == ngrids );\
	if ( order == "e" ){\
		__Allocate_Temporary__(es)\
		xc_lda_exc(&func, ngrids, rhos.data(), es.data());\
		__Append_XC__(Es, es)\
	}else if ( order == "v" ){\
		__Allocate_Temporary__(e1rhos)\
		xc_lda_vxc(&func, ngrids, rhos.data(), e1rhos.data());\
		__Append_XC__(E1Rhos, e1rhos)\
	}else if ( order == "ev" ){\
		__Allocate_Temporary__(es)\
		__Allocate_Temporary__(e1rhos)\
		xc_lda_exc_vxc(&func, ngrids, rhos.data(), es.data(), e1rhos.data());\
		__Append_XC__(Es, es)\
		__Append_XC__(E1Rhos, e1rhos)\
	}else if ( order == "f" ){\
		__Allocate_Temporary__(e2rho2s)\
		xc_lda_fxc(&func, ngrids, rhos.data(), e2rho2s.data());\
		__Append_XC__(E2Rho2s, e2rho2s)\
	}/*else if ( order == "k" ){\
		__Allocate_Temporary__(e3rho3s)\
		xc_lda_kxc(&func, ngrids, rhos.data(), e3rho3s.data());\
		__Append_XC__(E3Rho3s, e3rho3s)\
	}else if ( order == "l" ){\
		__Allocate_Temporary__(e4rho4s)\
		xc_lda_lxc(&func, ngrids, rhos.data(), e4rho4s.data());\
		__Append_XC__(E4Rho4s, e4rho4s)\
	}*/

#define __XC_GGA__\
	assert( rhos.size() == ngrids );\
	assert( sigmas.size() == ngrids );\
	if ( order == "e" ){\
		__Allocate_Temporary__(es)\
		xc_gga_exc(&func, ngrids, rhos.data(), sigmas.data(), es.data());\
		__Append_XC__(Es, es)\
	}else if ( order == "v" ){\
		__Allocate_Temporary__(e1rhos)\
		__Allocate_Temporary__(e1sigmas)\
		xc_gga_vxc(&func, ngrids, rhos.data(), sigmas.data(), e1rhos.data(), e1sigmas.data());\
		__Append_XC__(E1Rhos, e1rhos)\
		__Append_XC__(E1Sigmas, e1sigmas)\
	}else if ( order == "ev" ){\
		__Allocate_Temporary__(es)\
		__Allocate_Temporary__(e1rhos)\
		__Allocate_Temporary__(e1sigmas)\
		xc_gga_exc_vxc(&func, ngrids, rhos.data(), sigmas.data(), es.data(), e1rhos.data(), e1sigmas.data());\
		__Append_XC__(Es, es)\
		__Append_XC__(E1Rhos, e1rhos)\
		__Append_XC__(E1Sigmas, e1sigmas)\
	}else if ( order == "f" ){\
		__Allocate_Temporary__(e2rho2s)\
		__Allocate_Temporary__(e2rhosigmas)\
		__Allocate_Temporary__(e2sigma2s)\
		xc_gga_fxc(&func, ngrids, rhos.data(), sigmas.data(), e2rho2s.data(), e2rhosigmas.data(), e2sigma2s.data());\
		__Append_XC__(E2Rho2s, e2rho2s)\
		__Append_XC__(E2RhoSigmas, e2rhosigmas)\
		__Append_XC__(E2Sigma2s, e2sigma2s)\
	}/*else if ( order == "k" ){\
		__Allocate_Temporary__(e3rho3s)\
		__Allocate_Temporary__(e3rho2sigmas)\
		__Allocate_Temporary__(e3rhosigma2s)\
		__Allocate_Temporary__(e3sigma3s)\
		xc_gga_kxc(&func, ngrids, rhos.data(), sigmas.data(), e3rho3s.data(), e3rho2sigmas.data(), e3rhosigma2s.data(), e3sigma3s.data());\
		__Append_XC__(E3Rho3s, e3rho3s)\
		__Append_XC__(E3Rho2Sigmas, e3rho2sigmas)\
		__Append_XC__(E3RhoSigma2s, e3rhosigma2s)\
		__Append_XC__(E3Sigma3s, e3sigma3s)\
	}else if ( order == "l" ){\
		__Allocate_Temporary__(e4rho4s)\
		__Allocate_Temporary__(e4rho3sigmas)\
		__Allocate_Temporary__(e4rho2sigma2s)\
		__Allocate_Temporary__(e4rhosigma3s)\
		__Allocate_Temporary__(e4sigma4s)\
		xc_gga_lxc(&func, ngrids, rhos.data(), sigmas.data(), e4rho4s.data(), e4rho3sigmas.data(), e4rho2sigma2s.data(), e4rhosigma3s.data(), e4sigma4s.data());\
		__Append_XC__(E4Rho4s, e4rho4s)\
		__Append_XC__(E4Rho3Sigmas, e4rho3sigmas)\
		__Append_XC__(E4Rho2Sigma2s, e4rho2sigma2s)\
		__Append_XC__(E4RhoSigma3s, e4rhosigma3s)\
		__Append_XC__(E4Sigma4s, e4sigma4s)\
	}*/

#define __XC_MGGA__\
	assert( rhos.size() == ngrids );\
	assert( sigmas.size() == ngrids );\
	assert( lapls.size() == ngrids );\
	assert( taus.size() == ngrids );\
	if ( order == "e" ){\
		__Allocate_Temporary__(es)\
		xc_mgga_exc(&func, ngrids, rhos.data(), sigmas.data(), lapls.data(), taus.data(), es.data());\
		__Append_XC__(Es, es)\
	}else if ( order == "v" ){\
		__Allocate_Temporary__(e1rhos)\
		__Allocate_Temporary__(e1sigmas)\
		__Allocate_Temporary__(e1lapls)\
		__Allocate_Temporary__(e1taus)\
		xc_mgga_vxc(&func, ngrids, rhos.data(), sigmas.data(), lapls.data(), taus.data(), e1rhos.data(), e1sigmas.data(), e1lapls.data(), e1taus.data());\
		__Append_XC__(E1Rhos, e1rhos)\
		__Append_XC__(E1Sigmas, e1sigmas)\
		__Append_XC__(E1Lapls, e1lapls)\
		__Append_XC__(E1Taus, e1taus)\
	}else if ( order == "ev" ){\
		__Allocate_Temporary__(es)\
		__Allocate_Temporary__(e1rhos)\
		__Allocate_Temporary__(e1sigmas)\
		__Allocate_Temporary__(e1lapls)\
		__Allocate_Temporary__(e1taus)\
		xc_mgga_exc_vxc(&func, ngrids, rhos.data(), sigmas.data(), lapls.data(), taus.data(), es.data(), e1rhos.data(), e1sigmas.data(), e1lapls.data(), e1taus.data());\
		__Append_XC__(Es, es)\
		__Append_XC__(E1Rhos, e1rhos)\
		__Append_XC__(E1Sigmas, e1sigmas)\
		__Append_XC__(E1Lapls, e1lapls)\
		__Append_XC__(E1Taus, e1taus)\
	}

#define __Alias_Density__(here, there, there_cache)\
	Eigen::Tensor<double, 1>& here = grid.there.size() > 0 ? grid.there : grid.there_cache;

void ExchangeCorrelation::Evaluate(std::string order, Grid& grid){
	const int ngrids = grid.NumGrids;
	Eigen::Tensor<double, 1>& rhos = grid.Rhos;
	Eigen::Tensor<double, 1>& sigmas = grid.Sigmas;
	Eigen::Tensor<double, 1>& lapls = grid.Lapls;
	Eigen::Tensor<double, 1>& taus = grid.Taus;
	if ( order == "e" ){
		grid.Es.setZero();
	}else if ( order == "v" ){
		grid.E1Rhos.setZero(); grid.E1Sigmas.setZero(); grid.E1Lapls.setZero(); grid.E1Taus.setZero();
	}else if ( order == "ev" ){
		grid.Es.setZero();
		grid.E1Rhos.setZero(); grid.E1Sigmas.setZero(); grid.E1Lapls.setZero(); grid.E1Taus.setZero();
	}else if ( order == "f" ){
		grid.E2Rho2s.setZero(); grid.E2RhoSigmas.setZero(); grid.E2Sigma2s.setZero();
	}/*else if ( order == "k" ){
		grid.E3Rho3s.setZero(); grid.E3Rho2Sigmas.setZero(); grid.E3RhoSigma2s.setZero(); E3Sigma3s.setZero();
	}else if ( order == "l" ){
		grid.E4Rho4s.setZero(); grid.E4Rho3Sigmas.setZero(); grid.E4Rho2Sigma2s.setZero(); grid.E4RhoSigma3s.setZero(); grid.E4Sigma4s.setZero();
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
