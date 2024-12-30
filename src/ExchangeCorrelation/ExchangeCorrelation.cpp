extern "C"{
	#include <xc.h>
}
#include <sstream>
#include <fstream>
#include <cstdio>
#include <cassert>
#include <string>
#include "MwfnXC1.h"
#include <iostream>
void ExchangeCorrelation::Read(std::string df, bool output){

	// Parsing df file.
	std::string filename = (std::string)std::getenv("CHINIUM_PATH") + "/DensityFunctionals/" + df + ".df";
	std::ifstream file(filename);
	assert(file.good() && "Density functional file is missing!");
	if (output) std::printf("Reading density functional file %s ...\n", filename.c_str());
	std::string thisline;

	std::getline(file, thisline);
	std::stringstream ss(thisline);
	int x = 0; int c = 0;
	ss>>x; ss>>c;

	if ( x == c){
		if (x){
			this->XCcode = x;
			xc_func_type xfunc;
			xc_func_init(&xfunc, x, XC_UNPOLARIZED);
			this->XCname = xfunc.info->name;
			this->XCkind = xfunc.info->kind;
			switch(xfunc.info->family){
				case(XC_FAMILY_LDA):
					this->XCfamily = "LDA"; break;
				case(XC_FAMILY_GGA):
				case(XC_FAMILY_HYB_GGA):
					this->XCfamily = "GGA"; break;
				case(XC_FAMILY_MGGA):
				case(XC_FAMILY_HYB_MGGA):
					this->XCfamily = "mGGA"; break;
			}
			this->EXX = xc_hyb_exx_coef(&xfunc);
			xc_func_end(&xfunc);
		}
	}else{
		if (x){
			this->Xcode = x;
			xc_func_type xfunc;
			xc_func_init(&xfunc, x, XC_UNPOLARIZED);
			this->Xname = xfunc.info->name;
			this->Xkind = xfunc.info->kind;
			switch(xfunc.info->family){
				case(XC_FAMILY_LDA):
					this->Xfamily = "LDA"; break;
				case(XC_FAMILY_GGA):
				case(XC_FAMILY_HYB_GGA):
					this->Xfamily = "GGA"; break;
				case(XC_FAMILY_MGGA):
				case(XC_FAMILY_HYB_MGGA):
					this->Xfamily = "mGGA"; break;
			}
			this->EXX = xc_hyb_exx_coef(&xfunc);
			xc_func_end(&xfunc);
		}
		if (c){
			this->Ccode = c;
			xc_func_type cfunc;
			xc_func_init(&cfunc, c, XC_UNPOLARIZED);
			this->Cname = cfunc.info->name;
			this->Ckind = cfunc.info->kind;
			switch(cfunc.info->family){
				case(XC_FAMILY_LDA):
					this->Cfamily = "LDA"; break;
				case(XC_FAMILY_GGA):
				case(XC_FAMILY_HYB_GGA):
					this->Cfamily = "GGA"; break;
				case(XC_FAMILY_MGGA):
				case(XC_FAMILY_HYB_MGGA):
					this->Cfamily = "mGGA"; break;
			}
			xc_func_end(&cfunc);
		}
	}
	if (output) this->Print();
}

void ExchangeCorrelation::Print(){
	std::printf("Density functional information:\n");
	if (this->XCcode){
		std::printf("| Exchange-Correlation:\n");
		std::printf("| | Name: %s\n", this->XCname.c_str());
		std::printf("| | Code: %d\n", this->XCcode);
		std::printf("| | Family: %s\n", this->XCfamily.c_str());
		std::printf("| | Exact exchange component: %f\n", this->EXX);
	}else{
		if (this->Xcode){
			std::printf("| Exchange:\n");
			std::printf("| | Name: %s\n", this->Xname.c_str());
			std::printf("| | Code: %d\n", this->Xcode);
			std::printf("| | Family: %s\n", this->Xfamily.c_str());
			std::printf("| | Exact exchange component: %f\n", this->EXX);
		}
		if (this->Ccode){
			std::printf("| Correlation:\n");
			std::printf("| | Name: %s\n", this->Cname.c_str());
			std::printf("| | Code: %d\n", this->Ccode);
			std::printf("| | Family: %s\n", this->Cfamily.c_str());
		}
	}
}

#define __XC_LDA__\
	if ( order.compare("e") == 0 ) xc_lda_exc(&func, ngrids, rhos, es);\
	else if ( order.compare("v") == 0 ) xc_lda_vxc(&func, ngrids, rhos, erhos);\
	else if ( order.compare("ev") == 0 ) xc_lda_exc_vxc(&func, ngrids, rhos, es, erhos);\
	else if ( order.compare("f") == 0 ) xc_lda_fxc(&func, ngrids, rhos, e2rho2s);\
	else if ( order.compare("k") == 0 ) xc_lda_kxc(&func, ngrids, rhos, e3rho3s);\
	else if ( order.compare("l") == 0 ) xc_lda_lxc(&func, ngrids, rhos, e4rho4s);

#define __XC_GGA__\
	if ( order.compare("e") == 0 ) xc_gga_exc(&func, ngrids, rhos, sigmas,  es);\
	else if ( order.compare("v") == 0 ) xc_gga_vxc(&func, ngrids, rhos, sigmas, erhos, esigmas);\
	else if ( order.compare("ev") == 0 ) xc_gga_exc_vxc(&func, ngrids, rhos, sigmas, es, erhos, esigmas);\
	else if ( order.compare("f") == 0 ) xc_gga_fxc(&func, ngrids, rhos, sigmas, e2rho2s, e2rhosigmas, e2sigma2s);\
	else if ( order.compare("k") == 0 ) xc_gga_kxc(&func, ngrids, rhos, sigmas, e3rho3s, e3rho2sigmas, e3rhosigma2s, e3sigma3s);\
	else if ( order.compare("l") == 0 ) xc_gga_lxc(&func, ngrids, rhos, sigmas, e4rho4s, e4rho3sigmas, e4rho2sigma2s, e4rhosigma3s, e4sigma4s);

#define __XC_MGGA__\
	if ( order.compare("e") == 0 ) xc_mgga_exc(&func, ngrids, rhos, sigmas, lapls, taus, es);\
	else if ( order.compare("v") == 0 ) xc_mgga_vxc(&func, ngrids, rhos, sigmas, lapls, taus, erhos, esigmas, elapls, etaus);\
	else if ( order.compare("ev") == 0 ) xc_mgga_exc_vxc(&func, ngrids, rhos, sigmas, lapls, taus, es, erhos, esigmas, elapls, etaus);

void ExchangeCorrelation::Evaluate(
		std::string order, long int ngrids,
		double* rhos, // Input for LDA
		double* sigmas, // Input for GGA
		double* lapls, double* taus, // Input for mGGA
		double* es, // Output epsilon
		double* erhos, double* esigmas, double* elapls, double* etaus, // First-order derivatives
		double* e2rho2s, double* e2rhosigmas, double* e2sigma2s, // Second-order derivatives
		double* e3rho3s, double* e3rho2sigmas, double* e3rhosigma2s, double* e3sigma3s, // Third-order derivatives
		double* e4rho4s, double* e4rho3sigmas, double* e4rho2sigma2s, double* e4rhosigma3s, double* e4sigma4s // Fourth-order derivatives
		){
	if (this->XCcode){
		xc_func_type func;
		xc_func_init(&func, this->XCcode, this->Spin);
		if ( this->XCfamily.compare("LDA") == 0 ){
			__XC_LDA__
		}else if ( this->XCfamily.compare("GGA") == 0 ){
			__XC_GGA__
		}else if ( this->Xfamily.compare("mGGA") == 0 ){
			__XC_MGGA__
		}
		xc_func_end(&func);
	}else{
		xc_func_type func;
		xc_func_init(&func, this->Xcode, this->Spin);
		if (this->Xcode){
			if ( this->Xfamily.compare("LDA") == 0 ){
				__XC_LDA__
			}else if ( this->Xfamily.compare("GGA") == 0 ){
				__XC_GGA__
			}else if ( this->Xfamily.compare("mGGA") == 0 ){
				__XC_MGGA__
			}
		}
		xc_func_end(&func);
		/*
		if (this->Ccode){
			if ( this->Cfamily.compare("LDA") == 0 ){
				__XC_LDA__
			}else if ( this->Cfamily.compare("GGA") == 0 ){
				double* erhos = new double[ngrids];
				double* esigmas = new double[ngrids];
				double* e2rho2s = new double[ngrids];
				double* e2rhosigmas = new double[ngrids];
				double* e2sigma2s = new double[ngrids];
				double* e3rho3s = new double[ngrids];
				double* e3rho2sigmas = new double[ngrids];
				double* e3rhosigma2s = new double[ngrids];
				double* e3sigma3s = new double[ngrids];
				double* e4rho4s = new double[ngrids];
				double* e4rho3sigmas = new double[ngrids];
				double* e4rho2sigma2s = new double[ngrids];
				double* e4rhosigma3s = new double[ngrids];
				double* e4sigma4s = new double[ngrids];
				__XC_GGA__
			}
		}*/
	}
}

/*
int main(){
	std::string df="pbe";
	ExchangeCorrelation(df,1);
}
*/
