extern "C"{
	#include <xc.h>
}
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstring>
#include <string>

void XCInfo(int id,char * name,int & kind,int & family,double & hf){
	xc_func_type func;
	xc_func_init(&func,id,XC_UNPOLARIZED);
	std::strcpy(name,func.info->name);
	kind=func.info->kind;
	family=func.info->family;
	hf=xc_hyb_exx_coef(&func);
	xc_func_end(&func);
}

void ReadDF(std::string df,int & x,int & c,double & hf,char & approx,const bool output){
	std::string filename=std::string(__DF_library_path__)+"/"+df+".df";
	std::ifstream file(filename);
	std::string thisline;

	getline(file,thisline);
	std::stringstream ss(thisline);
	int xx,cc;
	ss>>xx;x=xx;
	ss>>cc;c=cc;

	char xname[64],cname[64];
	int xkind,ckind,xfamily,cfamily;
	if(cc!=0) XCInfo(cc,cname,ckind,cfamily,hf);
	XCInfo(xx,xname,xkind,xfamily,hf); // Correlation functional is obtained firstly and exchange functional lastly so that 'hf' is finally the correct value. After all, a correlation functional's "HF component" is always zero.
	switch (xfamily){
		case(1): // XC_FAMILY_LDA
			approx='l';break;
		case(2): // XC_FAMILY_GGA
		case(32): // XC_FAMILY_HYB_GGA
			approx='g';break;
		case(4): // XC_FAMILY_MGGA
		case(64): // XC_FAMILY_HYB_MGGA
			approx='m';break;
	}

	if (output){
		std::cout<<"Density functional info:"<<std::endl;
		std::cout<<"| Name: "<<df<<std::endl;
		switch (xfamily){
			case(1): // XC_FAMILY_LDA
				std::cout<<"| Type: lda"<<std::endl;break;
			case(2): // XC_FAMILY_GGA
				std::cout<<"| Type: gga"<<std::endl;break;
			case(32): // XC_FAMILY_HYB_GGA
				std::cout<<"| Type: hybrid gga"<<std::endl;break;
			case(4): // XC_FAMILY_MGGA
				std::cout<<"| Type: mgga"<<std::endl;break;
			case(64): // XC_FAMILY_HYB_MGGA
				std::cout<<"| Type: hybrid mgga"<<std::endl;break;
			default:
				std::cout<<"| Type: unknown"<<std::endl;break;
		}
		if (xx==cc)
			std::cout<<"| Components: XC = "<<xname<<(hf>0?" ; HF = "+std::to_string(hf):"")<<std::endl;
		else{
			if (cc==0)
				std::cout<<"| Components: X = "<<xname<<(hf>0?" ; HF = "+std::to_string(hf):"")<<std::endl;
			else
				std::cout<<"| Components: X = "<<xname<<(hf>0?" ; HF = "+std::to_string(hf):"")<<" ; C = "<<cname<<std::endl;
		}
	}
}

void getEVxc(int id,double * ds,double * cgs,double * d2s,double * ts,int ngrids,double * es,double * vrs,double * vss,double * vls,double * vts){
        xc_func_type func;
        xc_func_init(&func,id,XC_UNPOLARIZED);
	switch(func.info->family){
		case XC_FAMILY_LDA:
			xc_lda_exc_vxc(&func,ngrids,ds,es,vrs);
			break;
		case XC_FAMILY_GGA:
		case XC_FAMILY_HYB_GGA:
			xc_gga_exc_vxc(&func,ngrids,ds,cgs,es,vrs,vss);
			break;
		case XC_FAMILY_MGGA:
		case XC_FAMILY_HYB_MGGA:
			xc_mgga_exc_vxc(&func,ngrids,ds,cgs,d2s,ts,es,vrs,vss,vls,vts);
			break;
	}
	xc_func_end(&func);
	for (long int igrid=0;igrid<ngrids;igrid++)
		es[igrid]*=ds[igrid];
}

/*
int main(){
	std::string df="m06-2x";
	int x,c;
	double hf;
	char approx;
	ReadDF(df,x,c,hf,approx,1);
}
*/
