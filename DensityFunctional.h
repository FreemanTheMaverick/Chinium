void XCInfo(int id,char * name,int & kind,int & family,double & hf);

void ReadDF(std::string df,int & x,int & c,double & hf,char & approx,const bool output);

void getVxc(int id,double * ds,double * cgs,double * d2s,double * ts,int ngrids,double * vrs,double * vss,double * vls,double * vts);

void getEVxc(int id,double * ds,double * cgs,double * d2s,double * ts,int ngrids,double * es,double * vrs,double * vss,double * vls,double * vts);

void getFxc(
		int id,int ngrids,
		double * ds, // Extra input for LDA
		double * cgs, // Extra input for GGA
		double * d2s,double * ts, // Extra input for mGGA
		double * vr2s, // Output for LDA
		double * vrss,double * vs2s, // Extra output for GGA
		double * vrls,double * vrts,double * vsls,double * vsts,double * vl2s,double * vlts,double * vt2s); // Extra output for mGGA
 
