void XCInfo(int id,char * name,int & kind,int & family,double & hf);

void ReadDF(std::string df,int & x,int & c,double & hf,char & approx,const bool output);

void getVxc(int id,double * ds,double * cgs,double * d2s,double * ts,int ngrids,double * vrs,double * vss,double * vls,double * vts);

void getEVxc(int id,double * ds,double * cgs,double * d2s,double * ts,int ngrids,double * es,double * vrs,double * vss,double * vls,double * vts);
