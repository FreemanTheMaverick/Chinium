long int UniformBoxGridNumber(const int natoms,double * atoms,const char * basisset,double overheadlength,double spacing);

void UniformBoxGrid(const int natoms,double * atoms,const char * basisset,double overheadlength,double spacing,double * xs,double * ys,double * zs);

void GetAoValues(const int natoms,double * atoms,const char * basisset,double * xs,double * ys,double * zs,long int size,double * results);

void GetDensity(double * aos,long int ngrids,EigenMatrix D,double * density);

double SumUp(double * ds,double * weights,long int ngrids);

EigenMatrix VxcMatrix(double * aos,double * weights,double * vrs,long int ngrids,int nbasis);
