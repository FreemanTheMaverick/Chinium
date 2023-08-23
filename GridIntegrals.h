long int SphericalGridNumber(std::string grid,const int natoms,double * atoms,const bool output);

void SphericalGrid(std::string grid,const int natoms,double * atoms,
                   double * xs,double * ys,double * zs,double * ws,
                   const bool output);

long int UniformBoxGridNumber(const int natoms,double * atoms,const char * basisset,double overheadlength,double spacing);

void UniformBoxGrid(const int natoms,double * atoms,const char * basisset,double overheadlength,double spacing,double * xs,double * ys,double * zs);

void GetAoValues(const int natoms,double * atoms,const char * basisset,
                 double * xs,double * ys,double * zs,long int size,
                 double * aos,
                 double * ao1xs,double * ao1ys,double * ao1zs);

void GetDensity(double * aos,long int ngrids,EigenMatrix D,double * density);

void GetContractedGradient(double * aos,double * ao1xs,double * ao1ys,double * ao1zs,long int ngrids,EigenMatrix D,double * cgs);

void VectorAddition(double * as,double * bs,long int ngrids);

double SumUp(double * ds,double * weights,long int ngrids);

EigenMatrix VxcMatrix(double * aos,double * weights,double * vrs,long int ngrids,int nbasis);
