int SphericalGridNumber(std::string grid,const int natoms,double * atoms,const bool output);

void SphericalGrid(std::string grid,const int natoms,double * atoms,
                   double * xs,double * ys,double * zs,double * ws,
                   const bool output);

long int UniformBoxGridNumber(const int natoms,double * atoms,const char * basisset,double overheadlength,double spacing);

void UniformBoxGrid(const int natoms,double * atoms,const char * basisset,double overheadlength,double spacing,double * xs,double * ys,double * zs);

void GetAoValues(const int natoms,double * atoms,const char * basisset,
                 double * xs,double * ys,double * zs,int ngrids,
                 double * aos,
                 double * ao1xs,double * ao1ys,double * ao1zs,
                 double * ao2s);

void GetDensity(double * aos,int ngrids,EigenMatrix D,double * density);

void GetDensityGradient(double * aos,double * ao1xs,double * ao1ys,double * ao1zs,int ngrids,EigenMatrix D,double * d1xs,double * d1ys,double * d1zs);

void GetContractedGradient(double * d1xs,double * d1ys,double * d1zs,int ngrids,double * cgs);

void VectorAddition(double * as,double * bs,int ngrids);

double SumUp(double * ds,double * weights,int ngrids);

EigenMatrix VxcMatrix(double * aos,double * vrs,
                      double * d1xs,double * d1ys,double * d1zs,
                      double * ao1xs,double * ao1ys,double * ao1zs,double * vss,
                      double * ws,int ngrids,int nbasis);
