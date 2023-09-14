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
                 double * ao2ls);

void GetDensity(double * aos,
                double * ao1xs,double * ao1ys,double * ao1zs,
                double * ao2ls,
                int ngrids,EigenMatrix D,
                double * ds,
                double * d1xs,double * d1ys,double * d1zs,double * cgs,
                double * d2s,double * ts);

void VectorAddition(double * as,double * bs,int ngrids);

double SumUp(double * ds,double * weights,int ngrids);

EigenMatrix FxcMatrix(double * aos,double * vrs,
                      double * d1xs,double * d1ys,double * d1zs,
                      double * ao1xs,double * ao1ys,double * ao1zs,double * vss,
                      double * ao2ls,double * vls,double * vts,
                      double * ws,int ngrids,int nbasis);
