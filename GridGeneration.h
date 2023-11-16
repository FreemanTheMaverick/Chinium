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
                 double * ao2ls,
                 double * ao2xxs,double * ao2yys,double * ao2zzs,
                 double * ao2xys,double * ao2xzs,double * ao2yzs);
