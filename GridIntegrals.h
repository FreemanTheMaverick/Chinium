void CalculateDensity(double * xxx,double * yyy,double * zzz,long int size,EigenMatrix D,libint2::BasisSet obs,double * results);

double UniformBoxGridDensity(const int natoms,double * atoms,const char * basisset,EigenMatrix D,double overheadlength,int griddensity);
