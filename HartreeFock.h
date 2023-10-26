void PurifyDensity(EigenMatrix overlap,EigenMatrix & density);

EigenMatrix GMatrix(double * repulsion,short int * indices,long int n2integrals,EigenMatrix D,double kscale,const int nprocs);

double RKS(int nele,double temperature,double chemicalpotential,
           EigenMatrix overlap,EigenMatrix hcore,
           double * repulsion,short int * indices,long int n2integrals,
           int dfxid,int dfcid,int ngrids,double * gridweights,
           double * aos,
           double * ao1xs,double * ao1ys,double * ao1zs,
           double * ao2ls,
           double *& d1xs,double *& d1ys,double *& d1zs,
           double *& vrxcs,double *& vsxcs,
           EigenVector & orbitalenergies,EigenMatrix & coefficients,
           EigenVector & occupancies,EigenMatrix & D,EigenMatrix & F,
           const int nprocs,const bool output);

double RHF(int nele,double temperature,double chemicalpotential,
           EigenMatrix overlap,EigenMatrix hcore,
           double * repulsion,short int * indices,long int n2integrals,
           EigenVector & orbitalenergies,EigenMatrix & coefficients,
           EigenVector & occupancies,EigenMatrix & D,EigenMatrix & F,
           const int nprocs,const bool output);
