void PurifyDensity(EigenMatrix overlap,EigenMatrix & density);

double RKS(int nele,EigenMatrix overlap,EigenMatrix hcore,
            double * repulsion,short int * indices,long int n2integrals,
            int dfxid,int dfcid,int ngrids,double * gridweights,
            double * gridaos,
            double * gridao1xs,double * gridao1ys,double * gridao1zs,
            double * gridao2s,
            EigenVector & orbitalenergies,EigenMatrix & coefficients,
            EigenMatrix & density,EigenMatrix & fock,
            const int nprocs,const bool output);

double RHF(int nele,EigenMatrix overlap,EigenMatrix hcore,
           double * repulsion,short int * indices,long int n2integrals,
           EigenVector & orbitalenergies,EigenMatrix & coefficients,
           EigenMatrix & density,EigenMatrix & fock,
           const int nprocs,const bool output);
