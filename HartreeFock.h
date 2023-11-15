void PurifyDensity(EigenMatrix overlap,EigenMatrix & density);

EigenMatrix GhfMatrix(double * repulsion,short int * indices,long int n2integrals,EigenMatrix D,double kscale,const int nprocs);

EigenMatrix GxcMatrix(
		EigenMatrix D,
		int dfxid,int dfcid,int ngrids,double * ws,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2ls,
		double * ds,double * d2s,double * ts,double * cgs,
		double * Exc_ptr,
		const int nprocs);

double RKS(int nele,double temperature,double chemicalpotential,
           EigenMatrix overlap,EigenMatrix hcore,
           double * repulsion,short int * indices,long int n2integrals,
           int dfxid,int dfcid,int ngrids,double * gridweights,
           double * aos,
           double * ao1xs,double * ao1ys,double * ao1zs,
           double * ao2ls,
           EigenVector & orbitalenergies,EigenMatrix & coefficients,
           EigenVector & occupancies,EigenMatrix & D,EigenMatrix & F,
           const int nprocs,const bool output);

double RHF(int nele,double temperature,double chemicalpotential,
           EigenMatrix overlap,EigenMatrix hcore,
           double * repulsion,short int * indices,long int n2integrals,
           EigenVector & orbitalenergies,EigenMatrix & coefficients,
           EigenVector & occupancies,EigenMatrix & D,EigenMatrix & F,
           const int nprocs,const bool output);
