EigenMatrix FockOccupationGradientCPSCF(
		double temperature,double * repulsion,short int * indices,long int n2integrals,double kscale,
		EigenMatrix * ovlgrads,EigenMatrix * fskeletons,EigenMatrix * Dxns,EigenVector * exns,int natoms,
		EigenMatrix coefficients,EigenVector occupancies,EigenVector orbitalenergies,
		const int nprocs,const bool output);

EigenMatrix DensityOccupationGradientCPSCF(
		int natoms,int * bf2atom,double temperature,
		EigenMatrix * ovlgrads,EigenMatrix * fskeletons,EigenMatrix * Dxns,EigenVector * exns,
		double * repulsion,short int * indices,long int n2integrals,
		int dfxid,int dfcid,int ngrids,double * ws,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2xxs,double * ao2yys,double * ao2zzs,
		double * ao2xys,double * ao2xzs,double * ao2yzs,
		EigenMatrix coefficients,EigenVector occupancies,EigenVector orbitalenergies,
		const int nprocs,const bool output);
