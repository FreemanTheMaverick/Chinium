void NonIdempotentCPSCF(int natoms,short int * bf2atoms,
                        EigenMatrix * ovlgrads,EigenMatrix * fskeletons,
                        double * repulsion,short int * indices,long int n2integrals,
			int dfxid,int dfcid,int ngrids,double * ws,
			double * aos,
			double * ao1xs,double * ao1ys,double * ao1zs,
			double * ao2xxs,double * ao2yys,double * ao2zzs,
			double * ao2xys,double * ao2xzs,double * ao2yzs,
                        EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupancies,
		        EigenMatrix * wxn,EigenMatrix * dxn,EigenVector * exn,
		        const int nprocs,const bool output);

EigenMatrix FockOccupationGradientCPSCF(
		double temperature,double * repulsion,short int * indices,long int n2integrals,double kscale,
		EigenMatrix * ovlgrads,EigenMatrix * fskeletons,EigenMatrix * dxn,EigenVector * exn,int natoms,
		EigenMatrix coefficients,EigenVector occupancies,EigenVector orbitalenergies,
		const int nprocs,const bool output);

EigenMatrix DensityOccupationGradientCPSCF(
		double temperature,double * repulsion,short int * indices,long int n2integrals,double kscale,
		EigenMatrix * ovlgrads,EigenMatrix * fskeletons,EigenMatrix * dxn,EigenVector * exn,int natoms,
		EigenMatrix coefficients,EigenVector occupancies,EigenVector orbitalenergies,
		const int nprocs,const bool output);
	
