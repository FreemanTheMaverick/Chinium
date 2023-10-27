void NonIdempotent(int natoms,
                   EigenMatrix * ovlgrads,EigenMatrix * fskeletons,
                   double * repulsion,short int * indices,long int n2integrals,double kscale,
                   EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupancies,
		   EigenMatrix * wxn,EigenMatrix * dxn,
		   const int nprocs,const bool output);

