void NonIdempotent(int natoms,
                   EigenMatrix * ovlgrads,EigenMatrix * fskeletons,
                   double * repulsion,short int * indices,long int n2integrals,double kscale,
                   EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupancies,
		   EigenMatrix * wxn,EigenMatrix * dxn,EigenVector * exn,
		   const int nprocs,const bool output);

EigenMatrix OccupancyNuclearGradientSCF(double temperature,double * repulsion,short int * indices,long int n2integrals,double kscale,
                                        EigenMatrix * ovlgrads,EigenMatrix * fskeletons,EigenMatrix * dxn,EigenVector * exn,int natoms,
                                        EigenMatrix coefficients,EigenVector occupancies,EigenVector orbitalenergies,
                                        const int nprocs,const bool output);
