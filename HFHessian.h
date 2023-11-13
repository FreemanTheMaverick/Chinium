EigenMatrix RKSH(
		const int natoms,double * atoms,const char * basisset,
		EigenMatrix d,EigenMatrix * dxn,
		EigenMatrix w,EigenMatrix * wxn,
		EigenMatrix * ovlgrads,EigenMatrix * fskeletons,
		double kscale,
		const int nprocs,const bool output);
