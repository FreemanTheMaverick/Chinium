EigenMatrix RKSH(
		const int natoms,double * atoms,const char * basisset,
		EigenMatrix D,EigenMatrix * dxn,
		EigenMatrix W,EigenMatrix * wxn,
		EigenMatrix * ovlgrads,EigenMatrix * fskeletons,
		double kscale,
		const int nprocs,const bool output);
