void GhfSkeletons(
		const int natoms,double * atoms,const char * basisset,
		EigenMatrix D,
		double kscale,
		EigenMatrix * ghfskeletons);

void GxcSkeletons(
		const int natoms,double * atoms,const char * basisset,
		EigenMatrix D,
		int dfxid,int dfcid,int ngrids,double * ws,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2xxs,double * ao2yys,double * ao2zzs,
		double * ao2xys,double * ao2xzs,double * ao2yzs,
		EigenMatrix * gxcskeletons);
 
EigenMatrix RKSG(
		const int natoms,double * atoms,const char * basisset,
		EigenMatrix * ovlgrads,EigenMatrix * hcoregrads,EigenMatrix * fskeletons,
		int dfxid,int dfcid,int ngrids,double * ws,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2xxs,double * ao2yys,double * ao2zzs,
		double * ao2xys,double * ao2xzs,double * ao2yzs,
		EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupancies,
		const int nprocs,const bool output);

EigenMatrix RHFG(
		const int natoms,double * atoms,const char * basisset,
		EigenMatrix * ovlgrads,EigenMatrix * hcoregrads,EigenMatrix * fskeletons,
		EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupancies,
		const int nprocs,const bool output);
