EigenMatrix RKSG(const int natoms,double * atoms,const char * basisset,
                 EigenMatrix * ovlgrads,EigenMatrix * hcoregrads,EigenMatrix * fskeletons,
                 double kscale,int ngrids,double * ws,
                 double * aos,
                 double * ao1xs,double * ao1ys,double * ao1zs,
                 double * ao2xxs,double * ao2yys,double * ao2zzs,
                 double * ao2xys,double * ao2xzs,double * ao2yzs,
                 double * d1xs,double * d1ys,double * d1zs,
                 double * vrxcs,double * vsxcs,
                 EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupation,
                 const int nprocs,const bool output);

EigenMatrix RHFG(const int natoms,double * atoms,const char * basisset,
                 EigenMatrix * ovlgrads,EigenMatrix * hcoregrads,EigenMatrix * fskeletons,
                 EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupation,
                 const int nprocs,const bool output);
