double NuclearRepulsion(const int natoms,double * atoms,const bool output);

int nBasis(const int natoms,double * atoms,const char * basisset,const bool output);

int nOneElectronIntegrals(const int natoms,double * atoms,const char * basisset,const bool output);

EigenMatrix Overlap(const int natoms,double * atoms,const char * basisset,const bool output);

EigenMatrix Kinetic(const int natoms,double * atoms,const char * basisset,const bool output);

EigenMatrix Nuclear(const int natoms,double * atoms,const char * basisset,const bool output);

EigenMatrix RepulsionDiag(const int natoms,double * atoms,const char * basisset,const bool output);

long int nTwoElectronIntegrals(const int natoms,double * atoms,const char * basisset,EigenMatrix repulsiondiag,int & nshellquartets,const bool output);

void Repulsion(const int natoms,double * atoms,const char * basisset,int nshellquartets,EigenMatrix repulsiondiag,double * repulsion,short int * indices,const int nprocs,const bool output);


