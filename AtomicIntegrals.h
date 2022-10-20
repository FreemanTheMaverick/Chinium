double NuclearRepulsion(const int natoms,double * atoms,const bool output);

int nBasis(const int natoms,double * atoms,const char * basisset,const bool output); // Size of basis set.

int nOneElectronIntegrals(const int natoms,double * atoms,const char * basisset,const bool output); // Number of one-electron integrals.

void Overlap(const int natoms,double * atoms,const char * basisset,double * overlap,const bool output);

void Kinetic(const int natoms,double * atoms,const char * basisset,double * kinetic,const bool output);

void Nuclear(const int natoms,double * atoms,const char * basisset,double * nuclear,const bool output);

void RepulsionDiag(const int natoms,double * atoms,const char * basisset,double * repulsiondiag,const bool output); // Computing the diagonal elements of electron repulsion tensor. Used for Cauchy-Schwarz screening.

long int nTwoElectronIntegrals(const int natoms,double * atoms,const char * basisset,double * repulsiondiag,int & nshellquartets,const bool output); // Numbers of two-electron integrals and nonequivalent shell quartets after Cauchy-Schwarz screening.

void Repulsion(const int natoms,double * atoms,const char * basisset,int nshellquartets,double * repulsiondiag,double * repulsion,short int * indices,const int nprocs,const bool output);


