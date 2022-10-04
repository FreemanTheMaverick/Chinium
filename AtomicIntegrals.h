double NuclearRepulsion(const int natoms,double * atoms);

int nBasis(const int natoms,double * atoms,const char * basisset); // Size of basis set.

int nOneElectronIntegrals(const int natoms,double * atoms,const char * basisset); // Number of one-electron integrals.

void Overlap(const int natoms,double * atoms,const char * basisset,double * overlap);

void Kinetic(const int natoms,double * atoms,const char * basisset,double * kinetic);

void Nuclear(const int natoms,double * atoms,const char * basisset,double * nuclear);

void RepulsionDiag(const int natoms,double * atoms,const char * basisset,double * repulsiondiag); // Computing the diagonal elements of electron repulsion tensor. Used for Cauchy-Schwarz screening.

long int nTwoElectronIntegrals(const int natoms,double * atoms,const char * basisset,double * repulsiondiag,int & nshellquartets); // Numbers of two-electron integrals and nonequivalent shell quartets after Cauchy-Schwarz screening.

void Repulsion(const int natoms,double * atoms,const char * basisset,int nshellquartets,double * repulsiondiag,double * repulsion,short int * indices);


