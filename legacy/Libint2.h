int nShell(const int natoms,double * atoms,const char * basisset,const bool output);

int nBasis(const int natoms,double * atoms,const char * basisset,const bool output);

int nPrim(const int natoms,double * atoms,const char * basisset,const bool output);

int nPrimShell(const int natoms,double * atoms,const char * basisset,const bool output);
/*
void BF2Shell(const int natoms,double * atoms,const char * basisset,short int * bf2shell);

void BF2Atom(const int natoms,double * atoms,const char * basisset,short int * bf2atom);
*/
void ShellInfo(
		const int natoms,double * atoms,const char * basisset,
		int * bf2shell,int * bf2atom,
		int * shell2type,int * shell2atom,int * shell2cd,
		double * primexp,double * primcontr);
