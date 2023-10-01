#include <libint2.hpp>
#include <vector>
#include "Aliases.h"

void BF2Shell(const int natoms,double * atoms,const char * basisset,short int * bf2shell){
	__Basis_From_Atoms__
	__nBasis_From_OBS__
	for (int ishell=0,kbasis=0;ishell<(int)obs.size();ishell++)
		for (int jbasis=0;jbasis<(int)obs[ishell].size();jbasis++,kbasis++)
			bf2shell[kbasis]=ishell;
}

void BF2Atom(const int natoms,double * atoms,const char * basisset,short int * bf2atom){
	__Basis_From_Atoms__
	__nBasis_From_OBS__
	for (int ishell=0,kbasis=0;ishell<(int)obs.size();ishell++)
		for (int jbasis=0;jbasis<(int)obs[ishell].size();jbasis++,kbasis++)
			bf2atom[kbasis]=obs.shell2atom(libint2atoms)[ishell];
}
