#include <libint2.hpp>
#include <vector>
#include "Aliases.h"

int nShell(const int natoms,double * atoms,const char * basisset,const bool output){ // Size of basis set.
	__Basis_From_Atoms__
	if (output) std::cout<<"Number of shells ... "<<obs.size()<<std::endl;
	return obs.size();
}

int nBasis(const int natoms,double * atoms,const char * basisset,const bool output){ // Size of basis set.
	__Basis_From_Atoms__
	__nBasis_From_OBS__
	if (output) std::cout<<"Number of atomic bases ... "<<nbasis<<std::endl;
	return nbasis;
}

int nPrim(const int natoms,double * atoms,const char * basisset,const bool output){ // Size of basis set.
	__Basis_From_Atoms__
	__nPrim_From_OBS__
	if (output) std::cout<<"Number of primitive gaussians ... "<<nprim<<std::endl;
	return nprim;
}


int nPrimShell(const int natoms,double * atoms,const char * basisset,const bool output){ // Size of basis set.
	__Basis_From_Atoms__
	__nPrimShell_From_OBS__
	if (output) std::cout<<"Number of primitive shells ... "<<nprimshell<<std::endl;
	return nprimshell;
}

void ShellInfo(
		const int natoms,double * atoms,const char * basisset,
		int * bf2shell,int * bf2atom,
		int * shell2type,int * shell2atom,int * shell2cd,
		double * primexp,double * primcontr){
	__Basis_From_Atoms__
	__nBasis_From_OBS__
	__nPrim_From_OBS__
	for (int ishell=0,kbasis=0,kprim=0;ishell<(int)obs.size();ishell++){
		if (shell2type) shell2type[ishell]=(int)(obs[ishell].size()-1)/2;
		if (shell2atom) shell2atom[ishell]=(int)(obs.shell2atom(libint2atoms)[ishell]);
		if (shell2cd) shell2cd[ishell]=(int)(obs[ishell].nprim());
		for (int jbasis=0;jbasis<(int)obs[ishell].size();jbasis++,kbasis++){
			if (bf2shell) bf2shell[kbasis]=ishell;
			if (bf2atom) bf2atom[kbasis]=obs.shell2atom(libint2atoms)[ishell];
		}
		for (int jprim=0;jprim<(int)obs[ishell].nprim();jprim++,kprim++){
			if (primexp) primexp[kprim]=(double)obs[ishell].alpha[jprim];
			if (primcontr) primcontr[kprim]=(double)obs[ishell].coeff_normalized(0,jprim);
		}
	}
}
