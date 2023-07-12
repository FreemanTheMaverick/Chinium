#include <Eigen/Dense>
#include <map>
#include <vector>
#include <iostream>
#include "Aliases.h"
#include "AtomicIntegrals.h"
#include "HartreeFock.h"

EigenMatrix CoreHamiltonian(const int natoms,double * atoms,const char * basisset){
	const int nbasis=nBasis(natoms,atoms,basisset,0);
	return EigenZero(nbasis,nbasis);
}

EigenMatrix SuperpositionAtomicDensity(int nele,const int natoms,double * atoms,const char * basisset){
	EigenMatrix densitymatrix=CoreHamiltonian(natoms,atoms,basisset); // Initializing the density matrix.
	std::map<int,std::vector<int>> unique_atoms; // A map to store all unique atoms and their leading basis functions' indices.
	int ibasis=0; // The leading basis function's index.
	for (int iatom=0;iatom<natoms;iatom++){
		int Z=(int)atoms[4*iatom];
		double atom[]={(double)Z,0,0,0};
		bool unique=1;
		for (std::pair<const int,std::vector<int>> & unique_atom:unique_atoms) // Looping over all unique atoms in the map to see whether this atom is a new one.
			if (unique_atom.first==Z) unique=0; // An identical atom is found in the map, meaning that this atom is not new.
		if (unique) unique_atoms.insert(std::pair<int,std::vector<int>>(Z,std::vector<int>{ibasis})); // Adding a new unique atom and its first leading basis function's index.
		else unique_atoms[Z].push_back(ibasis); // Adding this leading basis function's index to the corresponding unique atom.
		ibasis+=nBasis(1,atom,basisset,0);
	}
	for (std::pair<const int,std::vector<int>> & unique_atom:unique_atoms){ // Every unique atom will go through a RHF procedure for its relaxed atomic density matrix.
		double atom[]={(double)unique_atom.first,0,0,0};
		const int ne=(unique_atom.first%2==0)?unique_atom.first:(unique_atom.first+1); // Since only RHF is supported now, the number of electron must be set to an even number.
		const int nbasis=nBasis(1,atom,basisset,0);
		EigenMatrix atomicdensitymatrix=EigenZero(nbasis,nbasis);
		const EigenMatrix overlap=Overlap(1,atom,basisset,0);
		const EigenMatrix kinetic=Kinetic(1,atom,basisset,0);
		const EigenMatrix nuclear=Nuclear(1,atom,basisset,0);
		const EigenMatrix hcore=kinetic+nuclear;
		const EigenMatrix repulsiondiag=RepulsionDiag(1,atom,basisset,0);
		int nshellquartets;
		const int n2integrals=nTwoElectronIntegrals(1,atom,basisset,repulsiondiag,nshellquartets,0);
		double * repulsion=new double[n2integrals];
		short int * indices=new short int[n2integrals*5];
		Repulsion(1,atom,basisset,nshellquartets,repulsiondiag,repulsion,indices,1,0);
		EigenMatrix orbitalenergies(1,nbasis);
		EigenMatrix coefficients(nbasis,nbasis);
		RHF(ne,overlap,hcore,repulsion,indices,n2integrals,orbitalenergies,coefficients,atomicdensitymatrix,1,0);
		delete [] repulsion;
		delete [] indices;
		for (int & ibasis:unique_atom.second)
			densitymatrix.block(ibasis,ibasis,nbasis,nbasis)=atomicdensitymatrix;
	}
	return densitymatrix;
}

/*
void SuperpositionAtomicDensity(int nele,const int natoms,double * atoms,const char * basisset,double * densitymatrix){
	int atomicconfiguration[][6]={{ 0, 0, 0, 0, 0, 0}, // Fuck indices from 0
	                              { 1, 0, 0, 0, 0, 0}, // H
	                              { 2, 0, 0, 0, 0, 0}, // He
	                              { 3, 0, 0, 0, 0, 0}, // Li
	                              { 4, 0, 0, 0, 0, 0}, // Be
	                              { 4, 1, 0, 0, 0, 0}, // B
	                              { 4, 2, 0, 0, 0, 0}, // C
	                              { 4, 3, 0, 0, 0, 0}, // N
	                              { 4, 4, 0, 0, 0, 0}, // O
	                              { 4, 5, 0, 0, 0, 0}, // F
	                              { 4, 6, 0, 0, 0, 0}, // Ne
	                              { 5, 6, 0, 0, 0, 0}, // Na
	                              { 6, 6, 0, 0, 0, 0}, // Mg
	                              { 6, 7, 0, 0, 0, 0}, // Al
	                              { 6, 8, 0, 0, 0, 0}, // Si
	                              { 6, 9, 0, 0, 0, 0}, // P
	                              { 6,10, 0, 0, 0, 0}, // S
	                              { 6,11, 0, 0, 0, 0}, // Cl
	                              { 6,12, 0, 0, 0, 0}, // Ar
	                              { 7,12, 0, 0, 0, 0}, // K
	                              { 8,12, 0, 0, 0, 0}}; // Ca
	std::vector<libint2::Atom> libint2atoms=Libint2Atoms_for_InitialGuess(natoms,atoms);
	int Ztotal=0;
	for (int iatom=0;iatom<natoms;iatom++){
		libint2::Atom atomi=libint2atoms[iatom];
		Ztotal+=atomi.atomic_number;
	}
	double scalingfactor=(double)nele/Ztotal;
	libint2::BasisSet obs(basisset,libint2atoms);
	int nbasis=nBasis_for_InitialGuess(obs);
	double densitymatrixdiag[nbasis];
	for (int i=0;i<nbasis;i++) densitymatrixdiag[i]=0;
	double * densitymatrixdiagranger=densitymatrixdiag;
	int iatom=0;
	double lastx=obs[0].O[0];
	double lasty=obs[0].O[1];
	double lastz=obs[0].O[2];
	int lastl=0;
	std::vector<double> similar_shell_effective_alpha;
	for (int ishell=0;ishell<(int)obs.size();ishell++){
		libint2::Shell shell=obs[ishell];
		bool sameatom=(lastx==shell.O[0] && lasty==shell.O[1] && lastz==shell.O[2]);
		bool samel=(lastl==shell.contr[0].l);
		bool sameshell=sameatom&&samel;
		double alpha_with_largest_coeff=114514;
		double largest_coeff=-114514;
		if (! sameshell){
			int l=lastl;
			double neleinshell_unrelaxed=scalingfactor*atomicconfiguration[libint2atoms[iatom].atomic_number][l];
			for (double & effective_alpha:similar_shell_effective_alpha)
				for (int i=0;i<(2*l+1);i++)
					*(densitymatrixdiagranger++)=neleinshell_unrelaxed*effective_alpha/std::accumulate(similar_shell_effective_alpha.begin(),similar_shell_effective_alpha.end(),decltype(similar_shell_effective_alpha)::value_type(0))/(2*l+1);
			similar_shell_effective_alpha.clear();
			if (! sameatom) iatom++;
		}
		for (int iprimitive=0;iprimitive<(int)shell.alpha.size();iprimitive++){
			largest_coeff=largest_coeff>abs(shell.contr[0].coeff[iprimitive])?largest_coeff:abs(shell.contr[0].coeff[iprimitive]);
			alpha_with_largest_coeff=largest_coeff>abs(shell.contr[0].coeff[iprimitive])?alpha_with_largest_coeff:shell.alpha[iprimitive];
		}
		similar_shell_effective_alpha.push_back(alpha_with_largest_coeff);
		if ((ishell+1)==(int)obs.size()){
			int l=shell.contr[0].l;
			double neleinshell_unrelaxed=scalingfactor*atomicconfiguration[libint2atoms[iatom].atomic_number][l];
			for (double & effective_alpha:similar_shell_effective_alpha)
				for (int i=0;i<(2*l+1);i++)
					*(densitymatrixdiagranger++)=neleinshell_unrelaxed*effective_alpha/std::accumulate(similar_shell_effective_alpha.begin(),similar_shell_effective_alpha.end(),decltype(similar_shell_effective_alpha)::value_type(0))/(2*l+1);
		}


		lastx=shell.O[0];
		lasty=shell.O[1];
		lastz=shell.O[2];
		lastl=shell.contr[0].l;
	}
	for (int i=0;i<nbasis;i++)
		for (int j=0;j<=i;j++)
			densitymatrix[i*(i+1)/2+j]=(i==j)?densitymatrixdiag[i]:0;
}
*/





