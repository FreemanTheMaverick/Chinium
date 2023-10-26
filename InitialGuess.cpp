#include <Eigen/Dense>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <cassert>
#include <vector>
#include "Aliases.h"
#include "AtomicIntegrals.h"
#include "HartreeFock.h"
#include "GridIntegrals.h"

EigenMatrix SuperpositionAtomicDensity(const int natoms,double * atoms,const char * basisset){
	const int nbasis=nBasis(natoms,atoms,basisset,0);
	EigenMatrix density=EigenZero(nbasis,nbasis); // Initializing the density matrix.
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
		EigenMatrix atomicdensity=EigenZero(nbasis,nbasis);
		const EigenMatrix overlap=Overlap(1,atom,basisset,0);
		const EigenMatrix kinetic=Kinetic(1,atom,basisset,0);
		const EigenMatrix nuclear=Nuclear(1,atom,basisset,0);
		const EigenMatrix hcore=kinetic+nuclear;
		const EigenMatrix repulsiondiag=RepulsionDiag(1,atom,basisset,0);
		int nshellquartets;
		const int n2integrals=nTwoElectronIntegrals(1,atom,basisset,repulsiondiag,nshellquartets,0);
		double * repulsion=new double[n2integrals];
		short int * indices=new short int[n2integrals*5];
		Repulsion(1,atom,basisset,nshellquartets,repulsiondiag,n2integrals,repulsion,indices,1,0);
		EigenVector orbitalenergies(nbasis);
		EigenVector occupancies(nbasis);
		EigenMatrix coefficients(nbasis,nbasis);
		EigenMatrix dummy;
		RHF(ne,0./0.,0./0.,overlap,hcore,repulsion,indices,n2integrals,orbitalenergies,coefficients,occupancies,atomicdensity,dummy,1,0);
		delete [] repulsion;
		delete [] indices;
		for (int & ibasis:unique_atom.second)
			density.block(ibasis,ibasis,nbasis,nbasis)=atomicdensity/ne*unique_atom.first;
	}
	return density;
}

#define __nlines__ 751

EigenMatrix SuperpositionAtomicPotential(const int natoms,double * atoms,int nbasis,
                                         double * xs,double * ys,double * zs,double * ws,int ngrids,double * aos){
	__Z_2_Name__
	double * vsap=new double[ngrids]();
	for (int iatom=0;iatom<natoms;iatom++){
		const std::string atomname=Z2Name[(int)atoms[iatom*4]];
		const double atomx=atoms[iatom*4+1];
		const double atomy=atoms[iatom*4+2];
		const double atomz=atoms[iatom*4+3];
		std::ifstream sapfile(std::string(__SAP_library_path__)+"/v_"+atomname+".dat");
		assert("Missing element SAP file in" __SAP_library_path__ && sapfile.good());
		double Rs[__nlines__];
		double Zs[__nlines__];
		for (int iline=0;iline<__nlines__;iline++){
			std::string thisline;
			getline(sapfile,thisline);
			std::stringstream ss(thisline);
			ss>>Rs[iline];
			ss>>Zs[iline];
		}
		for (int igrid=0;igrid<ngrids;igrid++){
			const double x=xs[igrid]-atomx;
			const double y=ys[igrid]-atomy;
			const double z=zs[igrid]-atomz;
			const double r=sqrt(x*x+y*y+z*z)+1.e-12;
			double bestdiff=114514;
			double bestvap=114514;
			for (int iline=0;iline<__nlines__;iline++){
				bestdiff=(bestdiff<abs(Rs[iline]-r))?bestdiff:abs(Rs[iline]-r);
				bestvap=(bestdiff<abs(Rs[iline]-r))?bestvap:Zs[iline]/r;
			}
			vsap[igrid]+=bestvap;
		}
	}
	return FxcMatrix(aos,vsap,
	                 nullptr,nullptr,nullptr,
	                 nullptr,nullptr,nullptr,nullptr,
	                 nullptr,nullptr,nullptr,
	                 ws,ngrids,nbasis);
}
