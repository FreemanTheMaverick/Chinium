#include <Eigen/Dense>
#include <string>
#include <iostream>
#include "Aliases.h"
#include "Gateway.h"
#include "AtomicIntegrals.h"
#include "HartreeFock.h"
#include "InitialGuess.h"
#include "GridIntegrals.h"

int main(int argc,char *argv[]){

	std::cout<<"*** Chinium started ***"<<std::endl;
	const int nprocs=ReadNProcs(argv[1],1);
	double atoms[10000];
	const int natoms=ReadXYZ(argv[1],atoms,1);
	const std::string basisset_=ReadBasisSet(argv[1],1);
	const char * basisset=basisset_.data();
	const int ne=ReadNElectrons(argv[1],1);
	const int nbasis=nBasis(natoms,atoms,basisset,1);
	nOneElectronIntegrals(natoms,atoms,basisset,1);
	const double nuclearrepulsion=NuclearRepulsion(natoms,atoms,1);

	const std::string guess=ReadGuess(argv[1],1);
	EigenMatrix densitymatrix;
	if (guess.compare("core")==0)
		densitymatrix=CoreHamiltonian(natoms,atoms,basisset);
	else if (guess.compare("sad")==0)
		densitymatrix=SuperpositionAtomicDensity(ne,natoms,atoms,basisset);

	const EigenMatrix overlap=Overlap(natoms,atoms,basisset,1);
	const EigenMatrix kinetic=Kinetic(natoms,atoms,basisset,1);
	const EigenMatrix nuclear=Nuclear(natoms,atoms,basisset,1);
	const EigenMatrix hcore=kinetic+nuclear;
	const EigenMatrix repulsiondiag=RepulsionDiag(natoms,atoms,basisset,1);

	int nshellquartets;
	const long int n2integrals=nTwoElectronIntegrals(natoms,atoms,basisset,repulsiondiag,nshellquartets,1);
	double * repulsion=new double[n2integrals];
	short int * indices=new short int[n2integrals*5];
	Repulsion(natoms,atoms,basisset,nshellquartets,repulsiondiag,repulsion,indices,nprocs,1);

	EigenVector orbitalenergies(nbasis);
	EigenMatrix coefficients(nbasis,nbasis);

	double energy=RHF(ne,overlap,hcore,repulsion,indices,n2integrals,orbitalenergies,coefficients,densitymatrix,nprocs,1);
	std::cout<<"Total energy ... "<<nuclearrepulsion+energy<<" a.u."<<std::endl;

	delete [] repulsion;
	delete [] indices;

	std::cout<<"*** Chinium terminated normally ***"<<std::endl;
double overheadlength=6;
double griddensity=20;
std::cout<<UniformBoxGridDensity(natoms,atoms,basisset,densitymatrix,overheadlength,griddensity)<<std::endl;
	return 0;
}

