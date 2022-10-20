#include <string>
#include <iostream>
#include "Gateway.h"
#include "AtomicIntegrals.h"
#include "HartreeFock.h"
#include "InitialGuess.h"
//#include "LinearAlgebra.h"

int main(int argc,char *argv[]){

	std::cout<<"*** Chinium started ***"<<std::endl;
	const int nprocs=ReadNProcs(argv[1],1);
	double atoms[10000];
	const int natoms=ReadXYZ(argv[1],atoms,1);
	const std::string basisset_=ReadBasisSet(argv[1],1);
	const char * basisset=basisset_.data();
	const int ne=ReadNElectrons(argv[1],1);
	const int nbasis=nBasis(natoms,atoms,basisset,1);
	const int n1integrals=nOneElectronIntegrals(natoms,atoms,basisset,1);
	const double nuclearrepulsion=NuclearRepulsion(natoms,atoms,1);

	double densitymatrix[n1integrals];
	SuperpositionAtomicDensity(ne,natoms,atoms,basisset,densitymatrix,1);
//PrintMatrix_eigen(densitymatrix,nbasis,nbasis,'l');

	double * overlap=new double[n1integrals];
	Overlap(natoms,atoms,basisset,overlap,1);

	double * kinetic=new double[n1integrals];
	Kinetic(natoms,atoms,basisset,kinetic,1);

	double * nuclear=new double[n1integrals];
	Nuclear(natoms,atoms,basisset,nuclear,1);

	double * repulsiondiag=new double[n1integrals];
	RepulsionDiag(natoms,atoms,basisset,repulsiondiag,1);

	int nshellquartets;
	const int n2integrals=nTwoElectronIntegrals(natoms,atoms,basisset,repulsiondiag,nshellquartets,1);
	double * repulsion=new double[n2integrals];
	short int * indices=new short int[n2integrals*5];
	Repulsion(natoms,atoms,basisset,nshellquartets,repulsiondiag,repulsion,indices,nprocs,1);

	double orbitalenergies[nbasis];
	for (int i=0;i<nbasis;i++) orbitalenergies[i]=0;
	double coefficients[nbasis*nbasis];
	for (int i=0;i<nbasis*nbasis;i++) coefficients[i]=0;

	double energy=RHF(ne,overlap,kinetic,nuclear,n1integrals,repulsion,indices,n2integrals,orbitalenergies,coefficients,densitymatrix,nprocs,1);
//PrintMatrix_eigen(densitymatrix,nbasis,nbasis,'l');
	std::cout<<"Total energy ... "<<nuclearrepulsion+energy<<" a.u."<<std::endl;

	delete [] overlap;
	delete [] kinetic;
	delete [] nuclear;
	delete [] repulsiondiag;
	delete [] repulsion;
	delete [] indices;

	std::cout<<"*** Chinium terminated normally ***"<<std::endl;

	return 0;
}

