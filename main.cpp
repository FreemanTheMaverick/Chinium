#include <string>
#include <iostream>
#include "Gateway.h"
#include "AtomicIntegrals.h"
#include "HartreeFock.h"

int main(int argc,char *argv[]){

	std::cout<<"*** Chinium started ***"<<std::endl;
	double * atoms=new double[10000];
	const int natoms=ReadXYZ(argv[1],atoms,1);
	const std::string basisset_=ReadBasisSet(argv[1]);
	const char * basisset=basisset_.data();
	const int ne=ReadNElectrons(argv[1]);
	const int nbasis=nBasis(natoms,atoms,basisset);
	const int n1integrals=nOneElectronIntegrals(natoms,atoms,basisset);
	const double nuclearrepulsion=NuclearRepulsion(natoms,atoms);

	double * overlap=new double[n1integrals];
	Overlap(natoms,atoms,basisset,overlap);

	double * kinetic=new double[n1integrals];
	Kinetic(natoms,atoms,basisset,kinetic);

	double * nuclear=new double[n1integrals];
	Nuclear(natoms,atoms,basisset,nuclear);

	double * repulsiondiag=new double[n1integrals];
	RepulsionDiag(natoms,atoms,basisset,repulsiondiag);

	int nshellquartets;
	const int n2integrals=nTwoElectronIntegrals(natoms,atoms,basisset,repulsiondiag,nshellquartets);
	double * repulsion=new double[n2integrals];
	short int * indices=new short int[n2integrals*5];
	Repulsion(natoms,atoms,basisset,nshellquartets,repulsiondiag,repulsion,indices);

	double * orbitalenergies=new double[nbasis];
	for (int i=0;i<nbasis;i++) orbitalenergies[i]=0;
	double * coefficients=new double[nbasis*nbasis];
	for (int i=0;i<nbasis*nbasis;i++) coefficients[i]=0;
	double * densitymatrix=new double[n1integrals];
	for (int i=0;i<n1integrals;i++) densitymatrix[i]=0;
	double energy=RHF(ne,overlap,kinetic,nuclear,n1integrals,repulsion,indices,n2integrals,orbitalenergies,coefficients,densitymatrix);
	std::cout<<"Total energy ... "<<nuclearrepulsion+energy<<" a.u."<<std::endl;

	return 0;
}

