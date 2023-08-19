#include <Eigen/Dense>
#include <string>
#include <cassert>
#include <iostream>
#include "Aliases.h"
#include "Gateway.h"
#include "AtomicIntegrals.h"
#include "HartreeFock.h"
#include "InitialGuess.h"
#include "GridIntegrals.h"
#include "DensityFunctional.h"

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
	EigenMatrix density;
	if (guess.compare("core")==0)
		density=CoreHamiltonian(natoms,atoms,basisset);
	else if (guess.compare("sad")==0)
		density=SuperpositionAtomicDensity(ne,natoms,atoms,basisset);

	std::string grid=ReadGrid(argv[1],1);
	long int ngrids=0;
	double * xs=nullptr;
	double * ys=nullptr;
	double * zs=nullptr;
	double * ws=nullptr;
	double * gridaos=nullptr;
	double * gridao1derivs=nullptr;
	if (! grid.empty()){
		ngrids=SphericalGridNumber(grid,natoms,atoms,1);
		xs=new double[ngrids];
		ys=new double[ngrids];
		zs=new double[ngrids];
		ws=new double[ngrids];
		SphericalGrid(grid,natoms,atoms,xs,ys,zs,ws,1);
	}

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

	double energy=114514;
	const std::string method=ReadMethod(argv[1],1);
	if (method.compare("rhf")==0)
		energy=RHF(ne,overlap,hcore,
		           repulsion,indices,n2integrals,
		           orbitalenergies,coefficients,density,
		           nprocs,1);
	else{
		assert((void("DFT needs a grid to be set!"),ngrids>0));
		int dfxid,dfcid;
		double kscale;
		char approx;
		ReadDF(method,dfxid,dfcid,kscale,approx,1);
		if (! gridaos){
			gridaos=new double[nbasis*ngrids];
			GetAoValues(natoms,atoms,basisset,xs,ys,zs,ngrids,gridaos);
		}
		if (! gridao1derivs && approx=='g'){
			gridao1derivs=new double[nbasis*ngrids*3];
		}
		energy=RKS(ne,overlap,hcore,
		           repulsion,indices,n2integrals,
		           dfxid,dfcid,ngrids,gridaos,gridao1derivs,ws,
		           orbitalenergies,coefficients,density,
		           nprocs,1);
	}
double overhead=3;
double spacing=0.04;
ngrids=UniformBoxGridNumber(natoms,atoms,basisset,overhead,spacing);
xs=new double[ngrids];
ys=new double[ngrids];
zs=new double[ngrids];
ws=new double[ngrids];
UniformBoxGrid(natoms,atoms,basisset,overhead,spacing,xs,ys,zs);
gridaos=new double[nbasis*ngrids];
for (long int igrid=0;igrid<ngrids;igrid++)
	ws[igrid]=spacing*spacing*spacing;
double * ds=new double[ngrids];
GetAoValues(natoms,atoms,basisset,xs,ys,zs,ngrids,gridaos);
GetDensity(gridaos,ngrids,2*density,ds);
std::cout<<SumUp(ds,ws,ngrids);
	std::cout<<"Total energy ... "<<nuclearrepulsion+energy<<" a.u."<<std::endl;

	delete [] xs;
	delete [] ys;
	delete [] zs;
	delete [] ws;
	delete [] gridaos;
	delete [] gridao1derivs;
	delete [] repulsion;
	delete [] indices;

	std::cout<<"*** Chinium terminated normally ***"<<std::endl;
	return 0;
}
