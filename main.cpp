#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <cassert>
#include <iostream>
#include "Aliases.h"
#include "Gateway.h"
#include "NuclearRepulsion.h"
#include "AtomicIntegrals.h"
#include "AtoIntGradients.h"
#include "HartreeFock.h"
#include "InitialGuess.h"
#include "GridIntegrals.h"
#include "DensityFunctional.h"
#include "HFGradient.h"

int main(int argc,char *argv[]){

	std::cout<<"*** Chinium started ***"<<std::endl;
	assert(("Running Chinium requires an input file!" && argc>1));
	const int nprocs=ReadNProcs(argv[1],1);
	double atoms[10000];
	const int natoms=ReadXYZ(argv[1],atoms,1);
	const std::string basisset_=ReadBasisSet(argv[1],1);
	const char * basisset=basisset_.data();
	const int ne=ReadNElectrons(argv[1],1);
	const int nbasis=nBasis(natoms,atoms,basisset,1);
	nOneElectronIntegrals(natoms,atoms,basisset,1);
	const double nuclearrepulsion=NuclearRepulsion(natoms,atoms,1);
	int derivative=ReadDerivative(argv[1],1);

	int ngrids=0;
	double * xs=nullptr;
	double * ys=nullptr;
	double * zs=nullptr;
	double * ws=nullptr;
	std::string grid=ReadGrid(argv[1],1);
	if (!grid.empty()){
		ngrids=SphericalGridNumber(grid,natoms,atoms,1);
		xs=new double[ngrids];
		ys=new double[ngrids];
		zs=new double[ngrids];
		ws=new double[ngrids];
		SphericalGrid(grid,natoms,atoms,xs,ys,zs,ws,1);
	}

	const std::string method=ReadMethod(argv[1],1);
	int dfxid=0;
	int dfcid=0;
	double kscale;
	char approx;
	const std::string scf="rijcosx";
	const std::string guess=ReadGuess(argv[1],1);

	double * gridaos=nullptr;
	double * gridao1xs=nullptr;
	double * gridao1ys=nullptr;
	double * gridao1zs=nullptr;
	double * gridao2s=nullptr;
	if (method.compare("rhf")!=0 || scf.compare("rijcosx")==0 || guess.compare("sap")==0){ // Three job types require grids.
		if (method.compare("rhf")!=0){
			assert("DFT needs a grid to be set!" && ! grid.empty());
			ReadDF(method,dfxid,dfcid,kscale,approx,1);
			if (approx=='l' || approx=='g' || approx=='m') 
				gridaos=new double[nbasis*ngrids];
			if (approx=='g' || approx=='m'){
				gridao1xs=new double[nbasis*ngrids];
				gridao1ys=new double[nbasis*ngrids];
				gridao1zs=new double[nbasis*ngrids];
			}
			if (approx=='m')
				gridao2s=new double[nbasis*ngrids];
		}else if (scf.compare("rijcosx")==0){
			assert(("RIJCOSX needs a grid to be set!" && ! grid.empty()));
			if (gridaos==nullptr) gridaos=new double[nbasis*ngrids];
		}else if (guess.compare("sap")==0){
			assert(("SAP needs a grid to be set!" && ! grid.empty()));
			if (gridaos==nullptr) gridaos=new double[nbasis*ngrids];
		}

		GetAoValues(natoms,atoms,basisset,xs,ys,zs,ngrids,
		            gridaos,gridao1xs,gridao1ys,gridao1zs,gridao2s);
		for (long int i=0;i<nbasis*ngrids;i++){
			if (approx=='l' || approx=='g' || approx=='m' || scf.compare("rijcosx")==0) 
				if (isnan(gridaos[i])) gridaos[i]=0;
			if (approx=='g' || approx=='m'){
				if (isnan(gridao1xs[i])) gridao1xs[i]=0;
				if (isnan(gridao1ys[i])) gridao1ys[i]=0;
				if (isnan(gridao1zs[i])) gridao1zs[i]=0;
			}
			if (approx=='m')
				if (isnan(gridao2s[i])) gridao2s[i]=0;
		}
	}

	const EigenMatrix overlap=Overlap(natoms,atoms,basisset,1);
	const EigenMatrix kinetic=Kinetic(natoms,atoms,basisset,1);
	const EigenMatrix nuclear=Nuclear(natoms,atoms,basisset,1);
	const EigenMatrix hcore=kinetic+nuclear;
	const EigenMatrix repulsiondiag=RepulsionDiag(natoms,atoms,basisset,1);

	EigenMatrix density,fock;
	if (guess.compare("core")==0)
		density=CoreHamiltonian(natoms,atoms,basisset);
	else if (guess.compare("sad")==0)
		density=SuperpositionAtomicDensity(natoms,atoms,basisset);
	else if (guess.compare("sap")==0)
		fock=hcore+SuperpositionAtomicPotential(natoms,atoms,nbasis,
                                                        xs,ys,zs,ws,ngrids,gridaos);

	int nshellquartets;
	const long int n2integrals=nTwoElectronIntegrals(natoms,atoms,basisset,repulsiondiag,nshellquartets,1);
	double * repulsion=new double[n2integrals];
	short int * indices=new short int[n2integrals*5];
	Repulsion(natoms,atoms,basisset,nshellquartets,repulsiondiag,repulsion,indices,nprocs,1);

	EigenMatrix coefficients(nbasis,nbasis);
	EigenVector orbitalenergies(nbasis);
	EigenVector occs(nbasis);occs.setZero();

	double energy=114514;
	if (method.compare("rhf")==0)
		energy=RHF(ne,overlap,hcore,
		           repulsion,indices,n2integrals,
		           orbitalenergies,coefficients,
		           density,fock,
		           nprocs,1);
	else
		energy=RKS(ne,overlap,hcore,
		           repulsion,indices,n2integrals,
		           dfxid,dfcid,ngrids,ws,
		           gridaos,
		           gridao1xs,gridao1ys,gridao1zs,
		           gridao2s,
		           orbitalenergies,coefficients,
		           density,fock,
		           nprocs,1);

	std::cout<<"Total energy ... "<<nuclearrepulsion+energy<<" a.u."<<std::endl;

	if (derivative>0){
		EigenMatrix * ovlgrads=new EigenMatrix[3*natoms];
		EigenMatrix * kntgrads=new EigenMatrix[3*natoms];
		EigenMatrix * nclgrads=new EigenMatrix[3*natoms];
		EigenMatrix * hcoregrads=new EigenMatrix[3*natoms];
		OvlGrads(natoms,atoms,basisset,ovlgrads,1);
		KntGrads(natoms,atoms,basisset,kntgrads,1);
		NclGrads(natoms,atoms,basisset,nclgrads,1);
		for (int iatom=0;iatom<natoms*3;iatom++)
			hcoregrads[iatom]=kntgrads[iatom]+nclgrads[iatom];
		for (int i=0;i<ne/2;i++)
			occs[i]=1;
		const EigenMatrix nrg=NRG(natoms,atoms,1);
		const EigenMatrix rhfg=RHFG(natoms,atoms,basisset,
		                           ovlgrads,hcoregrads,
		                           coefficients,orbitalenergies,occs,
		                           1,1);
		const EigenMatrix g=nrg+rhfg;
		std::cout<<"Total gradient:"<<std::endl;
		__Z_2_Name__
		for (int iatom=0;iatom<natoms;iatom++)
			std::cout<<"| "<<Z2Name[(int)atoms[4*iatom]]<<" "<<g.row(iatom)<<std::endl;
	}

	delete [] xs;
	delete [] ys;
	delete [] zs;
	delete [] ws;
	delete [] gridaos;
	delete [] gridao1xs;
	delete [] gridao1ys;
	delete [] gridao1zs;
	delete [] gridao2s;
	delete [] repulsion;
	delete [] indices;
	std::cout<<"*** Chinium terminated normally ***"<<std::endl;
	return 0;
}
