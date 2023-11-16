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
#include "GridGeneration.h"
#include "DensityFunctional.h"
#include "HFGradient.h"
#include "CoupledPerturbed.h"
#include "HFHessian.h"

int main(int argc,char *argv[]){

	std::cout<<"*** Chinium started ***"<<std::endl;
	assert("Running Chinium requires an input file!" && argc>1);

	// Gateway
	const int nprocs=ReadNProcs(argv[1],1);
	double atoms[10000];
	const int natoms=ReadXYZ(argv[1],atoms,1);
	const std::string basisset_=ReadBasisSet(argv[1],1);
	const char * basisset=basisset_.data();
	const int ne=ReadNElectrons(argv[1],1);
	const int nbasis=nBasis(natoms,atoms,basisset,1);
	nOneElectronIntegrals(natoms,atoms,basisset,1);
	const double nuclearrepulsion=NuclearRepulsion(natoms,atoms,1);
	const int derivative=ReadDerivative(argv[1],1);
	const std::string grid=ReadGrid(argv[1],1);
	const std::string method=ReadMethod(argv[1],1);
	const std::string guess=ReadGuess(argv[1],1);
	const double temperature=ReadTemperature(argv[1],1);
	const double chemicalpotential=ReadChemicalPotential(argv[1],1);
	const std::string scf="rijcosx";

	// Density functional (if any)
	int dfxid=0;
	int dfcid=0;
	double kscale=1;
	char approx;

	// Grid generation (if needed)
	int ngrids=0;
	double * xs=nullptr;
	double * ys=nullptr;
	double * zs=nullptr;
	double * ws=nullptr;
	double * aos=nullptr;
	double * ao1xs=nullptr;
	double * ao1ys=nullptr;
	double * ao1zs=nullptr;
	double * ao2ls=nullptr;
	double * ao2xxs=nullptr;
	double * ao2yys=nullptr;
	double * ao2zzs=nullptr;
	double * ao2xys=nullptr;
	double * ao2xzs=nullptr;
	double * ao2yzs=nullptr;
	if (!grid.empty()){
		ngrids=SphericalGridNumber(grid,natoms,atoms,1);
		xs=new double[ngrids];
		ys=new double[ngrids];
		zs=new double[ngrids];
		ws=new double[ngrids];
		SphericalGrid(grid,natoms,atoms,xs,ys,zs,ws,1);
	}

	// Grid values evaluation (if needed)
	if (method.compare("rhf")!=0){ // Three job types require grids. Allocating memory to necessary grid arrays.
		assert("DFT needs a grid to be set!" && ! grid.empty());
		ReadDF(method,dfxid,dfcid,kscale,approx,1);
		if (approx=='l' || approx=='g' || approx=='m') // For single point energy.
			aos=new double[nbasis*ngrids];
		if (approx=='g' || approx=='m'){
			ao1xs=new double[nbasis*ngrids];
			ao1ys=new double[nbasis*ngrids];
			ao1zs=new double[nbasis*ngrids];
		}
		if (approx=='m')
			ao2ls=new double[nbasis*ngrids];
		if (derivative>=1){ // For nuclear gradient.
			if (approx=='l' || approx=='g' || approx=='m'){
				if (!ao1xs) ao1xs=new double[nbasis*ngrids];
				if (!ao1ys) ao1ys=new double[nbasis*ngrids];
				if (!ao1zs) ao1zs=new double[nbasis*ngrids];
			}
			if (approx=='g' || approx=='m'){
				ao2xxs=new double[nbasis*ngrids];
				ao2yys=new double[nbasis*ngrids];
				ao2zzs=new double[nbasis*ngrids];
				ao2xys=new double[nbasis*ngrids];
				ao2xzs=new double[nbasis*ngrids];
				ao2yzs=new double[nbasis*ngrids];
			}
			if (approx=='m'){;}
		}
		if (derivative>=2){;} // For nuclear hessian.
	}else if (scf.compare("rijcosx")==0){
		assert("RIJCOSX needs a grid to be set!" && ! grid.empty());
		if (! aos) aos=new double[nbasis*ngrids];
	}else if (guess.compare("sap")==0){
		assert("SAP needs a grid to be set!" && ! grid.empty());
		if (! aos) aos=new double[nbasis*ngrids];
	}
	GetAoValues(natoms,atoms,basisset,xs,ys,zs,ngrids,
	            aos,ao1xs,ao1ys,ao1zs,ao2ls,
	            ao2xxs,ao2yys,ao2zzs,ao2xys,ao2xzs,ao2yzs);

	// Storable electron integrals 
	const EigenMatrix overlap=Overlap(natoms,atoms,basisset,1);
	const EigenMatrix kinetic=Kinetic(natoms,atoms,basisset,1);
	const EigenMatrix nuclear=Nuclear(natoms,atoms,basisset,1);
	const EigenMatrix hcore=kinetic+nuclear;
	const EigenMatrix repulsiondiag=RepulsionDiag(natoms,atoms,basisset,1);
	int nshellquartets;
	const long int n2integrals=nTwoElectronIntegrals(natoms,atoms,basisset,repulsiondiag,nshellquartets,1);
	double * repulsion=new double[n2integrals];
	short int * indices=new short int[n2integrals*5];
	Repulsion(natoms,atoms,basisset,nshellquartets,repulsiondiag,n2integrals,repulsion,indices,nprocs,1);
	EigenMatrix * ovlgrads=nullptr;
	EigenMatrix * kntgrads=nullptr;
	EigenMatrix * nclgrads=nullptr;
	EigenMatrix * hcoregrads=nullptr;
	if (derivative>=1){
		ovlgrads=new EigenMatrix[3*natoms];
		kntgrads=new EigenMatrix[3*natoms];
		nclgrads=new EigenMatrix[3*natoms];
		hcoregrads=new EigenMatrix[3*natoms];
		OvlGrads(natoms,atoms,basisset,ovlgrads,1);
		KntGrads(natoms,atoms,basisset,kntgrads,1);
		NclGrads(natoms,atoms,basisset,nclgrads,1);
	}

	// Initial guess
	EigenMatrix density,fock;
	if (guess.compare("core")==0)
		fock=hcore;
	else if (guess.compare("sad")==0)
		density=SuperpositionAtomicDensity(natoms,atoms,basisset);
	else if (guess.compare("sap")==0)
		fock=hcore+SuperpositionAtomicPotential(natoms,atoms,nbasis,
                                                        xs,ys,zs,ws,ngrids,aos);

	EigenMatrix coefficients(nbasis,nbasis);
	EigenVector orbitalenergies(nbasis);
	EigenVector occupancies(nbasis);occupancies.setZero();

	// KS preparation
	double energy=114514;
	if (method.compare("rhf")==0)
		energy=RHF(
			ne,temperature,chemicalpotential,
			overlap,hcore,
			repulsion,indices,n2integrals,
			orbitalenergies,coefficients,
			occupancies,density,fock,
			nprocs,1);
	else
		energy=RKS(
			ne,temperature,chemicalpotential,
			overlap,hcore,
			repulsion,indices,n2integrals,
			dfxid,dfcid,ngrids,ws,
			aos,
			ao1xs,ao1ys,ao1zs,
			ao2ls,
			orbitalenergies,coefficients,
			occupancies,density,fock,
			nprocs,1);

	std::cout<<"Total energy ... "<<nuclearrepulsion+energy<<" a.u."<<std::endl;

	EigenMatrix * fskeletons=nullptr;
	EigenMatrix W=EigenZero(nbasis,nbasis);
	for (int i=0;i<nbasis;i++)
		W+=occupancies[i]*coefficients.col(i)*coefficients.col(i).transpose()*orbitalenergies[i];

	if (derivative>=1){
		for (int i=0;i<natoms*3;i++)
			hcoregrads[i]=kntgrads[i]+nclgrads[i];
		__Delete_Matrices__(kntgrads,3*natoms);
		__Delete_Matrices__(nclgrads,3*natoms);
		if (derivative>=2){
			fskeletons=new EigenMatrix[3*natoms];
			for (int it=0;it<3*natoms;it++)
				fskeletons[it]=EigenZero(nbasis,nbasis);
		}
		const EigenMatrix nrg=NRG(natoms,atoms,1);
		EigenMatrix gele=EigenZero(natoms,3);
		if (method.compare("rhf")==0)
			gele=RHFG(
				natoms,atoms,basisset,
				ovlgrads,hcoregrads,fskeletons,
				coefficients,orbitalenergies,occupancies,
				1,1);
		else
			gele=RKSG(
				natoms,atoms,basisset,
				ovlgrads,hcoregrads,fskeletons,
				dfxid,dfcid,ngrids,ws,
				aos,
				ao1xs,ao1ys,ao1zs,
				ao2xxs,ao2yys,ao2zzs,
				ao2xys,ao2xzs,ao2yzs,
				coefficients,orbitalenergies,occupancies,
				1,1);
		const EigenMatrix g=nrg+gele;
		std::cout<<"Total gradient:"<<std::endl;
		__Z_2_Name__
		for (int iatom=0;iatom<natoms;iatom++){
			std::cout<<"| "<<Z2Name[(int)atoms[4*iatom]];
			if (Z2Name[(int)atoms[4*iatom]].length()==1)
				std::cout<<"   ";
			else if (Z2Name[(int)atoms[4*iatom]].length()==2)
				std::cout<<"  ";
			std::printf(" %17.10f  %17.10f  %17.10f\n",g(iatom,0),g(iatom,1),g(iatom,2));
		}
	}

	if (derivative>=2){
		EigenMatrix * dxn=new EigenMatrix[3*natoms];
		EigenMatrix * wxn=new EigenMatrix[3*natoms];
		EigenVector * exn=new EigenVector[3*natoms];
		NonIdempotentCPSCF(
			natoms,
			ovlgrads,fskeletons,
			repulsion,indices,n2integrals,kscale,
			dfxid,dfcid,ngrids,ws,
			aos,
			ao1xs,ao1ys,ao1zs,
			ao2ls,
			coefficients,orbitalenergies,occupancies,
			wxn,dxn,exn,
			nprocs,1);
		EigenMatrix hessian=NRH(natoms,atoms,1);
/*		hessian+=RKSH(
			natoms,atoms,basisset,
			density,dxn,
			W,wxn,
			ovlgrads,fskeletons,
			kscale,
			nprocs,1);
*/		if (temperature>0)
			hessian+=DensityOccupationGradientCPSCF(
				temperature,repulsion,indices,n2integrals,kscale,
				ovlgrads,fskeletons,dxn,exn,natoms,
				coefficients,occupancies,orbitalenergies,
				nprocs,1);

		__Delete_Matrices__(dxn,3*natoms);
		__Delete_Matrices__(wxn,3*natoms);
		__Delete_Vectors__(exn,3*natoms);
		//std::cout<<"Total hessian:"<<std::endl;
		//std::cout<<hessian<<std::endl;
	}

	delete [] xs;
	delete [] ys;
	delete [] zs;
	delete [] ws;
	delete [] aos;
	delete [] ao1xs;
	delete [] ao1ys;
	delete [] ao1zs;
	delete [] ao2ls;
	delete [] repulsion;
	delete [] indices;
	std::cout<<"*** Chinium terminated normally ***"<<std::endl;
	return 0;
}
