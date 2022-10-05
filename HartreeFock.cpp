#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include "LinearAlgebra.h"

#define __convergence_threshold__ 1.e-8
#define __damping_factor__ 0.2

void GMatrix(int n1integrals,double * repulsion,short int * indices,int n2integrals,double * densitymatrix,double * jmatrix,double * kmatrix,double * gmatrix){
	int nbasis=(int)(sqrt(8*n1integrals+1)-1)/2;
	double *repulsionranger=repulsion;
	short int *indicesranger=indices;
	double fulldensitymatrix[nbasis*nbasis]; // The density matrix stored as full must be provided, or the G matrix formation loop will contain a lot of logic judgment, which will reduce GPU performance.
	double rawjmatrix[nbasis*nbasis];
	double rawkmatrix[nbasis*nbasis];
	double meaninglessmatrix[nbasis*nbasis];
	for (int i=0;i<nbasis;i++){
		for (int j=0;j<nbasis;j++){
			fulldensitymatrix[i*nbasis+j]=i>j?densitymatrix[i*(i+1)/2+j]:densitymatrix[j*(j+1)/2+i];
			rawjmatrix[i*nbasis+j]=0;
			rawkmatrix[i*nbasis+j]=0;
			meaninglessmatrix[i*nbasis+j]=(i==j);
		}
	}
	for (int i=0;i<n2integrals;i++){
		const short int a=*(indicesranger++); // Moving the ranger pointer to the right, where the next index is located.
		const short int b=*(indicesranger++);
		const short int c=*(indicesranger++);
		const short int d=*(indicesranger++); // Moving the ranger pointer to the right, where the degeneracy factor is located
		const short int deg=*(indicesranger++);
		const double value=*(repulsionranger++); // Moving the ranger pointer to the right, where the next integral is located.
		const double deg_value=deg*value;
		rawjmatrix[a*nbasis+b]+=fulldensitymatrix[c*nbasis+d]*deg_value;
		rawjmatrix[c*nbasis+d]+=fulldensitymatrix[a*nbasis+b]*deg_value;
		rawkmatrix[a*nbasis+c]+=0.5*fulldensitymatrix[b*nbasis+d]*deg_value;
		rawkmatrix[b*nbasis+d]+=0.5*fulldensitymatrix[a*nbasis+c]*deg_value;
		rawkmatrix[a*nbasis+d]+=0.5*fulldensitymatrix[b*nbasis+c]*deg_value;
		rawkmatrix[b*nbasis+c]+=0.5*fulldensitymatrix[a*nbasis+d]*deg_value;
	}
	MultiplyMatrix(0.5,rawjmatrix,'f',0,meaninglessmatrix,'f',0,0.5,rawjmatrix,'f',1,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,'l',jmatrix);
	MultiplyMatrix(0.5,rawkmatrix,'f',0,meaninglessmatrix,'f',0,0.5,rawkmatrix,'f',1,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,'l',kmatrix);
	MultiplyMatrix(1,jmatrix,'l',0,meaninglessmatrix,'f',0,-0.5,kmatrix,'l',0,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,'l',gmatrix);
}

double RHF(int nele,double * overlap,double * kinetic,double * nuclear,int n1integrals,double * repulsion,short int * indices,long int n2integrals,double * orbitalenergies,double * coefficients,double * densitymatrix){
	std::cout<<"Restricted Hartree-Fock ..."<<std::endl;
	int nocc=nele/2;
	int nbasis=(int)(sqrt(8*n1integrals+1)-1)/2;
	double energy=114514;
	double lastenergy=1919810;
	double hmatrix[n1integrals];
	double meaninglessmatrix[nbasis*nbasis];
	for (int i=0;i<nbasis;i++)
		for (int j=0;j<nbasis;j++)
			meaninglessmatrix[i*nbasis+j]=(i==j);
	MultiplyMatrix(1,kinetic,'l',0,meaninglessmatrix,'f',0,1,nuclear,'l',0,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,'l',hmatrix);
	double jmatrix[n1integrals];
	double kmatrix[n1integrals];
	double gmatrix[n1integrals];
	double fmatrix[n1integrals];
	double C_occ[nbasis*nocc];
	double intermediate[n1integrals];
	int iiteration=0;
	while (abs(lastenergy-energy)>__convergence_threshold__){ // Normal HF SCF procedure.
		clock_t iterstart=clock();
		lastenergy=energy;
		for (int i=0;i<n1integrals;i++){
			jmatrix[i]=0;
			kmatrix[i]=0;
		}
		GMatrix(n1integrals,repulsion,indices,n2integrals,densitymatrix,jmatrix,kmatrix,gmatrix);
		MultiplyMatrix(1,hmatrix,'l',0,meaninglessmatrix,'f',0,1,gmatrix,'l',0,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,'l',fmatrix);
		GeneralizedSelfAdjointEigenSolver(fmatrix,overlap,nbasis,'l',orbitalenergies,coefficients);
		SliceMatrix(coefficients,nbasis,nbasis,0,nbasis-1,0,nocc-1,C_occ);
		MultiplyMatrix(1-__damping_factor__,C_occ,'f',0,C_occ,'f',1,__damping_factor__,densitymatrix,'l',0,nbasis,nocc,nbasis,nocc,nbasis,nbasis,nbasis,nbasis,'l',densitymatrix); // Mixing new density matrix with old one. This may improve robustness.
		MultiplyMatrix(1,hmatrix,'l',0,meaninglessmatrix,'f',0,1,fmatrix,'l',0,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,'l',intermediate);
		MultiplyMatrix(1,densitymatrix,'l',0,intermediate,'l',0,0,meaninglessmatrix,'f',0,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,'l',intermediate);
		energy=Trace(intermediate,nbasis,'l');
		clock_t iterend=clock();
		std::cout<<" Iteration "<<iiteration<<"  energy = "<<std::setprecision(12)<<energy<<" a.u."<<"  elapsed time = "<<std::setprecision(3)<<double(iterend-iterstart)/CLOCKS_PER_SEC<<" s"<<std::endl;
		iiteration++;
	}
	std::cout<<"Done; Final RHF energy = "<<std::setprecision(12)<<energy<<" a.u."<<std::endl;
	return energy;
}		

