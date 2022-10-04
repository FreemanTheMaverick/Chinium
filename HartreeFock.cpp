#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include "LinearAlgebra.h"

void GMatrix(int n1integrals,double * repulsion,short int * indices,int n2integrals,double * densitymatrix,double * jmatrix,double * kmatrix,double * gmatrix){ // Memory seats for J matrix and K matrix should be provided.
	int nbasis=(int)(sqrt(8*n1integrals+1)-1)/2;
	double *repulsionranger=repulsion;
	short int *indicesranger=indices;
	double rawjmatrix[nbasis*nbasis];
	double rawkmatrix[nbasis*nbasis];
	double meaninglessmatrix[nbasis*nbasis];
	for (int i=0;i<nbasis;i++){
		for (int j=0;j<nbasis;j++){
			meaninglessmatrix[i*nbasis+j]=(i==j);
			rawjmatrix[i*nbasis+j]=0;
			rawkmatrix[i*nbasis+j]=0;
		}
	}
	for (int i=0;i<n2integrals;i++){
		const short int a=*(indicesranger++); // Moving the ranger pointer to the right, where the next index is located.
		const short int b=*(indicesranger++);
		const short int c=*(indicesranger++);
		const short int d=*(indicesranger++); // Moving the ranger pointer to the right, where the degeneracy factor is located
		const short int ab_deg=(a==b)?1:2;
		const short int cd_deg=(c==d)?1:2;
		const short int ab_cd_deg=(a==c)?(b==d?1:2):2;
		const short int abcd_deg=ab_deg*cd_deg*ab_cd_deg;
		const double value=*(repulsionranger++); // Moving the ranger pointer to the right, where the value of the next integral is located.
		const double deg_value=abcd_deg*value;
		rawjmatrix[a*nbasis+b]+=(c>d?densitymatrix[c*(c+1)/2+d]:densitymatrix[d*(d+1)/2+c])*deg_value;
		rawjmatrix[c*nbasis+d]+=(a>b?densitymatrix[a*(a+1)/2+b]:densitymatrix[b*(b+1)/2+a])*deg_value;
		rawkmatrix[a*nbasis+c]+=0.5*(b>d?densitymatrix[b*(b+1)/2+d]:densitymatrix[d*(d+1)/2+b])*deg_value;
		rawkmatrix[b*nbasis+d]+=0.5*(a>c?densitymatrix[a*(a+1)/2+c]:densitymatrix[c*(c+1)/2+a])*deg_value;
		rawkmatrix[a*nbasis+d]+=0.5*(b>c?densitymatrix[b*(b+1)/2+c]:densitymatrix[c*(c+1)/2+b])*deg_value;
		rawkmatrix[b*nbasis+c]+=0.5*(a>d?densitymatrix[a*(a+1)/2+d]:densitymatrix[d*(d+1)/2+a])*deg_value;
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
	while (abs(lastenergy-energy)>1.0e-8){
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
		MultiplyMatrix(1./5,C_occ,'f',0,C_occ,'f',1,4./5,densitymatrix,'l',0,nbasis,nocc,nbasis,nocc,nbasis,nbasis,nbasis,nbasis,'l',densitymatrix); // Mixing new density matrix with old one. This may improve robustness.
		MultiplyMatrix(1,hmatrix,'l',0,meaninglessmatrix,'f',0,1,fmatrix,'l',0,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,'l',intermediate);
		MultiplyMatrix(1,densitymatrix,'l',0,intermediate,'l',0,0,meaninglessmatrix,'f',0,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,'l',intermediate);
		energy=Trace(intermediate,nbasis,'l');
		clock_t iterend=clock();
		std::cout<<" Iteration "<<iiteration<<"  energy = "<<std::setprecision(12)<<energy<<"  elapsed time = "<<std::setprecision(3)<<double(iterend-iterstart)/CLOCKS_PER_SEC<<" s"<<std::endl;
		iiteration++;
	}
	std::cout<<"Done; Final RHF energy = "<<std::setprecision(12)<<energy<<std::endl;
	return energy;
}		

