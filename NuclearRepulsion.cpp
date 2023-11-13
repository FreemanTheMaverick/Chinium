#include <Eigen/Dense>
#include <iostream>
#include "Aliases.h"

#define __Loop_Over_Atom_Pairs__\
	for (int iatom=0;iatom<natoms;iatom++){\
		const double Zi=atoms[iatom*4];\
		const double xi=atoms[iatom*4+1];\
		const double yi=atoms[iatom*4+2];\
		const double zi=atoms[iatom*4+3];\
		for (int jatom=0;jatom<natoms;jatom++){\
			const double Zj=atoms[jatom*4];\
			const double xj=atoms[jatom*4+1];\
			const double yj=atoms[jatom*4+2];\
			const double zj=atoms[jatom*4+3];\
			const double Zij=Zi*Zj;\
			const double xij=xi-xj;\
			const double yij=yi-yj;\
			const double zij=zi-zj;\
			const double dij=sqrt(xij*xij+yij*yij+zij*zij);

#define __End_Loop_Over_Atom_Pairs__\
		}\
	}

double NuclearRepulsion(const int natoms,double * atoms,const bool output){
	double nuclearrepulsion=0;
	__Loop_Over_Atom_Pairs__
		if (jatom<iatom)
			nuclearrepulsion=nuclearrepulsion+Zij/dij;
	__End_Loop_Over_Atom_Pairs__
	if (output) std::cout<<"Nuclear repulsion energy ... "<<nuclearrepulsion<<" a.u."<<std::endl;
	return nuclearrepulsion;
}

EigenMatrix NRG(const int natoms,double * atoms,const bool output){
	time_t start=time(0);
	if (output) std::cout<<"Calculating nuclear repulsion gradient ... ";
	EigenMatrix G=EigenZero(natoms,3);
	__Loop_Over_Atom_Pairs__
		if (jatom!=iatom){
			G(iatom,0)-=Zij/dij/dij/dij*xij;
			G(iatom,1)-=Zij/dij/dij/dij*yij;
			G(iatom,2)-=Zij/dij/dij/dij*zij;
		}
	__End_Loop_Over_Atom_Pairs__
	time_t end=time(0);
	if (output) std::cout<<"done "<<end-start<<" s"<<std::endl;
	return G;
}

EigenMatrix NRH(const int natoms,double * atoms,const bool output){
	time_t start=time(0);
	if (output) std::cout<<"Calculating nuclear repulsion hessian ... ";
	EigenMatrix H=EigenZero(natoms*3,natoms*3);
	__Loop_Over_Atom_Pairs__
		if (jatom==iatom) for (int katom=0;katom<natoms;katom++){
			if (iatom==katom) continue;
			const double Zk=atoms[katom*4];
			const double xk=atoms[katom*4+1];
			const double yk=atoms[katom*4+2];
			const double zk=atoms[katom*4+3];
			const double Zik=Zi*Zk;
			const double xik=xi-xk;
			const double yik=yi-yk;
			const double zik=zi-zk;
			const double dik=sqrt(xik*xik+yik*yik+zik*zik);
			H(3*iatom+0,3*iatom+0)+=-Zik/dik/dik/dik+3*xik*xik*Zik/dik/dik/dik/dik/dik;
			H(3*iatom+1,3*iatom+1)+=-Zik/dik/dik/dik+3*yik*yik*Zik/dik/dik/dik/dik/dik;
			H(3*iatom+2,3*iatom+2)+=-Zik/dik/dik/dik+3*zik*zik*Zik/dik/dik/dik/dik/dik;
			H(3*iatom+0,3*iatom+1)+=3*xik*yik*Zik/dik/dik/dik/dik/dik;
			H(3*iatom+0,3*iatom+2)+=3*xik*zik*Zik/dik/dik/dik/dik/dik;
			H(3*iatom+1,3*iatom+2)+=3*yik*zik*Zik/dik/dik/dik/dik/dik;
			H(3*iatom+1,3*iatom+0)=H(3*iatom+0,3*iatom+1);
			H(3*iatom+2,3*iatom+0)=H(3*iatom+0,3*iatom+2);
			H(3*iatom+2,3*iatom+1)=H(3*iatom+1,3*iatom+2);
		}else{
			H(3*iatom+0,3*jatom+0)=H(3*jatom+0,3*iatom+0)=Zij/dij/dij/dij-3*xij*xij*Zij/dij/dij/dij/dij/dij;
			H(3*iatom+1,3*jatom+1)=H(3*jatom+1,3*iatom+1)=Zij/dij/dij/dij-3*yij*yij*Zij/dij/dij/dij/dij/dij;
			H(3*iatom+2,3*jatom+2)=H(3*jatom+2,3*iatom+2)=Zij/dij/dij/dij-3*zij*zij*Zij/dij/dij/dij/dij/dij;
			H(3*iatom+0,3*jatom+1)=H(3*jatom+1,3*iatom+0)=-3*xij*yij*Zij/dij/dij/dij/dij/dij;
			H(3*iatom+0,3*jatom+2)=H(3*jatom+2,3*iatom+0)=-3*xij*zij*Zij/dij/dij/dij/dij/dij;
			H(3*iatom+1,3*jatom+2)=H(3*jatom+2,3*iatom+1)=-3*yij*zij*Zij/dij/dij/dij/dij/dij;
		}
	__End_Loop_Over_Atom_Pairs__
	time_t end=time(0);
	if (output) std::cout<<"done "<<end-start<<" s"<<std::endl;
	return H;
}
