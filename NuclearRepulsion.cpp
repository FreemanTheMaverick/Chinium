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
			nuclearrepulsion=nuclearrepulsion+Zi*Zj/dij;
	__End_Loop_Over_Atom_Pairs__
	if (output) std::cout<<"Nuclear repulsion energy ... "<<nuclearrepulsion<<" a.u."<<std::endl;
	return nuclearrepulsion;
}

EigenMatrix NRG(const int natoms,double * atoms,const bool output){
	time_t start=time(0);
	if (output) std::cout<<"Calculating nuclear repulsion gradients ... ";
	EigenMatrix G=EigenZero(natoms,3);
	__Loop_Over_Atom_Pairs__
		if (jatom!=iatom){
			G(iatom,0)-=Zi*Zj/dij/dij/dij*xij;
			G(iatom,1)-=Zi*Zj/dij/dij/dij*yij;
			G(iatom,2)-=Zi*Zj/dij/dij/dij*zij;
		}
	__End_Loop_Over_Atom_Pairs__
	time_t end=time(0);
	if (output) std::cout<<"done "<<end-start<<" s"<<std::endl;
	return G;
}
