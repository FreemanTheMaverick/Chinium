#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cmath> // std::abs in "../Macro.h"
#include <experimental/array> // std::array, std::experimental::make_array
#include <utility> // std::pair, std::make_pair
#include <cassert>
#include <chrono>

#include "../Macro.h"

#include "Multiwfn.h"


#define __Loop_Over_Atom_Pairs__\
	for (int iatom=0;iatom<natoms;iatom++){\
		const double Zi=std::get<0>(libint2charges[iatom]);\
		const double xi=std::get<1>(libint2charges[iatom])[0];\
		const double yi=std::get<1>(libint2charges[iatom])[1];\
		const double zi=std::get<1>(libint2charges[iatom])[2];\
		for (int jatom=0;jatom<natoms;jatom++){\
			const double Zj=std::get<0>(libint2charges[jatom]);\
			const double xj=std::get<1>(libint2charges[jatom])[0];\
			const double yj=std::get<1>(libint2charges[jatom])[1];\
			const double zj=std::get<1>(libint2charges[jatom])[2];\
			const double Zij=Zi*Zj;\
			const double xij=xi-xj;\
			const double yij=yi-yj;\
			const double zij=zi-zj;\
			const double dij=sqrt(xij*xij+yij*yij+zij*zij);

#define __End_Loop_Over_Atom_Pairs__\
		}\
	}

double NuclearRepulsion0(std::vector<std::pair<double, std::array<double, 3>>> libint2charges){
	const int natoms = libint2charges.size();
	double nuclearrepulsion=0;
	__Loop_Over_Atom_Pairs__
		if (jatom<iatom)
			nuclearrepulsion=nuclearrepulsion+Zij/dij;
	__End_Loop_Over_Atom_Pairs__
	return nuclearrepulsion;
}

EigenMatrix NuclearRepulsion1(std::vector<std::pair<double, std::array<double, 3>>> libint2charges){
	const int natoms = libint2charges.size();
	EigenMatrix G=EigenZero(natoms,3);
	__Loop_Over_Atom_Pairs__
		if (jatom!=iatom){
			G(iatom,0)-=Zij/dij/dij/dij*xij;
			G(iatom,1)-=Zij/dij/dij/dij*yij;
			G(iatom,2)-=Zij/dij/dij/dij*zij;
		}
	__End_Loop_Over_Atom_Pairs__
	return G;
}

EigenMatrix NuclearRepulsion2(std::vector<std::pair<double, std::array<double, 3>>> libint2charges){
	const int natoms = libint2charges.size();
	EigenMatrix H=EigenZero(natoms*3,natoms*3);
	__Loop_Over_Atom_Pairs__
		if (jatom==iatom) for (int katom=0;katom<natoms;katom++){
			if (iatom==katom) continue;
			const double Zk=std::get<0>(libint2charges[katom]);
			const double xk=std::get<1>(libint2charges[katom])[0];
			const double yk=std::get<1>(libint2charges[katom])[1];
			const double zk=std::get<1>(libint2charges[katom])[2];
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
	return H;
}

void Multiwfn::NuclearRepulsion(std::vector<double> orders, int output){
	std::vector<std::pair<double, std::array<double, 3>>> libint2charges = {}; // Making point charges.
	for ( MwfnCenter& center : this->Centers )
		libint2charges.push_back(std::make_pair(
				center.Nuclear_charge,
				std::experimental::make_array(
					center.Coordinates[0],
					center.Coordinates[1],
					center.Coordinates[2]
				)
		));

	if (std::find(orders.begin(), orders.end(), 0) != orders.end()){
		if (output) std::printf("Calculating nuclear repulsion energy ... ");
		const auto start = __now__;
		this->E_tot += NuclearRepulsion0(libint2charges);
		if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
	}
	if (std::find(orders.begin(), orders.end(), 1) != orders.end()){
		const auto start = __now__;
		if (output) std::printf("Calculating nuclear repulsion gradient ... ");
		assert(this->Gradient.rows() == this->getNumCenters() && this->Gradient.cols() == 3 && "Nuclear gradient is not allocated!");
		this->Gradient += NuclearRepulsion1(libint2charges);
		if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
	}
	if (std::find(orders.begin(), orders.end(), 2) != orders.end()){
		const auto start = __now__;
		if (output) std::printf("Calculating nuclear repulsion hessian ... ");
		assert(this->Hessian.rows() == 3*this->getNumCenters() && this->Hessian.cols() == 3*this->getNumCenters() && "Nuclear hessian is not allocated!");
		this->Hessian += NuclearRepulsion2(libint2charges);
		if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
	}
}
