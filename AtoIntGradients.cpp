#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>
#include <ctime>
#include <iostream>
#include <omp.h>
#include "Aliases.h"
#include "Libint2.h"

#define __Loop_Over_XYZ__(iatom,position){\
	for (short int f1=0,f12=0;f1!=n1;f1++){\
		const short int bf1=bf1_first+f1;\
		for (short int f2=0;f2!=n2;f2++,f12++,iint++){\
			const short int bf2=bf2_first+f2;\
			if (bf2<=bf1){\
				gs[3*(iatom)+0](bf1,bf2)+=buf_vec[3*(position)+0][f12];\
				gs[3*(iatom)+1](bf1,bf2)+=buf_vec[3*(position)+1][f12];\
				gs[3*(iatom)+2](bf1,bf2)+=buf_vec[3*(position)+2][f12];\
			}\
		}\
	}\
}

void OneEleDeris(const int natoms,double * atoms,const char * basisset,char type,EigenMatrix * gs,const bool output){
	libint2::Operator operator_=libint2::Operator::overlap;
	if (type=='s'){
		if (output) std::cout<<"Calculating overlap integral derivatives w.r.t nuclear coordinates ... ";
		operator_=libint2::Operator::overlap;
	}else if (type=='k'){
		if (output) std::cout<<"Calculating kinetic integral derivatives w.r.t nuclear coordinates ... ";
		operator_=libint2::Operator::kinetic;
	}else if (type=='n'){
		if (output) std::cout<<"Calculating nuclear integral derivatives w.r.t nuclear coordinates ... ";
		operator_=libint2::Operator::nuclear;
	}
	time_t start=time(0);
	__Basis_From_Atoms__
	int nbasis=nBasis_from_obs(obs);
	for (int i=0;i<natoms*3;i++)
		gs[i]=EigenZero(nbasis,nbasis);
	libint2::initialize();
	libint2::Engine engine(operator_,obs.max_nprim(),obs.max_l(),1);
	if (type=='n')
		engine.set_params(libint2::make_point_charges(libint2atoms));
	const auto & buf_vec=engine.results();
	auto shell2bf=obs.shell2bf();
	auto shell2atom=obs.shell2atom(libint2atoms);
	for (short int s1=0;s1!=(short int)obs.size();s1++){
		const short int atom1=shell2atom[s1];
		const short int bf1_first=shell2bf[s1];
		const short int n1=obs[s1].size();
		for (short int s2=0;s2<=s1;s2++){
			const short int atom2=shell2atom[s2];
			if (operator_!=libint2::Operator::nuclear and atom1==atom2) continue;
			const short int bf2_first=shell2bf[s2];
			const short int n2=obs[s2].size();
			engine.compute(obs[s1],obs[s2]);
int iint=0;
			__Loop_Over_XYZ__(atom1,0)
			__Loop_Over_XYZ__(atom2,1)
			if (operator_==libint2::Operator::nuclear)
				for (short int iatom=0;iatom<natoms;iatom++)
					__Loop_Over_XYZ__(iatom,iatom+2)
		}
	}
	for (int i=0;i<natoms*3;i++){
		const EigenMatrix transpose=gs[i].transpose();
		const EigenMatrix diagonal(gs[i].diagonal().asDiagonal());
		gs[i]+=transpose-diagonal;
	}
	libint2::finalize();
	time_t end=time(0);
	if (output) std::cout<<"done "<<end-start<<" s"<<std::endl;
}

void OvlGrads(const int natoms,double * atoms,const char * basisset,EigenMatrix * gs,const bool output){
	OneEleDeris(natoms,atoms,basisset,'s',gs,output);
}

void KntGrads(const int natoms,double * atoms,const char * basisset,EigenMatrix * gs,const bool output){
	OneEleDeris(natoms,atoms,basisset,'k',gs,output);
}

void NclGrads(const int natoms,double * atoms,const char * basisset,EigenMatrix * gs,const bool output){
	OneEleDeris(natoms,atoms,basisset,'n',gs,output);
}
