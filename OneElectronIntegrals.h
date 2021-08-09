#pragma once
#include <libint2.hpp>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

using namespace libint2;
using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
Matrix Overlap(BasisSet obs,const int nbasis){
	Matrix overlapMatrix(nbasis,nbasis);
	Engine s_engine(Operator::overlap,obs.max_nprim(),obs.max_l());
	const auto& buf_vec=s_engine.results();
	auto shell2bf=obs.shell2bf();
	for (int s1=0;s1!=obs.size();s1++){
		int bf1=shell2bf[s1];
		int n1=obs[s1].size();
		for (int s2=s1;s2!=obs.size();s2++){
			s_engine.compute(obs[s1],obs[s2]);
			auto ints_shellset=buf_vec[0];
			if (ints_shellset==nullptr){
				continue;
			}
			int bf2=shell2bf[s2];
			int n2=obs[s2].size();
			for (int f1=0;f1!=n1;f1++){
				for (int f2=0;f2!=n2;f2++){
					double thisvalue=ints_shellset[f1*n2+f2];
					overlapMatrix(bf1+f1,bf2+f2)=thisvalue;
					overlapMatrix(bf2+f2,bf1+f1)=thisvalue;
				}
			}
		}
	}
	return overlapMatrix;
}

Matrix Kinetic(BasisSet obs,const int nbasis){
	Matrix kineticMatrix(nbasis,nbasis);
	Engine k_engine(Operator::kinetic,obs.max_nprim(),obs.max_l());
	const auto& buf_vec=k_engine.results();
	auto shell2bf=obs.shell2bf();
	for (int s1=0;s1!=obs.size();s1++){
		for (int s2=s1;s2!=obs.size();s2++){
			k_engine.compute(obs[s1],obs[s2]);
			auto ints_shellset=buf_vec[0];
			if (ints_shellset==nullptr){
				continue;
			}
			int bf1=shell2bf[s1];
			int n1=obs[s1].size();
			int bf2=shell2bf[s2];
			int n2=obs[s2].size();
			for (int f1=0;f1!=n1;f1++){
				for (int f2=0;f2!=n2;f2++){
					double thisvalue=ints_shellset[f1*n2+f2];
					kineticMatrix(bf1+f1,bf2+f2)=thisvalue;
					kineticMatrix(bf2+f2,bf1+f1)=thisvalue;
				}
			}
		}
	}
	return kineticMatrix;
}

Matrix Nuclear(BasisSet obs,const int nbasis,vector<Atom> atoms){
	Matrix nuclearMatrix(nbasis,nbasis);
	Engine v_engine(Operator::nuclear,obs.max_nprim(),obs.max_l());
	v_engine.set_params(make_point_charges(atoms));
	const auto& buf_vec=v_engine.results();
	auto shell2bf=obs.shell2bf();
	for (int s1=0;s1!=obs.size();s1++){
		for (int s2=s1;s2!=obs.size();s2++){
			v_engine.compute(obs[s1],obs[s2]);
			auto ints_shellset=buf_vec[0];
			if (ints_shellset==nullptr){
				continue;
			}
			int bf1=shell2bf[s1];
			int n1=obs[s1].size();
			int bf2=shell2bf[s2];
			int n2=obs[s2].size();
			for (int f1=0;f1!=n1;f1++){
				for (int f2=0;f2!=n2;f2++){
					double thisvalue=ints_shellset[f1*n2+f2];
					nuclearMatrix(bf1+f1,bf2+f2)=thisvalue;
					nuclearMatrix(bf2+f2,bf1+f1)=thisvalue;
				}
			}
		}
	}
	return nuclearMatrix;
}
