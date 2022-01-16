#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <iostream>
#include "AtomicIntegrals.h"
#include "HartreeFock.h"
#include <string>
#include <vector>
#include <libint2.hpp>

int main(int argc,char *argv[]){
	libint2::initialize();
	std::string xyz="f.xyz";
	std::ifstream input(xyz);
	std::vector<libint2::Atom> atoms=libint2::read_dotxyz(input);
	libint2::BasisSet obs("cc-pvdz",atoms);
	EigenMatrix overlap=Overlap(obs);
	EigenMatrix kinetic=Kinetic(obs);
	EigenMatrix nuclear=Nuclear(obs,atoms);
	double *repulsion;
	short int *indices;
	int nintegrals;
	Repulsion(obs,&repulsion,&indices,nintegrals);
	libint2::finalize();
	double hfenergy;
	EigenMatrix coefficients;
	EigenMatrix guesscoefficients=Eigen::MatrixXd::Zero(kinetic.rows(),kinetic.rows());
	RHF(6*18,overlap,kinetic+nuclear,repulsion,indices,nintegrals,guesscoefficients,coefficients,hfenergy);
	return 0;
}
	


