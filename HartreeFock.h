#include <libint2.hpp>
#include <Eigen/Dense>
#include "OneElectronIntegrals.h"
#include "TwoElectronIntegrals.h"
#include <iostream>
#include <vector>
#include <string>
#include "OrbitalDetail.h"
#include <cmath>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
class HartreeFockJob{
	public:
		vector<Atom> Atoms;
		string BasisName;
		string Guess;
		Matrix CoefficientMatrix;
		Matrix DensityMatrix;
		Matrix JMatrix;
		Matrix KMatrix;
		Matrix OrbitalEnergies;
		double Energy;
		void setXYZ(std::string xyzfilename);
		void setBasisSet(std::string basisname);
		void setGuess(std::string guess);
		void Compute();
};

void HartreeFockJob::setXYZ(std::string xyzfilename){
	ifstream input_file(xyzfilename);
	Atoms=read_dotxyz(input_file);
}

void HartreeFockJob::setBasisSet(std::string basisname){
	BasisName=basisname;
}

void HartreeFockJob::setGuess(std::string guess){
	Guess=guess;
}

Matrix compute_soad(const std::vector<Atom>& atoms) {

  // compute number of atomic orbitals
  size_t nao = 0;
  for(const auto& atom: atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2) // H, He
      nao += 1;
    else if (Z <= 10) // Li - Ne
      nao += 5;
    else
      throw "SOAD with Z > 10 is not yet supported";
  }

  // compute the minimal basis density
  Matrix D = Matrix::Zero(nao, nao);
  size_t ao_offset = 0; // first AO of this atom
  for(const auto& atom: atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2) { // H, He
      D(ao_offset, ao_offset) = Z; // all electrons go to the 1s
      ao_offset += 1;
    }
    else if (Z <= 10) {
      D(ao_offset, ao_offset) = 2; // 2 electrons go to the 1s
      D(ao_offset+1, ao_offset+1) = (Z == 3) ? 1 : 2; // Li? only 1 electron in 2s, else 2 electrons
      // smear the remaining electrons in 2p orbitals
      const double num_electrons_per_2p = (Z > 4) ? (double)(Z - 4)/3 : 0;
      for(auto xyz=0; xyz!=3; ++xyz)
        D(ao_offset+2+xyz, ao_offset+2+xyz) = num_electrons_per_2p;
      ao_offset += 5;
    }
  }

  return D; // we use densities normalized to # of electrons/2
}


void HartreeFockJob::Compute(){
	BasisSet obs(BasisName,Atoms);
	int nbasis=nBasis(obs);
	Matrix overlap=Overlap(obs,nbasis);
	Matrix kinetic=Kinetic(obs,nbasis);
	Matrix nuclear=Nuclear(obs,nbasis,Atoms);
	Matrix Hcore=kinetic+nuclear;
	kinetic.resize(0,0);
	nuclear.resize(0,0);
	Matrix densitymatrix;
	int nocc=nOcc(Atoms);
	if (Guess=="hcore"){
		Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(Hcore,overlap);
		auto eps=gen_eig_solver.eigenvalues();
		auto C=gen_eig_solver.eigenvectors();
		auto C_occ=C.leftCols(nocc);
		densitymatrix=C_occ*C_occ.transpose();
	}else{
		densitymatrix=compute_soad(Atoms);
	}
	double lastenergy;
	double energy=0;
	do{
		lastenergy=energy;
		energy=0;
		Matrix g=G(obs,nbasis,densitymatrix);
		Matrix F=Hcore+g;
		Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F,overlap);
		OrbitalEnergies=gen_eig_solver.eigenvalues();
		CoefficientMatrix=gen_eig_solver.eigenvectors();
		Matrix C_occ=CoefficientMatrix.leftCols(nocc);
		densitymatrix=C_occ*C_occ.transpose();
		Matrix whatever=densitymatrix*(Hcore+F);
		for (int i=0;i<nbasis;i++){
			energy+=whatever(i,i);
		}
	}while (abs(lastenergy-energy)>0.00001);
	Energy=energy;
	DensityMatrix=densitymatrix;
	MOJK(JMatrix,KMatrix,obs,nbasis,CoefficientMatrix);
}
			

	

