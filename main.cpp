#include "HartreeFock.h"
#include "MBPT.h"
#include <iostream>
#include <libint2.hpp>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
int main(){
	initialize();
	HartreeFockJob hfjob;
	string filename,basisname;
	//std::cin>>filename>>basisname;
	filename="he.xyz";
	basisname="cc-pv5z";
	hfjob.setXYZ(filename);
	hfjob.setBasisSet(basisname);
	hfjob.setGuess("hcore");
	hfjob.Compute();
	std::cout<<"HF energy: "<<hfjob.Energy<<std::endl;
	std::cout<<"Orbital energies:"<<std::endl;
	std::cout<<hfjob.OrbitalEnergies<<std::endl;
	std::cout<<"Coefficient matrix:"<<std::endl;
	std::cout<<hfjob.CoefficientMatrix<<std::endl;
	std::cout<<"Density matrix:"<<std::endl;
	std::cout<<hfjob.DensityMatrix<<std::endl;
	std::cout<<"H matrix:"<<std::endl;
	std::cout<<hfjob.HMatrix<<std::endl;

	
	MP2Job mp2job;
	mp2job.setOrbitalEnergies(hfjob.OrbitalEnergies);
	mp2job.setMO2e(hfjob.MO_2e);
	mp2job.setnElectron(nElectron(hfjob.Atoms));
	mp2job.setShellType("Restricted");
	mp2job.Compute();
	std::cout<<"MP2 correlation energy: "<<mp2job.CorrelationEnergy<<std::endl;

	finalize();
}

	

