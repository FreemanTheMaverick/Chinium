#include "HartreeFock.h"
#include "MBPT.h"
#include <iostream>
#include <libint2.hpp>


int main(){
	initialize();
	HartreeFockJob hfjob;
	string filename,basisname;
	std::cin>>filename>>basisname;
	//filename="f.xyz";
	//basisname="sto-3g";
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
	std::cout<<"J matrix:"<<std::endl;
	std::cout<<hfjob.JMatrix<<std::endl;
	std::cout<<"K matrix:"<<std::endl;
	std::cout<<hfjob.KMatrix<<std::endl;
	
	MP2Job mp2job;
	mp2job.setXYZ(filename);
	mp2job.setBasisSet(basisname);
	mp2job.setCoefficientMatrix(DuplicateCol(hfjob.CoefficientMatrix));
	mp2job.setOrbitalEnergies(DuplicateRow(hfjob.OrbitalEnergies));
	mp2job.Compute();
	finalize();
	std::cout<<"MP2 correlation energy: "<<mp2job.CorrelationEnergy<<std::endl;
}

	

