#include "HartreeFock.h"
#include <iostream>
#include <libint2.hpp>


int main(){
	initialize();
	HartreeFockJob job;
	job.setXYZ("f.xyz");
	job.setBasisSet("sto-3g");
	job.setGuess("123");
	job.Compute();
	finalize();
	std::cout<<"HF energy: "<<job.Energy<<std::endl;
	std::cout<<"Orbital energies:"<<std::endl;
	std::cout<<job.OrbitalEnergies<<std::endl;
	std::cout<<"Coefficient matrix:"<<std::endl;
	std::cout<<job.CoefficientMatrix<<std::endl;
	std::cout<<"Density matrix:"<<std::endl;
	std::cout<<job.DensityMatrix<<std::endl;
	std::cout<<"J matrix:"<<std::endl;
	std::cout<<job.JMatrix<<std::endl;
	std::cout<<"K matrix:"<<std::endl;
	std::cout<<job.KMatrix<<std::endl;

}

	

