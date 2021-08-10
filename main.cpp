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
}

	

