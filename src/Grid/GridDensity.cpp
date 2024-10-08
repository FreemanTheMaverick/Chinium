#include <Eigen/Core>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdio>

#include "../Macro.h"
#include "../Multiwfn.h"

#include <iostream>

void Density(double* aos, long int ngrids, EigenMatrix D, double* ds){
	double* iao = aos;
	double* jao = aos;
	double Dij;
	for ( int ibasis = 0; ibasis < D.cols(); ibasis++ ){
		Dij = D(ibasis, ibasis);
		iao = aos + ibasis * ngrids; // ibasis*ngrids+jgrid
		for ( int kgrid = 0; kgrid < ngrids; kgrid++ )
			ds[kgrid] += Dij * iao[kgrid] * iao[kgrid];
		for ( int jbasis = 0; jbasis < ibasis; jbasis++ ){
			Dij = D(ibasis, jbasis);
			jao = aos + jbasis * ngrids;
			for ( long int kgrid = 0; kgrid < ngrids; kgrid++ )
				ds[kgrid] += 2 * Dij * iao[kgrid] * jao[kgrid];
		}
	}
}

void Multiwfn::getGridDensity(EigenMatrix D, const bool output){
	if (output) std::printf("Calculating density on grids ... ");
	auto start = __now__;
	this->GridDensity = new double[this->NumGrids]();
	Density(this->AOs, this->NumGrids, D, this->GridDensity);
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
}
