#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <string>
#include <cassert>
#include <vector>
#include <libmwfn.h>

#include "../Macro.h"
#include "../Integral/Int2C1E.h"
#include "../Grid/Grid.h"

#include <iostream>

#define __nlines__ 751

EigenMatrix SuperpositionAtomicPotential(std::string path, std::vector<MwfnCenter>& centers, Grid& grid){
	int ngrids = grid.NumGrids;
	__Z_2_Name__
	grid.E1Rhos.resize(ngrids); grid.E1Rhos.setZero();
	for ( MwfnCenter& center : centers ){
		const std::string atomname = Z2Name[center.Index];
		const double atomx = center.Coordinates[0];
		const double atomy = center.Coordinates[1];
		const double atomz = center.Coordinates[2];
		std::ifstream sapfile( path + "/v_" + atomname + ".dat" );
		assert("SAP file is missing!" && sapfile.good());
		double Rs[__nlines__];
		double Zs[__nlines__];
		for ( int iline=0 ; iline < __nlines__; iline++ ){
			std::string thisline;
			std::getline(sapfile, thisline);
			std::stringstream ss(thisline);
			ss >> Rs[iline];
			ss >> Zs[iline];
		}
		for ( long int igrid = 0; igrid < ngrids; igrid++ ){
			const double x = grid.Xs[igrid] - atomx;
			const double y = grid.Ys[igrid] - atomy;
			const double z = grid.Zs[igrid] - atomz;
			const double r = std::sqrt(x * x + y * y + z * z) + 1.e-12;
			double bestdiff = 114514;
			double bestvap = 114514;
			for ( int iline = 0; iline < __nlines__; iline++ ){
				bestdiff = ( bestdiff < std::abs( Rs[iline] - r ) ) ? bestdiff : std::abs( Rs[iline] - r );
				bestvap = ( bestdiff < std::abs( Rs[iline] - r ) ) ? bestvap : Zs[iline] / r;
			}
			grid.E1Rhos(igrid) += bestvap;
		}
	}
	return grid.getFock(0);
}

void GuessSCF(Mwfn& mwfn, Environment& env, Int2C1E& int2c1e, Grid& grid, std::string guess, const bool output){
	EigenMatrix V = EigenZero(mwfn.getNumBasis(), mwfn.getNumBasis());
	bool potential = 0;
	if ( guess == "SAP" ){
		std::string path = std::getenv("CHINIUM_PATH"); path += "/SAP/";
		if (output) std::printf("SCF initial guess type ... SAP\n");
		if (output) std::printf("Calculating superposition of atomic potential ... ");
		const auto start = __now__;
		potential = 1;
		V = SuperpositionAtomicPotential(path, mwfn.Centers, grid);
		if (output) std::printf("%f s\n", __duration__(start, __now__));
	}else assert("Unrecognized initial guess type!" && 0);

	if (potential){
		Eigen::GeneralizedSelfAdjointEigenSolver<EigenMatrix> solver;
		solver.compute(int2c1e.Kinetic + int2c1e.Nuclear + V, int2c1e.Overlap);
		for ( int spin : ( mwfn.Wfntype == 0 ? std::vector<int>{0} : std::vector<int>{1, 2} ) ){
			const EigenMatrix C = solver.eigenvectors();
			const EigenMatrix eps = solver.eigenvalues();
			mwfn.setCoefficientMatrix(C, spin);
			mwfn.setEnergy(eps, spin);
			if ( env.Temperature > 0 ){
				const EigenArray n = 1. / ( 1. + ( ( eps.array() - env.ChemicalPotential ) / env.Temperature ).exp() );
				if ( spin == 0 ) mwfn.setOccupation(2 * n, 0);
				else mwfn.setOccupation(n, spin);
			}
		}
	}
}
