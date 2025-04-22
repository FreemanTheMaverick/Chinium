#include <Eigen/Dense>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <string>
#include <cassert>
#include <vector>

#include "../Macro.h"
#include "../Multiwfn/Multiwfn.h"
#include "../Integral/Int2C1E.h"
#include "../Grid/GridPotential.h"

#include <iostream>

#define __nlines__ 751

EigenMatrix SuperpositionAtomicPotential(
		std::string path, std::vector<MwfnCenter>& centers, int nbasis,
		double* xs, double* ys, double* zs, double* ws, long int ngrids, double* aos){
	__Z_2_Name__
	double * vsap = new double[ngrids]();
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
			const double x = xs[igrid] - atomx;
			const double y = ys[igrid] - atomy;
			const double z = zs[igrid] - atomz;
			const double r = std::sqrt(x * x + y * y + z * z) + 1.e-12;
			double bestdiff = 114514;
			double bestvap = 114514;
			for ( int iline = 0; iline < __nlines__; iline++ ){
				bestdiff = ( bestdiff < std::abs( Rs[iline] - r ) ) ? bestdiff : std::abs( Rs[iline] - r );
				bestvap = ( bestdiff < std::abs( Rs[iline] - r ) ) ? bestvap : Zs[iline] / r;
			}
			vsap[igrid] += bestvap;
		}
	}
	return VMatrix(
			{0},
			ws, ngrids, nbasis,
			aos,
			nullptr, nullptr, nullptr,
			nullptr,
			nullptr, nullptr, nullptr,
			vsap, nullptr,
			nullptr, nullptr);
}

void GuessSCF(Multiwfn& mwfn, Int2C1E& int2c1e, std::string guess, const bool output){
	std::string path = std::getenv("CHINIUM_PATH");
	path += "/SAP/";
	EigenMatrix V = EigenZero(mwfn.getNumBasis(), mwfn.getNumBasis());
	bool potential = 0;
	if ( guess.compare("SAP") == 0 ){
		if (output) std::printf("SCF initial guess type ... SAP\n");
		if (output) std::printf("Calculating superposition of atomic potential ... ");
		const auto start = __now__;
		potential = 1;
		V = SuperpositionAtomicPotential(
			path, mwfn.Centers, mwfn.getNumBasis(),
			mwfn.Xs, mwfn.Ys, mwfn.Zs,
			mwfn.Ws, mwfn.NumGrids, mwfn.AOs);
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
			if ( mwfn.Temperature > 0 ){
				const EigenArray n = 1. / ( 1. + ( ( eps.array() - mwfn.ChemicalPotential ) / mwfn.Temperature ).exp() );
				if ( spin == 0 ) mwfn.setOccupation(2 * n, 0);
				else mwfn.setOccupation(n, spin);
			}
		}
	}
}
