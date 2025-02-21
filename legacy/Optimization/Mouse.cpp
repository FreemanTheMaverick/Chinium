#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <functional>
#include <tuple>
#include <deque>
#include <cstdio>
#include <chrono>
#include <cassert>

#include "../Macro.h"

#include "DIIS.h"

#include <iostream>

#define check(x) assert(x.array().isNaN().any());

bool Mouse(
		std::function<std::tuple<double, EigenMatrix, EigenMatrix, EigenMatrix> (EigenMatrix)>& ffunc,
		std::tuple<double, double, double> adtol,
		std::tuple<double, double, double> tol,
		int diis_space, int max_iter,
		double& E, EigenMatrix& F, int output){
	if (output > 0){
		std::printf("Using Mouse optimizer\n");
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", std::get<0>(tol));
		std::printf("| Gradient norm (Grad.)               : %E\n", std::get<1>(tol));
		std::printf("| Independent variable update (V. U.) : %E\n", std::get<2>(tol));
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  | Update |  V. U.  |  Time  |\n");
	}

	EigenMatrix Fupdate = F;
	EigenMatrix G = EigenZero(F.cols(), F.cols());
	EigenMatrix D = EigenZero(F.cols(), F.cols());
	std::deque<double> Es = {};
	std::deque<EigenMatrix> Fs = {};
	std::deque<EigenMatrix> Gs = {};
	std::deque<EigenMatrix> Ds = {};
	const auto start = __now__;
	
	for ( int iiter = 0; iiter < max_iter; iiter++ ){
		if (output > 0) std::printf("| %4d |", iiter);

		std::tie(E, F, G, D) = ffunc(F);
		const double deltaE = ( iiter == 0 ) ? E : ( E - Es.back() );

		if (output > 0) std::printf("  %17.10f  | % 5.1E | %5.1E |", E, deltaE, G.norm());

		Es.push_back(E);
		Fs.push_back(F);
		Gs.push_back(G);
		Ds.push_back(D);
		if ( (int)Es.size() == diis_space ){
			Es.pop_front();
			Fs.pop_front();
			Gs.pop_front();
			Ds.pop_front();
		}

		if ( Es.size() < 2 ){
			if (output > 0) std::printf("  Naive |");
		}else if ( G.norm() > std::get<1>(adtol) || iiter < 3 ){
			if (output > 0) std::printf("  ADIIS |");
			EigenMatrix AD1s = EigenZero(Ds.size(), 1);
			EigenMatrix AD2s = EigenZero(Ds.size(), Ds.size());
			for ( int i = 0; i < (int)Ds.size(); i++ ){
				AD1s(i) = ( ( Ds[i] - D ) * F.transpose() ).trace();
				for ( int j = 0; j < (int)Ds.size(); j++ )
					AD2s(i, j) = ( ( Ds[i] - D ) * ( Fs[j] - F ).transpose() ).trace();
			}
			F = ADIIS(AD1s, AD2s, Fs, output-1);
		}else{
			if (output > 0) std::printf("  CDIIS |");
			F = CDIIS(Gs, Fs);
		}

		const double deltaF = ( F - Fs.back() ).norm();
		if (output > 0) std::printf(" %5.1E | %6.3f |\n", deltaF, __duration__(start, __now__));

		if ( G.norm() < std::get<2>(tol) ){
			if ( iiter == 0 ) return 1;
			if ( std::abs(deltaE) < std::get<0>(tol) && deltaF < std::get<1>(tol) ) return 1;
		}

	}
	return 0;
}

