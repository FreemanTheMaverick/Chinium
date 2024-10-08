#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <tuple>
#include <deque>
#include <cstdio>
#include <chrono>

#include "../Macro.h"

#include "DIIS.h"

#include <iostream>

bool Loong(
		std::function<
			EigenMatrix
			(
				EigenMatrix,
				std::vector<EigenMatrix>
			)
		>& newton_update, std::vector<EigenMatrix> params,
		double tol, int diis_space, int maxiter, EigenMatrix& D, bool output){
	if (output){
		std::printf("Using Loong optimizer\n");
		std::printf("Convergence threshold:\n");
		std::printf("| Error                               : %E\n", tol);
		std::printf("| Itn. |       Residual      | Update |  Time  |\n");
	}

	EigenMatrix Din = D;
	std::deque<EigenMatrix> Ds;
	std::deque<EigenMatrix> Rs;
	const auto start = __now__;
	
	for ( int iiter = 0; iiter < maxiter; iiter++ ){
		if (output) std::printf("| %4d |", iiter);

		D = newton_update(Din, params);
		const EigenMatrix R = D - Din;

		if (output) std::printf("  %17.10f  |", R.norm());

		Ds.push_back(D);
		Rs.push_back(R);
		if ( (int)Ds.size() > diis_space ){
			Ds.pop_front();
			Rs.pop_front();
		}

		if ( Ds.size() < 2 ){
			if (output) std::printf("  Naive |");
			Din = D;
		}else{
			if (output) std::printf("  CDIIS |");
			Din = CDIIS(Rs, Ds);
		}

		if (output) std::printf(" %6.3f |\n", __duration__(start, __now__));

		if ( R.norm()  < tol ) return 1;
	}
	return 0;
}
