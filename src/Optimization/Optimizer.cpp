#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <tuple>
#include <deque>
#include <cstdio>
#include <chrono>
#include "../Macro.h"
#include "DIIS.h"

bool Mouse(
		std::function<std::tuple<double, EigenMatrix, EigenMatrix>
		(EigenMatrix, int)>& func,
		std::tuple<double, double, double> tol,
		int diis_space, int max_iter,
		double& Y, EigenMatrix& X, bool output){
	if (output){
		std::printf("Using Mouse optimizer\n");
		std::printf("Tolerance:\n");
		std::printf("| Target change : %E\n", std::get<0>(tol));
		std::printf("| Gradient      : %E\n", std::get<2>(tol));
		std::printf("| Iteration |   Update    |        Target         | Gradient norm | Wall time |\n");
	}
	
	std::deque<EigenMatrix> Xs;
	std::deque<double> Ys;
	std::deque<EigenMatrix> Gs;
	const auto start = __now__;
	
	for ( int iiter = 0; iiter < max_iter; iiter++ ){
		if (output) std::printf("|   %5d   |",iiter);

		EigenMatrix G;
		std::tie(Y, G, std::ignore)	= func(X, 1);
		EigenMatrix deltaX;
		if ( iiter == 0 ){
			if (output) std::printf("   Steepest  |");
			deltaX = - 0.001 * G;
		}else{
			if (output) std::printf("    CDIIS    |");
			deltaX = DIIS(Xs, Gs) - X;
		}

		if (output) std::printf("   %17.10f   | %13.6f | %9.6f |\n", Y, G.norm(), (float)__duration__(start, __now__));

		if ( G.norm() < std::get<2>(tol) ){
			if ( iiter == 0 ) return 1;
			if ( std::abs(Y - Ys.back()) < std::get<0>(tol) ) return 1;
		}

		X += deltaX;

		Xs.push_back(X);
		Ys.push_back(Y);
		Gs.push_back(G);
		if ( (int)Xs.size() == diis_space ){
			Xs.pop_front();
			Ys.pop_front();
			Gs.pop_front();
		}
	}
	return 0;
}


