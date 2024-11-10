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

EigenMatrix DIITS(std::deque<EigenMatrix>& Xs, std::deque<EigenMatrix>& Grs){
	const int length = Xs.size();
	std::deque<EigenMatrix> pXs(length); // Projections of Xs on the Grassmann manifold to the tangent space of the last X.
	std::deque<EigenMatrix> Gts(length);
	for ( int i = 0; i < length; i++ ){
		pXs[i] = GrassmannLog(Xs[i], Xs.back());
		const EigenMatrix tmp = GrassmannLog(Xs[length - 1], Xs[i]); // At the tangent space of a previous point, finding the log map of the current point.
		Gts[i] = GrassmannTransport(Xs[i], Grs[i], tmp); // Parallel transporting the Riemann gradient of previous points to the tangent space of the current one.
	}
	EigenMatrix pX = CDIIS(pXs, Gts); // Extrapolation among the projected density matrix on the tangent space.
	double norm = pX.norm();
	if ( norm > 1 ) pX /= norm;
	return GrassmannExp(Xs.back(), pX);
}

bool Tiger(
		std::function<std::tuple<double, EigenMatrix> (EigenMatrix)>& func,
		std::tuple<double, double, double> tol,
		int diis_space, int max_iter,
		double& L, EigenMatrix& X, bool output){
	if (output){
		std::printf("Using Tiger optimizer\n");
		std::printf("Tolerance:\n");
		std::printf("| Target change : %E\n", std::get<0>(tol));
		std::printf("| Gradient      : %E\n", std::get<2>(tol));
		std::printf("| Iteration |   Update    |        Target         | Gradient norm | Wall time |\n");
	}
	
	std::deque<EigenMatrix> Xs;
	std::deque<double> Ls;
	std::deque<EigenMatrix> Grs;
	const auto start = __now__;
	
	for ( int iiter = 0; iiter < max_iter; iiter++ ){
		if (output) std::printf("|   %5d   |",iiter);

		EigenMatrix Ge;
		std::tie(L, Ge) = func(X);

		const EigenMatrix Gr = GrassmannGrad(X, Ge);
		//std::cout<<Gr<<std::endl;

		Xs.push_back(X);
		Ls.push_back(L);
		Grs.push_back(Gr);
		if ( (int)Xs.size() == diis_space ){
			Xs.pop_front();
			Ls.pop_front();
			Grs.pop_front();
		}
		
		if ( iiter < 1000 ){
			X = GrassmannExp(X, - 0.1 * Gr);
			std::printf("  Steepest   |");
		}else{
			X = DIITS(Xs, Grs);
			std::printf("    DIITS    |");
		}
		
		const double deltaX = (X - Xs.back()).norm();

		if (output) std::printf("  %17.10f  | %13.6f | %9.6f |\n", L, Gr.norm(), __duration__(start, __now__));

		if ( Gr.norm() < std::get<2>(tol) ){
			if ( iiter == 0 ) return 1;
			if ( std::abs(L - Ls.back()) < std::get<0>(tol) ) return 1;
		}

	}
	return 0;
}

