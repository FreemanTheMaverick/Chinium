#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <functional>
#include <vector>
#include <tuple>
#include <deque>
#include <cstdio>
#include <chrono>
#include <cassert>

#include "../Macro.h"

#include "Loong.h"

#include <iostream>


EigenMatrix OrthogonalExp(EigenMatrix p, EigenMatrix X){
	// p - Point of tangency
	// X - Point to retract
	const EigenMatrix A = 0.5 * ( X * p.transpose() - p * X.transpose() );
	return p * A.exp();
}

EigenMatrix OrthogonalProj(EigenMatrix p, EigenMatrix X){
	return 0.5 * ( X - p * X.transpose() * p );
}

EigenMatrix OrthogonalGrad(EigenMatrix p, EigenMatrix Ge){
	return OrthogonalProj(p, Ge);
}

bool Rabbit(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<
					EigenMatrix
					(
						EigenMatrix,
						std::vector<EigenMatrix>
					)
				>
			>
			(
				EigenMatrix
			)
		>& func,
		std::tuple<double, double, double> tol,
		int max_iter,
		double& L, EigenMatrix& U, bool output){
	if (output){
		std::printf("Using Rabbit optimizer\n");
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", std::get<0>(tol));
		std::printf("| Gradient norm (Grad.)               : %E\n", std::get<1>(tol));
		std::printf("| Independent variable update (V. U.) : %E\n", std::get<2>(tol));
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  | Update |  V. U.  |  Time  |\n");
	}

	double Llast = 0;
	EigenMatrix Ulast = EigenZero(U.rows(), U.cols());
	const auto start = __now__;
	
	for ( int iiter = 0; iiter < max_iter; iiter++ ){
		if (output) std::printf("| %4d |", iiter);

		EigenMatrix Ge;
		std::function<EigenMatrix (EigenMatrix, std::vector<EigenMatrix>)> newton_update;
		std::tie(L, Ge, newton_update) = func(U);
		const EigenMatrix Gr = OrthogonalGrad(U, Ge);

		const double deltaL = L - Llast;
		Llast = L;

		if (output) std::printf("  %17.10f  | % 5.1E | %5.1E |", L, deltaL, Gr.norm());

		const std::vector<EigenMatrix> params = {Gr, Ge - Gr};
		EigenMatrix D = EigenZero(U.rows(), U.cols());
		if ( Loong(newton_update, params, 1.e-6, 100, 100, D, 0) )
			std::printf("Warning: Loong convergence failed!" );
		const EigenMatrix Unew = OrthogonalExp(U, D);
		const double deltaU = (Unew - U).norm();
		U = Unew;

		if (output) std::printf(" RieNew | %5.1E | %6.3f |\n", deltaU, __duration__(start, __now__));

		if ( Gr.norm() < std::get<1>(tol) ){
			if ( iiter == 0 ){
				if ( deltaU < std::get<2>(tol) ) return 1;
			}else{
				if ( std::abs(deltaL) < std::get<0>(tol) && deltaU < std::get<2>(tol) ) return 1;
			}
		}

	}

	return 0;
}

