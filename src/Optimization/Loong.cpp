#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <cstdio>
#include <chrono>
#include <cassert>

#include "../Macro.h"

#include <iostream>


EigenMatrix Loong(
		std::function<double (EigenMatrix, EigenMatrix)>& Inner,
		std::function<EigenMatrix (EigenMatrix)>& Hess,
		EigenMatrix b, double R, int ndim, bool output){

	const double b2 = Inner(b, b);
	const double tol = std::max(1e-8, std::sqrt(b2 * std::min(b2, 0.1)));

	if (output){
		std::printf("Using Loong optimizer\n");
		std::printf("Convergence threshold:\n");
		std::printf("| Error                               : %E\n", tol);
		std::printf("| Itn. |       Residual      |  Time  |\n");
	}

	EigenMatrix v = EigenZero(b.rows(), b.cols());
	EigenMatrix r = b;
	EigenMatrix p = b;
	double r2 = b2;
	const auto start = __now__;

	for ( int iiter = 0; iiter < ndim; iiter++ ){
		if (output) std::printf("| %4d |  %17.10f  | %6.3f |\n", iiter, std::sqrt(r2), __duration__(start, __now__));
		const EigenMatrix Hp = Hess(p);
		const double pHp = Inner(p, Hp);
		const double alpha = r2 / pHp;
		const EigenMatrix vplus = v + alpha * p;
		if ( pHp <= 0 || Inner(vplus, vplus) >= R * R ){
			const double A = Inner(p, p);
			const double B = Inner(v, p) * 2.;
			const double C = Inner(v, v) - R * R;
			const double t = ( std::sqrt( B * B - 4. * A * C ) - B ) / 2. / A;
			if (output && pHp <= 0) std::printf("Non-positive curvature!\n");
			if (output && Inner(vplus, vplus) >= R * R) std::printf("Out of trust region!\n");
			return v + t * p;
		}
		v = vplus;
		const double r2old = r2;
		r -= alpha * Hp;
		r2 = Inner(r, r);
		if ( r2 <= tol * tol ){
			if (output) std::printf("Tolerance met!\n");
			return v;
		}
		const double beta = r2 / r2old;
		p = r + beta * p;
	}
	if (output) std::printf("Dimension completed!\n");
	return v;
}


