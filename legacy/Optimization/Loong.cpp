#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <cstdio>
#include <chrono>
#include <cassert>
#include <string>
#include <tuple>

#include "../Macro.h"
#include "../Manifold/Manifold.h"

#include <iostream>


EigenMatrix Loong(
		Manifold& M, double R,
		std::tuple<double, double, double> tol, int output){

	const double tol0 = std::get<0>(tol) * M.getDimension();
	const double tol1 = std::get<1>(tol) * M.P.size();
	const double tol2 = std::get<2>(tol) * M.P.size();
	if (output > 0){
		std::printf("Using Loong optimizer on the tangent space of %s manifold\n", M.Name.c_str());
		std::printf("Convergence threshold:\n");
        std::printf("| Target change (T. C.)               : %E\n", tol0);
		std::printf("| Gradient norm (Grad.)               : %E\n", tol1);
		std::printf("| Independent variable update (V. U.) : %E\n", tol2);
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  |  V. U.  |  Time  |\n");
	}

	const double b2 = M.Inner(M.Gr, M.Gr);
	EigenMatrix v = EigenZero(M.Gr.rows(), M.Gr.cols());
	EigenMatrix r = -M.Gr;
	EigenMatrix p = -M.Gr;
	double r2 = b2;
	double L = 0;
	const auto start = __now__;

	EigenMatrix Hp = EigenZero(M.Gr.rows(), M.Gr.cols());
	EigenMatrix vplus = EigenZero(M.Gr.rows(), M.Gr.cols());

	for ( int iiter = 0; iiter < M.getDimension(); iiter++ ){
		if (output > 0) std::printf("| %4d |", iiter);
		Hp = M.TangentPurification(M.Hr(p));
		const double pHp = M.Inner(p, Hp);
		const double Llast = L;
		L = 0.5 * M.Inner(M.Hr(v), v) + M.Inner(M.Gr, v);
		const double deltaL = L - Llast;
		if (output > 0) std::printf("  %17.10f  | % 5.1E | %5.1E |", L, deltaL, std::sqrt(r2));

		const double alpha = r2 / pHp;
		vplus = M.TangentPurification(v + alpha * p);
		const double step = std::abs(alpha) * std::sqrt(M.Inner(p, p));
		if (output > 0) std::printf(" %5.1E | %6.3f |\n", step, __duration__(start, __now__));
		if (
				iiter > 0 && (
					(
						std::abs(deltaL) < tol0
						&& r2 < std::pow(tol1, 2)
						&& step < tol2
					) || std::abs(deltaL) < tol0 * tol0 /1000
				)
		){
			if (output > 0) std::printf("Tolerance met!\n");
			return v;
		}

		if ( pHp <= 0 || M.Inner(vplus, vplus) >= R * R ){
			const double A = M.Inner(p, p);
			const double B = M.Inner(v, p) * 2.;
			const double C = M.Inner(v, v) - R * R;
			const double t = ( std::sqrt( B * B - 4. * A * C ) - B ) / 2. / A;
			if (output > 0 && pHp <= 0) std::printf("Non-positive curvature!\n");
			if (output > 0 && M.Inner(vplus, vplus) >= R * R) std::printf("Out of trust region!\n");
			return v + t * p;
		}
		v = vplus;
		const double r2old = r2;
		r = M.TangentPurification(r - alpha * Hp);
		r2 = M.Inner(r, r);
		const double beta = r2 / r2old;
		p = M.TangentPurification(r + beta * p);
	}
	if (output > 0) std::printf("Dimension completed!\n");
	return v;
}


