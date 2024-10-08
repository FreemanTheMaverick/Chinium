#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <functional>
#include <tuple>
#include <cstdio>
#include <chrono>
#include <cassert>

#include "../Macro.h"

#include <iostream>


EigenVector SimplexExp(EigenVector p, EigenVector X){
	// p - Point of tangency
	// X - Point to retract
	const EigenVector Xp = X.cwiseProduct(p.array().rsqrt().matrix());
	const double norm = Xp.norm();
	const EigenVector Xpn = Xp / norm;
	const EigenVector tmp1 = 0.5 * (p + Xpn.cwiseProduct(Xpn));
	const EigenVector tmp2 = 0.5 * (p - Xpn.cwiseProduct(Xpn)) * std::cos(norm);
	const EigenVector tmp3 = Xpn.cwiseProduct(p.array().sqrt().matrix()) * std::sin(norm);
	return tmp1 + tmp2 + tmp3;
}

EigenMatrix SimplexProj(EigenVector p){
	const int n = p.size();
	const EigenMatrix ones = EigenZero(n, n).array() + 1;
	const EigenMatrix tmp2 = ones.array().colwise() * p.array();
	return EigenOne(n, n) - tmp2;
}

EigenVector SimplexProj(EigenVector p, EigenVector X){
	return SimplexProj(p) * X;
}

EigenVector SimplexGrad(EigenVector p, EigenVector Ge){
	return SimplexProj(p, p.cwiseProduct(Ge));
}

EigenVector SimplexNewton(EigenMatrix M, EigenMatrix N, EigenVector Gr, EigenMatrix He){
	EigenMatrix LHS = EigenZero(M.rows() + 1, M.cols());
	EigenMatrix hess = M * He + N;
	LHS.topRows(M.rows()) = hess;
	LHS.bottomRows(1) = EigenZero(1, M.cols()).array() + 1;
	EigenVector RHS(M.rows() + 1);
	RHS.topRows(M.rows()) = - Gr;
	RHS(M.rows()) = 0;
	return LHS.colPivHouseholderQr().solve(RHS);
}

bool Ox(std::function<std::tuple<double, EigenVector, EigenMatrix> (EigenVector)>& func,
		std::tuple<double, double, double> tol,
		int max_iter,
		double& L, EigenVector& C, bool output){
	if (output){
		std::printf("Using Ox optimizer\n");
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", std::get<0>(tol));
		std::printf("| Gradient norm (Grad.)               : %E\n", std::get<1>(tol));
		std::printf("| Independent variable update (V. U.) : %E\n", std::get<2>(tol));
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  | Update |  V. U.  |  Time  |\n");
	}

	EigenMatrix Gr;
	double Llast = 0;
	EigenMatrix Clast = EigenZero(C.rows(), C.cols());
	
	for ( int iiter = 0; iiter < max_iter; iiter++ ){
		if (output) std::printf("| %4d |", iiter);
		const auto start = __now__;

		EigenVector Ge;
		EigenMatrix He;
		std::tie(L, Ge, He) = func(C);
		Gr = SimplexGrad(C, Ge);

		const double deltaL = L - Llast;
		Llast = L;

		if (output) std::printf("  %17.10f  | % 5.1E | %5.1E |", L, deltaL, Gr.norm());

		const EigenMatrix ones = EigenZero(C.size(), C.size()).array() + 1;
		const EigenMatrix proj = SimplexProj(C);
		const EigenMatrix M = proj * (EigenMatrix)C.asDiagonal();
		const EigenMatrix N = proj * (EigenMatrix)(
				Ge
				- ones * Ge.cwiseProduct(C)
				- 0.5 * Gr.cwiseProduct(C.cwiseInverse())
		).asDiagonal();
		const EigenMatrix Cnew = SimplexExp(C, SimplexNewton(M, N, Gr, He));
		const double deltaC = (Cnew - C).norm();
		C = Cnew;

		if (output) std::printf(" RieNew | %5.1E | %6.3f |\n", deltaC, __duration__(start, __now__));

		if ( Gr.norm() < std::get<1>(tol) ){
			if ( iiter == 0 ){
				if ( deltaC < std::get<2>(tol) ) return 1;
			}else{
				if ( std::abs(deltaL) < std::get<0>(tol) && deltaC < std::get<2>(tol) ) return 1;
			}
		}

	}

	return 0;
}

