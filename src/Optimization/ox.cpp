#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <tuple>
#include <cstdio>
#include <chrono>
#include <cassert>

#include "../Macro.h"

#include "Loong.h"

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

double SimplexInner(EigenVector p, EigenVector X, EigenVector Y){
	return p.cwiseInverse().cwiseProduct(X.cwiseProduct(Y)).sum();
}

bool Ox(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<EigenMatrix (EigenMatrix)>
			> (EigenMatrix)>& func,
		std::tuple<double, double, double> tol,
		int max_iter,
		double& L, EigenMatrix& C, bool output){
	if (output){
		std::printf("Using Ox optimizer\n");
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", std::get<0>(tol));
		std::printf("| Gradient norm (Grad.)               : %E\n", std::get<1>(tol));
		std::printf("| Independent variable update (V. U.) : %E\n", std::get<2>(tol));
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  | Update |  V. U.  |  Time  |\n");
	}

	const auto start = __now__;
	const double R0 = 1;
	const double rho_thres = 0.1;
	EigenMatrix Ge;
	std::function<EigenMatrix (EigenMatrix)> He;
	std::tie(L, Ge, He) = func(C);
	double deltaL = L;
	double R = R0;
	
	for ( int iiter = 0; iiter < max_iter; iiter++ ){

		const EigenVector Gr = SimplexGrad(C, Ge);
		if (output) std::printf("| %4d |  %17.10f  | % 5.1E | %5.1E |", iiter, L, deltaL, Gr.norm());

		// Riemannian hessian
		const EigenMatrix ones = EigenZero(C.size(), C.size()).array() + 1;
		const EigenMatrix proj = SimplexProj(C);
		const EigenMatrix M = proj * (EigenMatrix)C.asDiagonal();
		const EigenMatrix N = proj * (EigenMatrix)(
				Ge
				- ones * Ge.cwiseProduct(C)
				- 0.5 * Gr.cwiseProduct(C.cwiseInverse())
		).asDiagonal();

		std::function<EigenMatrix (EigenMatrix)> Hr = [&He, M, N](EigenMatrix v_){
			return (EigenMatrix)(M * He(v_) + N * v_); // The forced conversion "(EigenMatrix)" is necessary. Without it the result will be wrong. I do not know why. Then I forced convert every EigenMatrix return value in std::function for ensurance.
		};
		std::function<double (EigenMatrix, EigenMatrix)> Inner = [C](EigenMatrix v1_, EigenMatrix v2_){
			return SimplexInner(C, v1_, v2_);
		};
		std::function<EigenMatrix (EigenMatrix)> Proj = [C](EigenMatrix v_){
			return SimplexProj(C, v_);
		};

		// Truncated conjugate gradient and rating the new step
		const EigenMatrix S = Loong(Inner, Hr, -Gr, R, C.size()-1, 1);
		const double S2 = SimplexInner(C, S, S);
		const EigenMatrix Cnew = SimplexExp(C, S);
		double Lnew;
		EigenMatrix Genew;
		std::function<EigenMatrix (EigenMatrix)> Henew;
		std::tie(Lnew, Genew, Henew) = func(Cnew);
		const double top = Lnew - L;
		const double bottom = SimplexInner(C, Gr + 0.5 * Hr(S), S);
		const double rho = top / bottom;

		// Determining whether to accept or reject the step
		const double deltaC = (Cnew - C).norm();
		if ( rho > rho_thres ){
			C = Cnew;
			L = Lnew;
			deltaL = Lnew - L;
			Ge = Genew;
			He = Henew;
			if (output) std::printf(" Accept |");
		}else if (output) std::printf(" Reject |");
		if (output) std::printf(" %5.1E | %6.3f |\n", deltaC, __duration__(start, __now__));

		// Adjusting the trust radius according to the score
		if ( rho < 0.25 ) R = 0.25 * R;
		else if ( rho > 0.75 || std::abs(S2 - R * R) < 1.e-10 ) R = std::min(2 * R, R0);

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

