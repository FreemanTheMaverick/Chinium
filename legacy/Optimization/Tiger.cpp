#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <tuple>
#include <cstdio>
#include <chrono>
#include <string>

#include "../Macro.h"
#include "../Manifold/Manifold.h"
#include "Loong.h"

#include <iostream>


bool Tiger(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<EigenMatrix (EigenMatrix)>
			> (EigenMatrix)
		>& func,
		std::tuple<double, double, double> tol,
		int max_iter,
		double& L, Manifold& M, Manifold& M_last, int output){

	const double tol0 = std::get<0>(tol) * M.getDimension();
	const double tol1 = std::get<1>(tol) * M.P.size();
	const double tol2 = std::get<2>(tol) * M.P.size();
	if (output > 0){
		std::printf("Using Tiger optimizer on %s\n", M.Name.c_str());
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", tol0);
		std::printf("| Gradient norm (Grad.)               : %E\n", tol1);
		std::printf("| Independent variable update (V. U.) : %E\n", tol2);
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  | Update |  V. U.  |  Time  |\n");
	}

	const auto start = __now__;
	const double R0 = 3;
	const double rho_thres = 0.1;
	std::tie(L, M.Ge, M.He) = func(M.P);
	double deltaL = L;
	double R = R0;
	EigenMatrix S_BFGS = EigenZero(M.Gr.rows(), M.Gr.cols());
	EigenMatrix Y_BFGS = EigenZero(M.Gr.rows(), M.Gr.cols());
	bool started = 0;

	for ( int iiter = 0; iiter < max_iter; iiter++ ){

		M.getGradient();
		Y_BFGS += M.Gr;
		if (output > 0) std::printf("| %4d |  %17.10f  | % 5.1E | %5.1E |", iiter, L, deltaL, M.Gr.norm());
		if (!started) M.getHessian();
		else{
			const std::function<double (EigenMatrix, EigenMatrix)> Inner_current = M.getInner();
			const std::function<EigenMatrix (EigenMatrix)> Hr_last = Hr_last;
			const EigenMatrix tmp = Y_BFGS / M.Inner(Y_BFGS, S_BFGS);
			const EigenMatrix Hr_last_s = Hr_last(S_BFGS);
			M.Hr = [Inner_current, Hr_last, S_BFGS, Hr_last_s, Y_BFGS, tmp](EigenMatrix v){ // This needs to be modified for non-vector-transport-free cases.
				const EigenMatrix Hr_last_v = Hr_last(v);
				return Hr_last_v - Inner_current(S_BFGS, Hr_last_v) / Inner_current(S_BFGS, Hr_last_s) * Hr_last_s + Inner_current(Y_BFGS, v) * tmp;
			};
		}

		// Truncated conjugate gradient and rating the new step
		const std::tuple<double, double, double> loong_tol = {
			tol0/M.getDimension(),
			0.1*std::min(M.Inner(M.Gr,M.Gr),std::sqrt(M.Inner(M.Gr,M.Gr)))/M.getDimension(),
			0.1*tol2/M.getDimension()
		};
		const EigenMatrix S = Loong(M, R, loong_tol, output-1);

		const double S2 = M.Inner(S, S);
		const EigenMatrix Pnew = M.Exponential(S);
		double Lnew;
		EigenMatrix Genew;
		std::function<EigenMatrix (EigenMatrix)> Henew;
		std::tie(Lnew, Genew, Henew) = func(Pnew);
		const double top = Lnew - L;
		const double bottom = M.Inner(M.Gr + 0.5 * M.Hr(S), S);
		const double rho = top / bottom;

		// Determining whether to accept or reject the step
		if ( rho > rho_thres ){
			started = 1;
			M_last = M;
			deltaL = Lnew - L;
			L = Lnew;
			S_BFGS = M.TransportManifold(S, Pnew);
			Y_BFGS = - M.TransportManifold(M.Gr, Pnew);
			M.Update(Pnew, 1);
			M.Ge = Genew;
			M.He = Henew;
			if (output > 0) std::printf(" Accept |");
		}else if (output > 0) std::printf(" Reject |");
		if (output > 0) std::printf(" %5.1E | %6.3f |\n", std::sqrt(S2), __duration__(start, __now__));

		// Adjusting the trust radius according to the score
		if ( rho < 0.25 ) R *= 0.25;
		else if ( rho > 0.75 || std::abs(S2 - R * R) < 1.e-10 ) R = std::min(2 * R, R0);
		if ( M.Gr.norm() < tol1 ){
			if ( iiter == 0 ){
				if ( std::sqrt(S2) < tol2 ) return 1;
			}else{
				if ( std::abs(deltaL) < tol0 && std::sqrt(S2) < tol2 ) return 1;
			}
		}

	}

	return 0;
}

