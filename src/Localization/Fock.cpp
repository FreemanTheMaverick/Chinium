#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <functional>
#include <vector>
#include <tuple>
#include <cstdio>
#include <chrono>
#include <cassert>

#include "../Macro.h"
#include "../Manifold/Orthogonal.h"
#include "../Optimization/Ox.h"

#include <iostream>

#define test(X) std::cout<<"X "<<X.rows()<<" "<<X.cols()<<std::endl;

EigenMatrix Fock(EigenMatrix Cref, EigenMatrix Fao, EigenMatrix F2ao, int output){
	double L = 0;
	const EigenMatrix kappa = Eigen::MatrixXd::Random(Cref.cols(), Cref.cols()) / 10 ;
	const EigenMatrix p = ( kappa - kappa.transpose() ).exp();
	Orthogonal M = Orthogonal(p);
	const EigenMatrix Fref = Cref.transpose() * Fao * Cref;
	const EigenMatrix F2ref = Cref.transpose() * F2ao * Cref;

	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		>(EigenMatrix)
	> func = [Fref, F2ref](EigenMatrix U){
		const EigenMatrix F = U.transpose() * Fref * U;
		const EigenMatrix F2 = U.transpose() * F2ref * U;
		const double L = F2.trace() - F.diagonal().norm() * F.diagonal().norm();

		const EigenMatrix DiagF = Diag(F);
		const EigenMatrix Ge = 2 * F2ref * U - 4 * Fref * U * DiagF;

		const EigenMatrix twoF2ref = 2 * F2ref;
		const EigenMatrix fourFref = 4 * Fref;
		const EigenMatrix eightFrefU = 8 * Fref * U;
		const EigenMatrix UTFref = U.transpose() * Fref;
		const std::function<EigenMatrix (EigenMatrix)> He = [DiagF, twoF2ref, fourFref, eightFrefU, UTFref](EigenMatrix v){
			return (EigenMatrix)( twoF2ref * v - fourFref * v * DiagF - eightFrefU * Diag( UTFref * v) );
		};

		return std::make_tuple(L, Ge, He);
	};

	assert( Ox( func, {1.e-6, 1.e-4, 1.e-7}, 100, L, M, output) && "Convergence failed!" );
	return M.P;
}
