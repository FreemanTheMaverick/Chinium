#include <Eigen/Core>
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

EigenMatrix Fock(EigenMatrix Fref, EigenMatrix F2ref, int output){
	double L = 0;
	const EigenMatrix kappa = Eigen::MatrixXd::Random(Fref.rows(), Fref.cols()) / 10 ;
	const EigenMatrix p = ( kappa - kappa.transpose() ).exp();
	Orthogonal M = Orthogonal(p);

	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		>(EigenMatrix)
	> func = [Fref, F2ref](EigenMatrix U){
		const EigenMatrix F = U.transpose() * Fref * U;
		const EigenMatrix F2 = U.transpose() * F2ref * U;
		const double L = F2.trace() - std::pow(F.diagonal().norm(), 2);

		const EigenDiagonal Fdiag = F.diagonal().asDiagonal();
		const EigenMatrix Ge = 2 * F2ref * U - 4 * Fref * U * Fdiag;

		const EigenMatrix twoF2ref = 2 * F2ref;
		const EigenMatrix fourFref = 4 * Fref;
		const EigenMatrix eightFrefU = 8 * Fref * U;
		const EigenMatrix UTFref = U.transpose() * Fref;
		const std::function<EigenMatrix (EigenMatrix)> He = [Fdiag, twoF2ref, fourFref, eightFrefU, UTFref](EigenMatrix v){
			return (EigenMatrix)(
					twoF2ref * v
					- fourFref * v * Fdiag
					- eightFrefU * (UTFref * v).diagonal().asDiagonal()
			);
		};

		return std::make_tuple(L, Ge, He);
	};

	assert( Ox( func, {1.e-6, 1.e-4, 1.e-7}, 100, L, M, output) && "Convergence failed!" );
	return M.P;
}
