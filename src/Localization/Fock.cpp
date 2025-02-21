#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <functional>
#include <vector>
#include <tuple>
#include <cstdio>
#include <chrono>
#include <cassert>
#include <memory>
#include <Maniverse/Manifold/Orthogonal.h>
#include <Maniverse/Optimizer/TrustRegion.h>

#include "../Macro.h"

#include <iostream>


EigenMatrix Fock(EigenMatrix Fref, EigenMatrix F2ref, int output){
	const EigenMatrix kappa = Eigen::MatrixXd::Random(Fref.rows(), Fref.cols()) / 10 ;
	const EigenMatrix p = ( kappa - kappa.transpose() ).exp();
	Orthogonal M = Orthogonal(p, 1);

	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		>(EigenMatrix, int)
	> func = [Fref, F2ref](EigenMatrix U, int order){
		const EigenMatrix F = U.transpose() * Fref * U;
		const EigenMatrix F2 = U.transpose() * F2ref * U;
		const double L = F2.trace() - std::pow(F.diagonal().norm(), 2);

		const EigenDiagonal Fdiag = F.diagonal().asDiagonal();
		const EigenMatrix Ge = 2 * F2ref * U - 4 * Fref * U * Fdiag;

		const EigenMatrix twoF2ref = 2 * F2ref;
		const EigenMatrix fourFref = 4 * Fref;
		const EigenMatrix eightFrefU = 8 * Fref * U;
		const EigenMatrix UTFref = U.transpose() * Fref;
		std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix v){ return v; };
		if ( order == 2 ) He = [Fdiag, twoF2ref, fourFref, eightFrefU, UTFref](EigenMatrix v){
			return (EigenMatrix)(
					twoF2ref * v
					- fourFref * v * Fdiag
					- eightFrefU * (UTFref * v).diagonal().asDiagonal()
			);
		};

		return std::make_tuple(L, Ge, He);
	};

	double L = 0;
	TrustRegionSetting tr_setting;
	assert(
			TrustRegion(
				func, tr_setting, {1.e-6, 1.e-4, 1.e-7},
				1, 100, L, M, output
			) && "Convergence Failed!"
	);

	return M.P;
}
