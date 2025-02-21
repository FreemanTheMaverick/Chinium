#include <Eigen/Dense>
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


EigenMatrix FosterBoys(
		EigenMatrix Cref,
		EigenMatrix Wxao, EigenMatrix Wyao, EigenMatrix Wzao,
		EigenMatrix W2aoSum, int output){
	const EigenMatrix kappa = Eigen::MatrixXd::Random(Cref.cols(), Cref.cols()) / 10 ;
	const EigenMatrix p = ( kappa - kappa.transpose() ).exp();
	Orthogonal M = Orthogonal(p, 1);
	const EigenMatrix Wxref = Cref.transpose() * Wxao * Cref;
	const EigenMatrix Wyref = Cref.transpose() * Wyao * Cref;
	const EigenMatrix Wzref = Cref.transpose() * Wzao * Cref;
	const EigenMatrix W2refSum = Cref.transpose() * W2aoSum * Cref;

	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		>(EigenMatrix, int)
	> func = [Wxref, Wyref, Wzref, W2refSum](EigenMatrix U, int order){
		const EigenMatrix Wx = U.transpose() * Wxref * U;
		const EigenMatrix Wy = U.transpose() * Wyref * U;
		const EigenMatrix Wz = U.transpose() * Wzref * U;
		const EigenMatrix DiagWx = Diag(Wx);
		const EigenMatrix DiagWy = Diag(Wy);
		const EigenMatrix DiagWz = Diag(Wz);
		const EigenMatrix W2Sum = U.transpose() * W2refSum * U;
		const double L = W2Sum.trace() - std::pow(DiagWx.norm(), 2) - std::pow(DiagWy.norm(), 2) - std::pow(DiagWz.norm(), 2);

		const EigenMatrix Ge = 2 * W2refSum * U - 4 * (
				Wxref * U * DiagWx +
				Wyref * U * DiagWy +
				Wzref * U * DiagWz
		);

		const EigenMatrix twoW2refSum = 2 * W2refSum;
		const EigenMatrix fourWxref = 4 * Wxref;
		const EigenMatrix fourWyref = 4 * Wyref;
		const EigenMatrix fourWzref = 4 * Wzref;
		const EigenMatrix eightWxrefU = 8 * Wxref * U;
		const EigenMatrix eightWyrefU = 8 * Wyref * U;
		const EigenMatrix eightWzrefU = 8 * Wzref * U;
		const EigenMatrix UTWxref = U.transpose() * Wxref;
		const EigenMatrix UTWyref = U.transpose() * Wyref;
		const EigenMatrix UTWzref = U.transpose() * Wzref;
		std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix v){ return v; };
		if ( order == 2) He = [
			DiagWx, DiagWy, DiagWz,
			twoW2refSum,
			fourWxref, fourWyref, fourWzref,
			eightWxrefU, eightWyrefU, eightWzrefU,
			UTWxref, UTWyref, UTWzref
		](EigenMatrix v){
			return (EigenMatrix)(
				twoW2refSum * v
				- fourWxref * v * DiagWx
				- fourWyref * v * DiagWy
				- fourWzref * v * DiagWz
				- eightWxrefU * Diag( UTWxref * v)
				- eightWyrefU * Diag( UTWyref * v)
				- eightWzrefU * Diag( UTWzref * v) );
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
