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


EigenMatrix PipekMezey(std::vector<EigenMatrix> Qrefs, int output){
	const EigenMatrix kappa = Eigen::MatrixXd::Random(Qrefs[0].cols(), Qrefs[0].cols()) / 100000000 ;
	const EigenMatrix p = ( kappa - kappa.transpose() ).exp();
	Orthogonal M = Orthogonal(p, 1);

	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		>(EigenMatrix, int)
	> func = [Qrefs](EigenMatrix U, int order){

		std::vector<EigenDiagonal> Qdiags(Qrefs.size());
		for ( int i = 0; i < (int)Qrefs.size(); i++ )
			Qdiags[i] = Diag(U.transpose() * Qrefs[i] * U);

		double L = 0;
		for ( int i = 0; i < (int)Qrefs.size(); i++ )
			L -= std::pow(Qdiags[i].diagonal().norm(), 2);

		EigenMatrix Ge = EigenZero(U.rows(), U.cols());
		for ( int i = 0; i < (int)Qrefs.size(); i++ )
			Ge += Qrefs[i] * U * Qdiags[i];
		Ge *= -4;

		std::vector<EigenMatrix> QrefUs(Qrefs.size());
		for ( int i = 0; i < (int)Qrefs.size(); i++ )
			QrefUs[i] = Qrefs[i] * U;
		std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix v){ return v; };
		if ( order == 2) He = [Qdiags, Qrefs, QrefUs](EigenMatrix v){
			EigenMatrix hess1 = EigenZero(v.rows(), v.cols());
			EigenMatrix hess2 = EigenZero(v.rows(), v.cols());
			for ( int i = 0; i < (int)Qrefs.size(); i++ ){
				hess1 += Qrefs[i] * v * Qdiags[i];
				hess2 += QrefUs[i] * Diag(QrefUs[i].transpose() * v);
			}
			return -4 * hess1 + -8 * hess2;
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
