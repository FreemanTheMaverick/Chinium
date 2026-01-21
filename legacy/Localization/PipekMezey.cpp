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

EigenMatrix PipekMezey(std::vector<EigenMatrix> Qrefs, int output){
	const EigenMatrix kappa = Eigen::MatrixXd::Random(Qrefs[0].cols(), Qrefs[0].cols()) / 100000000 ;
	const EigenMatrix p = ( kappa - kappa.transpose() ).exp();
	Maniverse::Iterate M({Maniverse::Orthogonal(p).Clone()}, 1);

	std::function<
		std::tuple<
			double,
			std::vector<EigenMatrix>,
			std::vector<std::function<EigenMatrix (EigenMatrix)>>
		>(std::vector<EigenMatrix>, int)
	> func = [Qrefs](std::vector<EigenMatrix> Us, int order){
		const EigenMatrix U = Us[0];

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
			EigenMatrix hess1 = EigenZero(v.rows(), v.cols()).eval();
			EigenMatrix hess2 = EigenZero(v.rows(), v.cols()).eval();
			for ( int i = 0; i < (int)Qrefs.size(); i++ ){
				hess1 += Qrefs[i] * v * Qdiags[i];
				hess2 += QrefUs[i] * Diag(QrefUs[i].transpose() * v);
			}
			return (-4 * hess1 + -8 * hess2).eval();
		};

		return std::make_tuple(
				L,
				std::vector<EigenMatrix>{Ge},
				std::vector<std::function<EigenMatrix (EigenMatrix)>>{He}
		);
	};

	double L = 0;
	Maniverse::TrustRegionSetting tr_setting;
	if ( ! Maniverse::TrustRegion(
			func, tr_setting, {1.e-6, 1.e-4, 1.e-7},
			0.001, 1, 100, L, M, output
	) ) throw std::runtime_error("Convergence Failed!");

	return M.Point;
}
