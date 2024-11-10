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

EigenMatrix PipekMezey(std::vector<EigenMatrix> Qrefs, int output){
	double L = 0;
	const EigenMatrix kappa = Eigen::MatrixXd::Random(Qrefs[0].cols(), Qrefs[0].cols()) / 100000000 ;
	const EigenMatrix p = ( kappa - kappa.transpose() ).exp();
	Orthogonal M = Orthogonal(p);

	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		>(EigenMatrix)
	> func = [Qrefs](EigenMatrix U){

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
		const std::function<EigenMatrix (EigenMatrix)> He = [Qdiags, Qrefs, QrefUs](EigenMatrix v){
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

	assert( Ox( func, {1.e-6, 1.e-4, 1.e-7}, 100, L, M, output) && "Convergence failed!" );
	return M.P;
}
