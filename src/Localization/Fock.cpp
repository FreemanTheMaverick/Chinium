#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <functional>
#include <vector>
#include <tuple>
#include <cstdio>
#include <chrono>
#include <cassert>

#include "../Macro.h"
#include "../Optimization/Rabbit.h"

#include <iostream>

#define test(X) std::cout<<"X "<<X.rows()<<" "<<X.cols()<<std::endl;

EigenMatrix Fock(EigenMatrix Cref, EigenMatrix Fao, EigenMatrix F2ao, bool output){
	double L = 0;
	const EigenMatrix kappa = Eigen::MatrixXd::Random(Cref.cols(), Cref.cols()) / 1000 ;
	EigenMatrix U = ( kappa - kappa.transpose() ).exp();
	const EigenMatrix Fref = Cref.transpose() * Fao * Cref;
	const EigenMatrix F2ref = Cref.transpose() * F2ao * Cref;

	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<
				EigenMatrix
				(
					EigenMatrix,
					std::vector<EigenMatrix>
				)
			>
		>
		(
			EigenMatrix
		)
	> Fock_func = [=](EigenMatrix U_){
		const EigenMatrix F_ = U_.transpose() * Fref * U_;
		const EigenMatrix F2_ = U_.transpose() * F2ref * U_;

		const double L_ = F2_.trace() - F_.diagonal().norm() * F_.diagonal().norm();

		const EigenMatrix Ge_ = 2 * F2ref * U_ - 4 * Fref * U_ * (EigenMatrix)F_.diagonal().asDiagonal();

		std::function<
			EigenMatrix
			(
				EigenMatrix,
				std::vector<EigenMatrix>
			)
		> newton_update_ = [=](EigenMatrix D__, std::vector<EigenMatrix> params__){
			const EigenMatrix Gr__ = params__[0];
			const EigenMatrix Grp__ = params__[1];
			const EigenMatrix tmp1 = + 2 * Fref * D__ * Diag(F_);
			const EigenMatrix tmp2 = + 4 * Fref * U_ * Diag( U_.transpose() * Fref * D__ );
			const EigenMatrix tmp3 = + 1 * U_ * D__.transpose() * F2ref * U_;
			const EigenMatrix tmp4 = - 2 * U_ * Diag(F_) * D__.transpose() * Fref * U_;
			const EigenMatrix tmp5 = - 4 * U * Diag( U_.transpose() * Fref * D__ ) * F_;
			const EigenMatrix tmp6 = - 0.5 * U_ * D__.transpose() * Grp__;
			const EigenMatrix tmp7 = - Gr__;
			const EigenMatrix tmp8 = F2ref - 0.5 * U_ * Grp__.transpose();
			const EigenMatrix UTD = U_.transpose() * tmp8.inverse() * ( tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7 );
			const EigenMatrix result = U_ * ( UTD - UTD.transpose() ) / 2;
			return result;
		};

		return std::make_tuple(L_, Ge_, newton_update_);
	};

	assert( Rabbit(Fock_func, {1.e-6, 1.e-4, 1.e-3}, 1000, L, U, output) && "Convergence failed!" );
	return U;
}
