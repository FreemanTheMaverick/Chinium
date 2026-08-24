#pragma once

#include <Eigen/Dense>
#include <vector>
#include <Maniverse/Manifold/Manifold.h>

namespace mv = Maniverse;

class ObjDeterminant: public mv::Objective{ public:
	Eigen::MatrixXd C0 = Eigen::MatrixXd::Zero(0, 0);
	Eigen::MatrixXd C = Eigen::MatrixXd::Zero(0, 0);
	Eigen::MatrixXd C0tC = Eigen::MatrixXd::Zero(0, 0);
	Eigen::MatrixXd C0tCinv = Eigen::MatrixXd::Zero(0, 0);
	int rank = 0;

	// Rank-deficient
	double beta = 0;

	// Rank-deficient 1
	Eigen::MatrixXd u0v0t = Eigen::MatrixXd::Zero(0, 0);

	// Rank-deficient 2
	Eigen::MatrixXd U0 = Eigen::MatrixXd::Zero(0, 0);
	Eigen::MatrixXd V0 = Eigen::MatrixXd::Zero(0, 0);

	ObjDeterminant(Eigen::MatrixXd C0) : C0(C0){
		C.resize(C0.rows(), C0.cols());
		C0tC.resize(C0.cols(), C0.cols());
		C0tCinv.resize(C0.cols(), C0.cols());
		u0v0t.resize(C0.rows(), C0.rows());
		U0.resize(C0.rows(), 2);
		V0.resize(C0.rows(), 2);
	};

	void Calculate(std::vector<Eigen::MatrixXd> C_, std::vector<int> derivatives) override{
		C = C_[0];
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			C0tC = C0.transpose() * C;
			Value = Eigen::HouseholderQR<Eigen::MatrixXd>(C0tC).determinant();
			if ( derivatives.size() == 1 ) return;
		}
		Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(C0tC);
		rank = svd.rank();
		const Eigen::VectorXd S = svd.singularValues().head(rank);
		C0tCinv = svd.matrixV().leftCols(rank) * S.cwiseInverse().asDiagonal() * svd.matrixU().leftCols(rank).transpose();
		if ( rank == C0.cols() ) Gradient = { Value * C0 * C0tCinv.transpose() };
		else{
			const double sing_prod = S.prod();
			const double detUV = ( svd.matrixU() * svd.matrixV() ).determinant();
			beta = sing_prod * detUV;
			if ( rank == C0.cols() - 1 ){
				const Eigen::VectorXd u0 = svd.matrixU().col(rank);
				const Eigen::VectorXd v0 = svd.matrixV().col(rank);
				u0v0t = u0 * v0.transpose();
				Gradient = { sing_prod * C0 * u0v0t };
			}else if ( rank == C0.cols() - 2 ){
				U0 = svd.matrixU().rightCols(2);
				V0 = svd.matrixV().rightCols(2);
				Gradient = { Eigen::MatrixXd::Zero(C0.rows(), C0.cols()) };
			}else Gradient = { Eigen::MatrixXd::Zero(C0.rows(), C0.cols()) };
		}
	};

	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> X_) const override{
		const Eigen::MatrixXd C0tX = C0.transpose() * X_[0];
		if ( rank == C0.cols() ) return std::vector<Eigen::MatrixXd>{ Value * C0 * (
				C0tCinv.transpose().cwiseProduct(C0tX).sum() * C0tCinv.transpose()
				- C0tCinv.transpose() * C0tX.transpose() * C0tCinv.transpose()
		) };
		if ( rank == C0.cols() - 1 ){
			return std::vector<Eigen::MatrixXd>{ beta * C0 * (
					0.5 * C0tCinv.transpose() * u0v0t.cwiseProduct(C0tX.transpose() )
					+ 0.5 * u0v0t * C0tCinv.cwiseProduct(C0tX)
					- C0tCinv.transpose() * C0tX.transpose() * u0v0t
			) };
		}
		if ( rank == C0.cols() - 2 ){
			Eigen::MatrixXd M = U0.transpose() * C0tX * V0;
			M(0, 1) *= -1;
			M(1, 0) *= -1;
			std::swap(M(0, 0), M(1, 1));
			return std::vector<Eigen::MatrixXd>{ 2 * beta * C0 * U0 * M.transpose() * V0.transpose() };
		}
		return { Eigen::MatrixXd::Zero(C0.rows(), C0.cols()) };
	};
};
