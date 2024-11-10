#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <cassert>

#include "../Macro.h"

#include "GrassmannQLocal.h"

#include <iostream>


EigenMatrix LowdinOrthogonalization(EigenMatrix p){
	const EigenMatrix S = p.transpose() * p;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(S);
	const EigenVector values = eigensolver.eigenvalues().cwiseSqrt();
	const EigenMatrix vectors = eigensolver.eigenvectors();
	const EigenMatrix X = vectors * values.asDiagonal() * vectors.transpose();
	return X * p;
}

EigenMatrix OrthogonalComplement(EigenMatrix p){
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(p * p.transpose());
	const EigenMatrix vectors = eigensolver.eigenvectors();
	return vectors.leftCols(p.rows() - p.cols());
}

GrassmannQLocal::GrassmannQLocal(EigenMatrix p){
	this->Name = "Grassmann (Quotient) manifold with complement mapping";
	this->P.resize(p.rows(), p.cols());
	this->Ge.resize(p.rows(), p.cols());
	this->Gr.resize(p.rows() - p.cols(), p.cols()); // In local coordinates, the tangents are represented by (n-r)*r.
	this->P = LowdinOrthogonalization(p);
	this->Aux = OrthogonalComplement(this->P);
}

int GrassmannQLocal::getDimension(){
	return this->Gr.size();
}

double GrassmannQLocal::Inner(EigenMatrix X, EigenMatrix Y){
	return Dot(X, Y); // Complement mapping does not change the metric.
}

std::function<double (EigenMatrix, EigenMatrix)> GrassmannQLocal::getInner(){
	const std::function<double (EigenMatrix, EigenMatrix)> inner = [](EigenMatrix X, EigenMatrix Y){
		return Dot(X, Y);
	};
	return inner;
}


double GrassmannQLocal::Distance(EigenMatrix q){
	return this->Logarithm(q).norm();
}

EigenMatrix GrassmannQLocal::Exponential(EigenMatrix X){
	Eigen::BDCSVD<EigenMatrix> svd(this->Aux * X, Eigen::ComputeFullU | Eigen::ComputeFullV);
	const EigenMatrix U = svd.matrixU();
	const EigenMatrix V = svd.matrixV();
	EigenVector S = svd.singularValues();
	const EigenDiagonal cosS = S.array().cos().matrix().asDiagonal();
	const EigenDiagonal sinS = S.array().sin().matrix().asDiagonal();
	const EigenMatrix z = ( this->P * V * cosS + U * sinS ) * V.transpose();
	Eigen::HouseholderQR<EigenMatrix> qr(z);
	return qr.householderQ();
}

EigenMatrix GrassmannQLocal::Logarithm(EigenMatrix q){
	assert (0 && "Not implemented!");
	const EigenMatrix tmp = (q.transpose() * this->P).inverse() * ( q.transpose() - q.transpose() * this->P * this->P.transpose() );
	Eigen::BDCSVD<EigenMatrix> svd(tmp, Eigen::ComputeFullU | Eigen::ComputeFullV);
	EigenVector S = svd.singularValues();
	const EigenDiagonal atanS = S.array().atan().matrix().asDiagonal();
	const EigenMatrix U = svd.matrixU();
	const EigenMatrix V = svd.matrixV();
	return V * atanS * U.transpose();
}

EigenMatrix GrassmannQLocal::TangentProjection(EigenMatrix x){
	const EigenMatrix tmp = x - this->P * this->P.transpose() * x;
	return this->Aux.transpose() * tmp;
}

EigenMatrix GrassmannQLocal::TangentPurification(EigenMatrix X){
	// In local coordinates, there is no constraint on X.
	return X;
}

EigenMatrix GrassmannQLocal::TransportTangent(EigenMatrix X, EigenMatrix Y){
	// X - Vector on TpM to transport
	// Y - Destination on TpM
	assert(0 && "Not implemented!");
	Eigen::BDCSVD<EigenMatrix> svd(Y, Eigen::ComputeFullU | Eigen::ComputeFullV);
	EigenVector S = svd.singularValues();
	const EigenDiagonal sinS = S.array().sin().matrix().asDiagonal();
	const EigenDiagonal cosS = S.array().cos().matrix().asDiagonal();
	const EigenMatrix U = svd.matrixU();
	const EigenMatrix V = svd.matrixV();
	return ( ( - this->P * V * sinS + U * cosS ) * U.transpose() + EigenOne(U.rows(), U.rows()) - U * U.transpose() ) * X;
}

EigenMatrix GrassmannQLocal::TransportManifold(EigenMatrix X, EigenMatrix q){
	// X - Vector to transport from P
	// q - Destination on the manifold
	// In local coordinates, vector transport is not needed.
	q = q;
	return X;
}

void GrassmannQLocal::Update(EigenMatrix p, bool purify){
	this->P = purify ? LowdinOrthogonalization(p) : p;
	this->Aux = OrthogonalComplement(this->P);
}

void GrassmannQLocal::getGradient(){
	this->Gr = this->TangentProjection(this->Ge);
}

void GrassmannQLocal::getHessian(){
	const EigenMatrix P = this->P;
	const EigenMatrix Aux = this->Aux;
	const EigenMatrix Com = EigenOne(P.rows(), P.rows()) - P * P.transpose();
	const EigenMatrix PtGe = this->P.transpose() * this->Ge;
	const std::function<EigenMatrix (EigenMatrix)> He = this->He;

	this->Hr = [P, Com, Aux, PtGe, He](EigenMatrix v){
		const EigenMatrix x = He(v) - v * PtGe;
		const EigenMatrix tmp = Com * x;
		return Aux.transpose() * tmp;
		// this->TangentProjection(this->He(v) - v * PtGe)
	};
}
