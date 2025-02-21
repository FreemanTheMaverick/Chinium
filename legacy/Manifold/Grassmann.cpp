#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <functional>
#include <tuple>
#include <deque>
#include <cstdio>
#include <chrono>
#include <cassert>

#include "../Macro.h"

#include "Grassmann.h"

#include <iostream>

Grassmann::Grassmann(EigenMatrix p){
	this->Name = "Grassmann";
	this->P.resize(p.rows(), p.cols());
	this->Ge.resize(p.rows(), p.cols());
	this->Gr.resize(p.rows(), p.cols());
	this->P = p;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(p);
	const EigenVector eigenvalues = eigensolver.eigenvalues();
	const EigenMatrix eigenvectors = eigensolver.eigenvectors();
	int rank = 0;
	for ( int i = 0; i < p.rows(); i++ )
		if ( eigenvalues(i) > 0.5 ) rank++;
	this->Aux.resize(p.rows(), rank);
	this->Aux = eigenvectors.rightCols(rank);
}

int Grassmann::getDimension(){
	const double rank = this->Aux.cols();
	return rank * ( this->P.rows() - rank );
}

double Grassmann::Inner(EigenMatrix X, EigenMatrix Y){
	return Dot(X, Y);
}

std::function<double (EigenMatrix, EigenMatrix)> Grassmann::getInner(){
	const std::function<double (EigenMatrix, EigenMatrix)> inner = [](EigenMatrix X, EigenMatrix Y){
		return Dot(X, Y);
	};
	return inner;
}


double Grassmann::Distance(EigenMatrix q){
	assert( 0 && "Geodesic length on Grassmann manifold is not implemented!" );
	return q.sum();
}

EigenMatrix Grassmann::Exponential(EigenMatrix X){
	const EigenMatrix Xp = X * this->P - this->P * X;
	const EigenMatrix pX = - Xp;
	const EigenMatrix expXp = Xp.exp();
	const EigenMatrix exppX = pX.exp();
	return expXp * this->P * exppX;
}

EigenMatrix Grassmann::Logarithm(EigenMatrix q){
	const EigenMatrix Omega = 0.5 * (
			( EigenOne(q.rows(), q.cols()) - 2 * q ) *
			( EigenOne(q.rows(), q.cols()) - 2 * this->P )
	).log();
	return Omega * this->P - this->P * Omega;
}

EigenMatrix Grassmann::TangentProjection(EigenMatrix X){
	// X must be symmetric.
	const EigenMatrix adPX = this->P * X - X * this->P;
	return this->P * adPX - adPX * this->P;
}

EigenMatrix Grassmann::TangentPurification(EigenMatrix A){
	const EigenMatrix symA = 0.5 * ( A + A.transpose() );
	const EigenMatrix pureA = symA - this->P * symA * this->P;
	return 0.5 * ( pureA + pureA.transpose() );
}

EigenMatrix Grassmann::TransportTangent(EigenMatrix X, EigenMatrix Y){
	// X - Vector to transport from P
	// Y - Destination on the tangent space of P
	const EigenMatrix dp = Y * this->P - this->P * Y;
	const EigenMatrix pd = - dp;
	const EigenMatrix expdp = dp.exp();
	const EigenMatrix exppd = pd.exp();
	return expdp * X * exppd;
}

EigenMatrix Grassmann::TransportManifold(EigenMatrix X, EigenMatrix q){
	// X - Vector to transport from P
	// q - Destination on the manifold
	const EigenMatrix Y = this->Logarithm(q);
	return this->TransportTangent(X, Y);
}

void Grassmann::Update(EigenMatrix p, bool purify){
	this->P = p;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(p);
	const EigenMatrix eigenvectors = eigensolver.eigenvectors();
	const int ncols = this->Aux.cols();
	this->Aux = eigenvectors.rightCols(ncols);
	if (purify) this->P = this->Aux * this->Aux.transpose();
}

void Grassmann::getGradient(){
	this->Gr = this->TangentProjection(this->Ge);
}

void Grassmann::getHessian(){
	const EigenMatrix P = this->P;
	const EigenMatrix Ge = this->Ge;
	const std::function<EigenMatrix (EigenMatrix)> He = this->He;
	this->Hr = [P, Ge, He](EigenMatrix v){
		const EigenMatrix he = He(v);
		const EigenMatrix partA = P * he - he * P;
		const EigenMatrix partB = Ge * v - v * Ge;
		const EigenMatrix sum = partA - partB;
		return EigenMatrix(P * sum - sum * P);
	};
}
