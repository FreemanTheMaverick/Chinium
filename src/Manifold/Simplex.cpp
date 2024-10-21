#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <cassert>

#include "../Macro.h"

#include "Simplex.h"

#include <iostream>

Simplex::Simplex(EigenMatrix p){
	this->Name = "Simplex";
	this->P.resize(p.rows(), p.cols());
	this->Ge.resize(p.rows(), p.cols());
	this->Gr.resize(p.rows(), p.cols());
	this->P = p;
}

int Simplex::getDimension(){
	return this->P.size() - 1;
}

double Simplex::Inner(EigenMatrix X, EigenMatrix Y){
	return this->P.cwiseInverse().cwiseProduct(X.cwiseProduct(Y)).sum();
}

double Simplex::Distance(EigenMatrix q){
	return 2 * std::acos( this->P.cwiseProduct(q).cwiseSqrt().sum() );
}

EigenMatrix Simplex::Exponential(EigenMatrix X){
	const EigenMatrix Xp = X.cwiseProduct(this->P.array().rsqrt().matrix());
	const double norm = Xp.norm();
	const EigenMatrix Xpn = Xp / norm;
	const EigenMatrix tmp1 = 0.5 * (this->P + Xpn.cwiseProduct(Xpn));
	const EigenMatrix tmp2 = 0.5 * (this->P - Xpn.cwiseProduct(Xpn)) * std::cos(norm);
	const EigenMatrix tmp3 = Xpn.cwiseProduct(this->P.cwiseSqrt()) * std::sin(norm);
	return tmp1 + tmp2 + tmp3;
}

EigenMatrix Simplex::Logarithm(EigenMatrix q){
	const double dot = Dot( this->P.cwiseSqrt(), q.cwiseSqrt() );
	const double tmp1 = this->Distance(q);
	const double tmp2 = 1. - dot;
	const EigenMatrix tmp3 = this->P.cwiseProduct(q).cwiseSqrt();
	const EigenMatrix tmp4 = dot * this->P;
	return tmp1 / tmp2 * ( tmp3 - tmp4 );
}

EigenMatrix Simplex::TangentProjection(EigenMatrix A){
	const int n = this->P.size();
	const EigenMatrix ones = EigenZero(n, n).array() + 1;
	EigenMatrix tmp = EigenZero(n, n);
	for ( int i = 0; i < n; i++ ) tmp.col(i) = P;
	return ( EigenOne(n, n) - tmp ) * A;
}

EigenMatrix Simplex::TangentPurification(EigenMatrix A){
	return A.array() - A.mean();
}

void Simplex::ManifoldPurification(){
	const EigenMatrix Pabs = this->P.cwiseAbs();
	this->P = Pabs / Pabs.sum();
}

void Simplex::getGradient(){
	this->Gr = this->TangentProjection(this->P.cwiseProduct(this->Ge));
}

void Simplex::getHessian(){
	const int n = this->P.size();
	const EigenMatrix ones = EigenZero(n, n).array() + 1;
	const EigenMatrix proj = this->TangentProjection(EigenOne(n, n));
	const EigenMatrix M = proj * (EigenMatrix)this->P.asDiagonal();
	const EigenMatrix N = proj * (EigenMatrix)(
			this->Ge
			- ones * this->Ge.cwiseProduct(this->P)
			- 0.5 * this->Gr.cwiseProduct(this->P.cwiseInverse())
	).asDiagonal();
	const std::function<EigenMatrix (EigenMatrix)> He = this->He;
	this->Hr = [He, M, N](EigenMatrix v){
		return (EigenMatrix)(M * He(v) + N * v); // The forced conversion "(EigenMatrix)" is necessary. Without it the result will be wrong. I do not know why. Then I forced convert every EigenMatrix return value in std::function for ensurance.
	};
}
