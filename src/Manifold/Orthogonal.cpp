#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <functional>
#include <cassert>

#include "../Macro.h"

#include "Orthogonal.h"

#include <iostream>

Orthogonal::Orthogonal(EigenMatrix p){
	this->Name = "Orthogonal";
	this->P.resize(p.rows(), p.cols());
	this->Ge.resize(p.rows(), p.cols());
	this->Gr.resize(p.rows(), p.cols());
	this->P = p;
}

int Orthogonal::getDimension(){
	return this->P.cols() * (this->P.cols() - 1) / 2;
}

double Orthogonal::Inner(EigenMatrix X, EigenMatrix Y){
	return 0.5 * Dot(X, Y);
}

double Orthogonal::Distance(EigenMatrix q){
	assert( 0 && "Geodesic length on Orthogonal manifold is not implemented!" );
	return q.sum() * 0;
}

EigenMatrix Orthogonal::Exponential(EigenMatrix X){
	return (X * this->P.transpose()).exp() * this->P;
}

EigenMatrix Orthogonal::Logarithm(EigenMatrix q){
	return ( this->P.transpose() * q ).log();
}

EigenMatrix Orthogonal::TangentProjection(EigenMatrix A){
	return 0.5 * ( A - this->P * A.transpose() * this->P );
}

EigenMatrix Orthogonal::TangentPurification(EigenMatrix A){
	const EigenMatrix Z = this->P.transpose() * A;
	const EigenMatrix Zpurified = 0.5  * (Z - Z.transpose());
	return this->P * Zpurified;
}

void Orthogonal::ManifoldPurification(){
	Eigen::BDCSVD<EigenMatrix> svd(this->P, Eigen::ComputeFullU | Eigen::ComputeFullV);
	this->P = svd.matrixU() * svd.matrixV().transpose();
}

void Orthogonal::getGradient(){
	this->Gr = this->TangentPurification(this->TangentProjection(this->Ge));
}

void Orthogonal::getHessian(){
	const EigenMatrix P = this->P;
	const EigenMatrix Grp = this->Ge - this->Gr;
	const EigenMatrix PGrpT = this->P * Grp.transpose();
	const std::function<EigenMatrix (EigenMatrix)> He = this->He;
	this->Hr = [P, Grp, PGrpT, He](EigenMatrix v){
		const EigenMatrix Hev = He(v);
		const EigenMatrix TprojHev = 0.5 * ( Hev - P * Hev.transpose() * P );
		return (EigenMatrix)( TprojHev - 0.5 * ( PGrpT * v - P * v.transpose() * Grp ) );
	};
}
