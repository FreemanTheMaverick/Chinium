#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <tuple>
#include <cassert>
#include <chrono>

#include "../Macro.h"
#include "../Multiwfn.h"
#include "../Optimization/Mouse.h"
#include "../Optimization/Tiger.h"

#include <iostream>


void Multiwfn::HartreeFockKohnSham(double temperature, double chemicalpotential, bool output, int nthreads){

	const EigenMatrix S = this->Overlap;
	const int nbasis = this->getNumBasis();
	const int nindbasis = this->getNumIndBasis();
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(S);
	const EigenVector sval = eigensolver.eigenvalues();
	const EigenMatrix tmp = sval.array().sqrt().inverse();
	const EigenMatrix sinvsqrt = tmp.asDiagonal();
	const EigenMatrix svec = eigensolver.eigenvectors();
	const EigenMatrix Z = svec * sinvsqrt * svec.transpose();

	EigenMatrix Dprime = Z.transpose().inverse() * this->getDensity() / 2 * Z.inverse();
	std::function<std::tuple<double, EigenMatrix> (EigenMatrix)> cfunc = [&](EigenMatrix Dprime_){
		EigenMatrix D_ = Z.transpose() * Dprime_ * Z;
		EigenMatrix F_ = this->calcFock(D_, nthreads);
		EigenMatrix Fprime_ = Z * F_ * Z.transpose(); // Euclidean gradient
		eigensolver.compute(Fprime_); // Fprime
		this->setEnergy(eigensolver.eigenvalues()); // Orbital energy
		this->E_tot = ( D_ * ( this->Kinetic + this->Nuclear + F_ ) ).trace();
		return std::make_tuple(this->E_tot, Fprime_);
	};

	EigenMatrix F = this->getFock();
	std::function<std::tuple<double, EigenMatrix, EigenMatrix, EigenMatrix> (EigenMatrix)> ffunc = [&](EigenMatrix Fupdate){
		const EigenMatrix Fprime_ = Z * Fupdate * Z.transpose();
		eigensolver.compute(Fprime_);
		this->setEnergy(eigensolver.eigenvalues());
		this->setCoefficientMatrix(Z * eigensolver.eigenvectors());
		const EigenMatrix D_ = this->getDensity() / 2.;
		const EigenMatrix F_ = this->calcFock(D_, nthreads);
		const EigenMatrix G = F_ * D_ * S - S * D_ * F_;
		this->E_tot = (D_ * ( this->Kinetic + this->Nuclear + F_ )).trace();
		return std::make_tuple(this->E_tot, F_, G, D_);
	};

	//assert( Tiger( cfunc, {1.e-8, 1.e-5, 1.e-5}, 20, 100, this->E_tot, Dprime, output) && "Convergence failed!" );
	assert( Mouse( ffunc, {1.e3, 0.000005, 1.e3}, {1.e-8, 1.e-5, 1.e-5}, 20, 100, this->E_tot, F, output) && "Convergence failed!" );
}
