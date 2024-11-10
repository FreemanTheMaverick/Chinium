#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <tuple>
#include <cassert>
#include <chrono>
#include <cstdio>

#include "../Macro.h"
#include "../Multiwfn.h"
#include "../Optimization/Mouse.h"
#include "../Manifold/Grassmann.h"
#include "../Optimization/Ox.h"
#include "FockFormation.h"

#include <iostream>


void Multiwfn::HartreeFockKohnSham(double temperature, double chemicalpotential, int output, int nthreads){

	if (output > 0) std::printf("Self-consistent field\n");

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

	const EigenMatrix Hcore = this->Kinetic + this->Nuclear;
	short int* is = this->RepulsionIs;
	short int* js = this->RepulsionJs;
	short int* ks = this->RepulsionKs;
	short int* ls = this->RepulsionLs;
	char* degs = this->RepulsionDegs;
	double* ints = this->Repulsions;
	long int length = this->RepulsionLength;

	Grassmann Dprime = Grassmann(Z.inverse() * this->getDensity() / 2 * Z.inverse());
	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		> (EigenMatrix)
	> dfunc = [Z, Hcore, is, js, ks, ls, degs, ints, length, nthreads](EigenMatrix Dprime_){
		const EigenMatrix D_ = Z * Dprime_ * Z;
		const EigenMatrix F_ = Hcore + Ghf(is, js, ks, ls, degs, ints, length, D_, 1, nthreads);
		const EigenMatrix Fprime_ = Z * F_ * Z; // Euclidean gradient
		const double E_ = ( D_ * ( Hcore + F_ ) ).trace();
		const std::function<EigenMatrix (EigenMatrix)> He = [Z, is, js, ks, ls, degs, ints, length, nthreads](EigenMatrix vprime){
			const EigenMatrix v = Z * vprime * Z;
			return EigenMatrix(Z * Ghf(is, js, ks, ls, degs, ints, length, v, 1, nthreads) * Z);
		};
		return std::make_tuple(E_, Fprime_, He);
	};

	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		> (EigenMatrix)
	> cfunc = [Z, Hcore, is, js, ks, ls, degs, ints, length, nthreads](EigenMatrix Cprime_){
		const EigenMatrix Dprime_ = Cprime_ * Cprime_.transpose();
		const EigenMatrix C_ = Z * Cprime_;
		const EigenMatrix D_ = Z.transpose() * Dprime_ * Z;
		const EigenMatrix F_ = Hcore + Ghf(is, js, ks, ls, degs, ints, length, D_, 1, nthreads);
		const EigenMatrix Fprime_ = Z * F_ * Z.transpose(); // Euclidean gradient
		const double E_ = ( D_ * ( Hcore + F_ ) ).trace();
		const std::function<EigenMatrix (EigenMatrix)> He = [Cprime_, Z, is, js, ks, ls, degs, ints, length, nthreads, F_](EigenMatrix v){
			const EigenMatrix tmp1 = Z.transpose() * Cprime_ * v.transpose() * Z;
			const EigenMatrix tmp2 = Z * Ghf(is, js, ks, ls, degs, ints, length, tmp1, 1, nthreads) * Z.transpose() * Cprime_;
			return 4 * tmp2 + 2 * F_ * v;
		};
		return std::make_tuple(E_, Fprime_ * Cprime_, He);
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

	double E = 0;
	assert( Mouse( ffunc, {1.e3, 0.5, 1.e3}, {1.e-8, 1.e-5, 1.e-5}, 20, 100, this->E_tot, F, output-1) && "Convergence failed!" );
	assert( Ox( dfunc, {1.e-8, 1.e-7, 1.e-7}, 100, E, Dprime, output-1) && "Convergence failed!" );
	if (output > 0) std::printf("Converged!\n");
}
