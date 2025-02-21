#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <tuple>
#include <deque>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <Maniverse/Manifold/Grassmann.h>
#include <Maniverse/Optimizer/TrustRegion.h>

#include "../Macro.h"
#include "../Multiwfn.h"
#include "../DIIS.h"
#include "FockFormation.h"

#include <iostream>


bool DIISSCF(
		std::function<std::tuple<double, EigenMatrix, EigenMatrix, EigenMatrix> (EigenMatrix)>& ffunc,
		std::tuple<double, double, double> adtol,
		std::tuple<double, double, double> tol,
		int diis_space, int max_iter,
		double& E, EigenMatrix& F, int output){
	if (output > 0){
		std::printf("Using DIIS SCF optimizer\n");
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", std::get<0>(tol));
		std::printf("| Gradient norm (Grad.)               : %E\n", std::get<1>(tol));
		std::printf("| Independent variable update (V. U.) : %E\n", std::get<2>(tol));
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  | Update |  V. U.  |  Time  |\n");
	}

	EigenMatrix Fupdate = F;
	EigenMatrix G = EigenZero(F.cols(), F.cols());
	EigenMatrix D = EigenZero(F.cols(), F.cols());
	std::deque<double> Es = {};
	std::deque<EigenMatrix> Fs = {};
	std::deque<EigenMatrix> Gs = {};
	std::deque<EigenMatrix> Ds = {};
	const auto start = __now__;
	
	for ( int iiter = 0; iiter < max_iter; iiter++ ){
		if (output > 0) std::printf("| %4d |", iiter);

		std::tie(E, F, G, D) = ffunc(F);
		const double deltaE = ( iiter == 0 ) ? E : ( E - Es.back() );

		if (output > 0) std::printf("  %17.10f  | % 5.1E | %5.1E |", E, deltaE, G.norm());

		Es.push_back(E);
		Fs.push_back(F);
		Gs.push_back(G);
		Ds.push_back(D);
		if ( (int)Es.size() == diis_space ){
			Es.pop_front();
			Fs.pop_front();
			Gs.pop_front();
			Ds.pop_front();
		}

		if ( Es.size() < 2 ){
			if (output > 0) std::printf("  Naive |");
		}else if ( G.norm() > std::get<1>(adtol) || iiter < 3 ){
			if (output > 0) std::printf("  ADIIS |");
			EigenMatrix AD1s = EigenZero(Ds.size(), 1);
			EigenMatrix AD2s = EigenZero(Ds.size(), Ds.size());
			for ( int i = 0; i < (int)Ds.size(); i++ ){
				AD1s(i) = ( ( Ds[i] - D ) * F.transpose() ).trace();
				for ( int j = 0; j < (int)Ds.size(); j++ )
					AD2s(i, j) = ( ( Ds[i] - D ) * ( Fs[j] - F ).transpose() ).trace();
			}
			F = ADIIS(AD1s, AD2s, Fs, output-1);
		}else{
			if (output > 0) std::printf("  CDIIS |");
			F = CDIIS(Gs, Fs);
		}

		const double deltaF = ( F - Fs.back() ).norm();
		if (output > 0) std::printf(" %5.1E | %6.3f |\n", deltaF, __duration__(start, __now__));

		if ( G.norm() < std::get<2>(tol) ){
			if ( iiter == 0 ) return 1;
			if ( std::abs(deltaE) < std::get<0>(tol) && deltaF < std::get<1>(tol) ) return 1;
		}

	}
	return 0;
}

void Multiwfn::HartreeFockKohnSham(int output, int nthreads){
	Eigen::setNbThreads(nthreads);

	const double T = this->Temperature;
	const double Mu = this->ChemicalPotential;
	if (output > 0) std::printf("Self-consistent field in %s-canonical ensemble\n", T > 0 ? "grand" : "micro");

	const EigenMatrix S = this->Overlap;
	const EigenMatrix Z = this->getCoefficientMatrix();

	const EigenMatrix Hcore = this->Kinetic + this->Nuclear;
	short int* is = this->RepulsionIs;
	short int* js = this->RepulsionJs;
	short int* ks = this->RepulsionKs;
	short int* ls = this->RepulsionLs;
	char* degs = this->RepulsionDegs;
	double* ints = this->Repulsions;
	long int length = this->RepulsionLength;
	double E = 0;

	if ( T == 0){
		EigenMatrix Dprime = (this->getOccupation() / 2).asDiagonal();
		Grassmann M = Grassmann(Dprime, 1);
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<EigenMatrix (EigenMatrix)>
			> (EigenMatrix, int)
		> dfunc = [Z, Hcore, is, js, ks, ls, degs, ints, length, nthreads](EigenMatrix Dprime_, int order){
			const EigenMatrix D_ = Z * Dprime_ * Z.transpose();
			const EigenMatrix F_ = Hcore + Ghf(is, js, ks, ls, degs, ints, length, D_, 1, nthreads);
			const EigenMatrix Fprime_ = Z.transpose() * F_ * Z; // Euclidean gradient
			const double E_ = ( D_ * ( Hcore + F_ ) ).trace();
			std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix vprime){ return vprime; };
			if ( order == 2 ) He = [Z, is, js, ks, ls, degs, ints, length, nthreads](EigenMatrix vprime){
				const EigenMatrix v = Z * vprime * Z.transpose();
				return (EigenMatrix)(Z.transpose() * Ghf(is, js, ks, ls, degs, ints, length, v, 1, nthreads) * Z);
			};
			return std::make_tuple(E_, Fprime_, He);
		};

		TrustRegionSetting tr_setting;
		assert(
				TrustRegion(
					dfunc, tr_setting, {1.e-8, 1.e-7, 1.e-7},
					1, 100, E, M, output-1
				) && "Convergence failed!"
		);
	}

	else{
		EigenMatrix F = this->getFock();
		Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
		std::function<std::tuple<double, EigenMatrix, EigenMatrix, EigenMatrix> (EigenMatrix)> ffunc = [&](EigenMatrix Fupdate){
			const EigenMatrix Fprime_ = Z.transpose() * Fupdate * Z;
			eigensolver.compute(Fprime_);
			this->setEnergy(eigensolver.eigenvalues());
			this->setCoefficientMatrix(Z * eigensolver.eigenvectors());
			double E_ = 0;
			if ( T ){
				const EigenArray es = eigensolver.eigenvalues();
				const EigenArray ns = 1. / ( 1. + ( ( es - Mu ) / T ).exp() );
				this->setOccupation( 2 * (EigenVector)ns.matrix());
				E_ += 2 * (
						T * (
							ns.pow(ns).log() + ( 1. - ns ).pow( 1. - ns ).log()
						).sum()
						- Mu * ns.sum()
				);
			}
			const EigenMatrix D_ = this->getDensity() / 2.;
			EigenMatrix Fhf_ = EigenZero(Fupdate.rows(), Fupdate.cols());
			EigenMatrix Fxc_ = EigenZero(Fupdate.rows(), Fupdate.cols());
			double Exc_ = 0;
			std::tie(Fhf_, Fxc_, Exc_) = this->calcFock(D_, nthreads);
			const EigenMatrix F_ = Fhf_ + Fxc_;
			const EigenMatrix G = F_ * D_ * S - S * D_ * F_;
			E_ += (D_ * ( this->Kinetic + this->Nuclear + Fhf_ )).trace() + Exc_;
			return std::make_tuple(E_, F_, G, D_);
		};

		assert( DIISSCF( ffunc, {1.e3, 0.15, 1.e3}, {1.e-8, 1.e-5, 1.e-5}, 20, 100, E, F, output-1) && "Convergence failed!" );
	}

	this->E_tot += E;


	/*
	EigenMatrix Call = this->getCoefficientMatrix();
	GrassmannQLocal Cprime = GrassmannQLocal(Z.inverse() * Call.leftCols(nocc));
	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		> (EigenMatrix)
	> cfunc = [Z, Hcore, is, js, ks, ls, degs, ints, length, nthreads](EigenMatrix Cprime_){
		const EigenMatrix Dprime_ = Cprime_ * Cprime_.transpose();
		const EigenMatrix C_ = Z * Cprime_;
		const EigenMatrix D_ = C_ * C_.transpose();
		const EigenMatrix F_ = Hcore + Ghf(is, js, ks, ls, degs, ints, length, D_, 1, nthreads);
		const EigenMatrix Fprime_ = Z * F_ * Z; // Euclidean gradient
		const double E_ = ( D_ * ( Hcore + F_ ) ).trace();
		const EigenMatrix ZCprime_ = Z * Cprime_;
		const std::function<EigenMatrix (EigenMatrix)> He = [Z, ZCprime_, is, js, ks, ls, degs, ints, length, nthreads, Fprime_](EigenMatrix v){
std::cout<<11<<std::endl;
			const EigenMatrix tmp1 = ZCprime_ * v.transpose() * Z;
std::cout<<22<<std::endl;
			const EigenMatrix tmp2 = Z * Ghf(is, js, ks, ls, degs, ints, length, tmp1, 1, nthreads) * ZCprime_;
std::cout<<33<<std::endl;
			return 4 * tmp2 + 2 * Fprime_ * v;
		};
		return std::make_tuple(E_, Fprime_ * Cprime_, He);
	};
	*/

}
