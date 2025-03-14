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

std::tuple<double, EigenVector, EigenMatrix> RestrictedNewton(
		EigenMatrix Dprime, EigenMatrix Z, EigenMatrix Hcore,
		short int* is, short int* js, short int* ks, short int* ls,
		double* ints, long int length, int output, int nthreads){
	double E = 0;
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	Grassmann M = Grassmann(Dprime, 1);
	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		> (EigenMatrix, int)
	> dfunc_newton = [&epsilons, &C, &eigensolver, Z, Hcore, is, js, ks, ls, ints, length, nthreads](EigenMatrix Dprime_, int order){
		const EigenMatrix D_ = Z * Dprime_ * Z.transpose();
		const EigenMatrix F_ = Hcore + Grhf(is, js, ks, ls, ints, length, D_, 1, nthreads);
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z; // Euclidean gradient
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		const double E_ = 0.5 * ( D_ * ( Hcore + F_ ) ).trace();
		std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix vprime){ return vprime; };
		if ( order == 2 ) He = [Z, is, js, ks, ls, ints, length, nthreads](EigenMatrix vprime){
			const EigenMatrix v = Z * vprime * Z.transpose();
			return (EigenMatrix)(Z.transpose() * Grhf(is, js, ks, ls, ints, length, v, 1, nthreads) * Z);
		};
		return std::make_tuple(E_, Fprime_, He);
	};
	TrustRegionSetting tr_setting;
	assert(
			TrustRegion(
				dfunc_newton, tr_setting, {1.e-8, 1.e-5, 1.e-5},
				0.00001, 1, 100, E, M, output
			) && "Convergence failed!"
	);
	return std::make_tuple(E, epsilons, C);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedQuasiNewton(
		EigenMatrix Dprime, EigenMatrix Z, EigenMatrix Hcore,
		short int* is, short int* js, short int* ks, short int* ls,
		double* ints, long int length,
		ExchangeCorrelation& xc,
		double* ws, long int ngrids, int nbasis,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		double* rhos,
		double* rho1xs, double* rho1ys, double* rho1zs, double* sigmas,
		double* lapls, double* taus,
		double* es,
		double* erhos, double* esigmas,
		double* elapls, double* etaus,
		int output, int nthreads){
	double E = 0;
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	Grassmann M = Grassmann(Dprime, 1);
	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		> (EigenMatrix, int)
	> dfunc_quasi = [&](EigenMatrix Dprime_, int order){
		const EigenMatrix D_ = Z * Dprime_ * Z.transpose();
		const EigenMatrix Ghf_ = Grhf(is, js, ks, ls, ints, length, D_, xc.EXX, nthreads);
		auto [Exc_, Gxc_] = xc.XCcode ? Gxc( // double, EigenMatrix
				xc,
				ws, ngrids, nbasis,
				aos,
				ao1xs, ao1ys, ao1zs,
				ao2ls,
				{D_},
				rhos,
				rho1xs, rho1ys, rho1zs, sigmas,
				lapls, taus,
				es,
				erhos, esigmas,
				elapls, etaus,
				nthreads
		) : std::make_tuple(0, EigenZero(nbasis, nbasis));
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix F_ = Fhf_+ Gxc_;
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z; // Euclidean gradient
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		const double E_ = 0.5 * (( D_ * ( Hcore + Fhf_ ) ).trace() + Exc_);
		std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix vprime){ return vprime; };
		if ( order == 2 ) He = [](EigenMatrix vprime){
			return EigenZero(vprime.rows(), vprime.cols());
		};
		return std::make_tuple(E_, Fprime_, He);
	};
	TrustRegionSetting tr_setting;
	if ( xc.XCcode ) tr_setting.R0 = 0.5;
	assert(
			TrustRegion(
				dfunc_quasi, tr_setting, {1.e-8, 1.e-5, 1.e-5},
				0.005, 20, 100, E, M, output
			) && "Convergence failed!"
	);
	return std::make_tuple(E, epsilons, C);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedDIIS(
		double T, double Mu, EigenVector Occ,
		EigenMatrix F, EigenMatrix S, EigenMatrix Z, EigenMatrix Hcore,
		short int* is, short int* js, short int* ks, short int* ls,
		double* ints, long int length,
		ExchangeCorrelation& xc,
		double* ws, long int ngrids, int nbasis,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		double* rhos,
		double* rho1xs, double* rho1ys, double* rho1zs, double* sigmas,
		double* lapls, double* taus,
		double* es,
		double* erhos, double* esigmas,
		double* elapls, double* etaus,
		int output, int nthreads){
	double E = 0;
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenVector occupations = Occ;
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	std::function<
			std::vector<std::tuple<double, EigenMatrix, EigenMatrix>>
			(std::vector<EigenMatrix>&)
	> RawUpdate = [&](std::vector<EigenMatrix>& Fs){
		assert(Fs.size() == 1 && "Only one Fock matrix should be optimized in spin-restricted SCF!");
		const EigenMatrix F_ = Fs[0];
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z;
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		double E_ = 0;
		if ( T ){
			const EigenArray ns = 1. / ( 1. + ( ( epsilons.array() - Mu ) / T ).exp() );
			occupations = ns.matrix();
			E_ += 2 * (
					T * (
						ns.pow(ns).log() + ( 1. - ns ).pow( 1. - ns ).log()
					).sum()
					- Mu * ns.sum()
			);
		}
		const EigenMatrix D_ = C * occupations.asDiagonal() * C.transpose();
		const EigenMatrix Ghf_ = Grhf(is, js, ks, ls, ints, length, D_, xc.EXX, nthreads);
		auto [Exc_, Gxc_] = xc.XCcode ? Gxc( // double, EigenMatrix
				xc,
				ws, ngrids, nbasis,
				aos,
				ao1xs, ao1ys, ao1zs,
				ao2ls,
				{2 * D_},
				rhos,
				rho1xs, rho1ys, rho1zs, sigmas,
				lapls, taus,
				es,
				erhos, esigmas,
				elapls, etaus,
				nthreads
		) : std::make_tuple(0, EigenZero(nbasis, nbasis));
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix Fnew_ = Fhf_ + Gxc_;
		E_ += (D_ * ( Hcore + Fhf_ )).trace() + Exc_;
		const EigenMatrix G_ = Fnew_ * D_ * S - S * D_ * Fnew_;
		return std::vector<std::tuple<double, EigenMatrix, EigenMatrix>>{std::make_tuple(E_, G_, Fnew_)};
	};
	std::vector<EigenMatrix> Fs = {F};
	assert( GeneralizedDIIS(RawUpdate, 1e-5, 10, 100, E, Fs, output) && "Convergence failed!");
	return std::make_tuple(E, epsilons, occupations, C);
}

/*
std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedDIIS(
		double T, double Mu, EigenVector Occupations,
		EigenMatrix F, EigenMatrix S, EigenMatrix Z, EigenMatrix Hcore,
		short int* is, short int* js, short int* ks, short int* ls,
		double* ints, long int length,
		ExchangeCorrelation& xc,
		double* ws, long int ngrids, int nbasis,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		double* rhos,
		double* rho1xs, double* rho1ys, double* rho1zs, double* sigmas,
		double* lapls, double* taus,
		double* es,
		double* erhos, double* esigmas,
		double* elapls, double* etaus,
		int output, int nthreads){
	double E = 0;
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenVector occupations = Occupations;
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	std::function<
		std::tuple<
			double,
			EigenMatrix,
			EigenMatrix,
			EigenMatrix
		> (EigenMatrix)
	> ffunc = [&](EigenMatrix Fupdate){
		const EigenMatrix Fprime_ = Z.transpose() * Fupdate * Z;
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		double E_ = 0;
		if ( T ){
			const EigenArray ns = 1. / ( 1. + ( ( epsilons.array() - Mu ) / T ).exp() );
			occupations = ns.matrix();
			E_ += 2 * (
					T * (
						ns.pow(ns).log() + ( 1. - ns ).pow( 1. - ns ).log()
					).sum()
					- Mu * ns.sum()
			);
		}
		const EigenMatrix D_ = C * occupations.asDiagonal() * C.transpose();
		const EigenMatrix Ghf_ = Grhf(is, js, ks, ls, ints, length, D_, xc.EXX, nthreads);
		auto [Exc_, Gxc_] = xc.XCcode ? Gxc( // double, EigenMatrix
				xc,
				ws, ngrids, nbasis,
				aos,
				ao1xs, ao1ys, ao1zs,
				ao2ls,
				{2 * D_},
				rhos,
				rho1xs, rho1ys, rho1zs, sigmas,
				lapls, taus,
				es,
				erhos, esigmas,
				elapls, etaus,
				nthreads
		) : std::make_tuple(0, EigenZero(nbasis, nbasis));
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix F_ = Fhf_ + Gxc_;
		const EigenMatrix G = F_ * D_ * S - S * D_ * F_;
		E_ += (D_ * ( Hcore + Fhf_ )).trace() + Exc_;
		return std::make_tuple(E_, F_, G, D_);
	};
	assert( DIISSCF( ffunc, {1.e3, 0.15, 1.e3}, {1.e-8, 1.e-5, 1.e-5}, 20, 100, E, F, output) && "Convergence failed!" );
	return std::make_tuple(E, epsilons, occupations, C);
}
*/

void Multiwfn::HartreeFockKohnSham(std::string scf, int output, int nthreads){
	Eigen::setNbThreads(nthreads);

	const double T = this->Temperature;
	const double Mu = this->ChemicalPotential;
	if (output > 0) std::printf("Self-consistent field in %s-canonical ensemble\n", T > 0 ? "grand" : "micro");
	if ( T > 0 && scf.compare("DIIS") != 0 ) throw std::runtime_error("Only DIIS optimization is supported for finite-temperature DFT!");

	const EigenMatrix S = this->Overlap;
	const EigenMatrix Z = this->getCoefficientMatrix();

	const EigenMatrix Hcore = this->Kinetic + this->Nuclear;
	short int* is = this->BasisIs.data();
	short int* js = this->BasisJs.data();
	short int* ks = this->BasisKs.data();
	short int* ls = this->BasisLs.data();
	double* ints = this->RepulsionInts.data();
	long int length = this->RepulsionInts.size();

	ExchangeCorrelation& xc = this->XC;
	double* ws = this->Ws;
	long int ngrids = this->NumGrids;
	int nbasis = this->getNumBasis();
	double* aos = this->AOs;
	double* ao1xs = this->AO1Xs;
	double* ao1ys = this->AO1Ys;
	double* ao1zs = this->AO1Zs;
	double* ao2ls = this->AO2Ls;
	double* rhos = this->Rhos;
	double* rho1xs = this->Rho1Xs;
	double* rho1ys = this->Rho1Ys;
	double* rho1zs = this->Rho1Zs;
	double* sigmas = this->Sigmas;
	double* lapls = this->Lapls;
	double* taus = this->Taus;
	double* es = this->Es;
	double* e1rhos = this->E1Rhos;
	double* e1sigmas = this->E1Sigmas;
	double* e1lapls = this->E1Lapls;
	double* e1taus = this->E1Taus;

	if ( scf.compare("NEWTON") == 0 ){
		EigenMatrix Dprime = (this->getOccupation() / 2).asDiagonal();
		auto [E, epsilons, C] = RestrictedNewton(Dprime, Z, Hcore, is, js, ks, ls, ints, length, output-1, nthreads);
		this->E_tot += 2 * E;
		this->setEnergy(epsilons);
		this->setCoefficientMatrix(C);
	}else if ( scf.compare("QUASI") == 0 ){
		EigenMatrix Dprime = (this->getOccupation() / 2).asDiagonal();
		auto [E, epsilons, C] = RestrictedQuasiNewton(
				Dprime, Z, Hcore,
				is, js, ks, ls,
				ints, length,
				xc,
				ws, ngrids, nbasis,
				aos,
				ao1xs, ao1ys, ao1zs,
				ao2ls,
				rhos,
				rho1xs, rho1ys, rho1zs, sigmas,
				lapls, taus,
				es,
				e1rhos, e1sigmas,
				e1lapls, e1taus,
				output-1, nthreads
		);
		this->E_tot += 2 * E;
		this->setEnergy(epsilons);
		this->setCoefficientMatrix(C);
	}else if ( scf.compare("DIIS") == 0 ){
		EigenMatrix F = this->getFock();
		EigenVector Occ = this->getOccupation() / 2;
		auto [E, epsilons, occupations, C] = RestrictedDIIS(
				T, Mu, Occ,
				F, S, Z, Hcore,
				is, js, ks, ls,
				ints, length,
				xc,
				ws, ngrids, nbasis,
				aos,
				ao1xs, ao1ys, ao1zs,
				ao2ls,
				rhos,
				rho1xs, rho1ys, rho1zs, sigmas,
				lapls, taus,
				es,
				e1rhos, e1sigmas,
				e1lapls, e1taus,
				output-1, nthreads
		);
		this->E_tot += E;
		this->setEnergy(epsilons);
		this->setOccupation(occupations * 2);
		this->setCoefficientMatrix(C);
	}

}
