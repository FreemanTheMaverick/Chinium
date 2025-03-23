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
#include "../DIIS/CDIIS.h"
#include "../DIIS/ADIIS.h"
#include "FockFormation.h"

#include <iostream>


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

std::tuple<double, EigenVector, EigenVector, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedDIIS(
		double T, double Mu,
		EigenVector Occa, EigenVector Occb,
		EigenMatrix Fa, EigenMatrix Fb,
		EigenMatrix Za, EigenMatrix Zb,
		EigenMatrix S, EigenMatrix Hcore,
		short int* is, short int* js, short int* ks, short int* ls,
		double* ints, long int length, int output, int nthreads){
	double E = 0;
	EigenVector epsa = EigenZero(Za.cols(), 1);
	EigenVector epsb = EigenZero(Zb.cols(), 1);
	EigenVector occa = Occa;
	EigenVector occb = Occb;
	EigenMatrix Ca = EigenZero(Za.rows(), Za.cols());
	EigenMatrix Cb = EigenZero(Zb.rows(), Zb.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	std::vector<EigenMatrix> Jas, Jbs, Kas, Kbs;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> es(S);
	EigenMatrix sinvsqrt = es.operatorInverseSqrt();
	std::function<
		std::tuple<
			std::vector<double>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
		> (std::vector<EigenMatrix>&)
	> update_func = [&](std::vector<EigenMatrix>& Fs_){
		assert(Fs_.size() == 1 && "Two Fock matrices packed together in spin-unrestricted SCF!");
		const EigenMatrix Fa_ = Fs_[0].leftCols(Fa.cols());
		const EigenMatrix Fb_ = Fs_[0].rightCols(Fb.cols());
		const EigenMatrix Faprime_ = Za.transpose() * Fa_ * Za;
		const EigenMatrix Fbprime_ = Zb.transpose() * Fb_ * Zb;
		eigensolver.compute(Faprime_);
		epsa = eigensolver.eigenvalues();
		Ca = Za * eigensolver.eigenvectors();
		eigensolver.compute(Fbprime_);
		epsb = eigensolver.eigenvalues();
		Cb = Zb * eigensolver.eigenvectors();
		E = 0;
		if ( T ){
			const EigenArray nas = 1. / ( 1. + ( ( epsa.array() - Mu ) / T ).exp() );
			const EigenArray nbs = 1. / ( 1. + ( ( epsb.array() - Mu ) / T ).exp() );
			occa = nas.matrix();
			occb = nbs.matrix();
			E += (
					T * (
						nas.pow(nas).log() + ( 1. - nas ).pow( 1. - nas ).log()
					).sum()
					- Mu * nas.sum()
			);
			E += (
					T * (
						nbs.pow(nbs).log() + ( 1. - nbs ).pow( 1. - nbs ).log()
					).sum()
					- Mu * nbs.sum()
			);
		}
		const EigenMatrix Da_ = Ca * occa.asDiagonal() * Ca.transpose();
		const EigenMatrix Db_ = Cb * occb.asDiagonal() * Cb.transpose();
		auto [Ja_, Jb_, Ka_, Kb_] = Guhf(
			is, js, ks, ls,
			ints, length,
			Da_, Db_, 1, nthreads
		);
		const EigenMatrix Fhfa_ = Hcore + Ja_ + Jb_ - Ka_;
		const EigenMatrix Fhfb_ = Hcore + Ja_ + Jb_ - Kb_;
		const EigenMatrix Fnewa_ = Fhfa_;
		const EigenMatrix Fnewb_ = Fhfb_;
		E += 0.5 * ( Dot(Da_, Hcore + Fnewa_) + Dot(Db_, Hcore + Fnewb_) );
		const EigenMatrix Ga_ = sinvsqrt * (Fnewa_ * Da_ * S - S * Da_ * Fnewa_ ) * sinvsqrt;
		const EigenMatrix Gb_ = sinvsqrt * (Fnewb_ * Db_ * S - S * Db_ * Fnewb_ ) * sinvsqrt;
		EigenMatrix Fnew_ = EigenZero(Fa.rows(), Fa.cols() * 2);
		Fnew_ << Fnewa_, Fnewb_;
		EigenMatrix G_ = EigenZero(Fa.rows(), Fa.cols() * 2);
		G_ << Ga_, Gb_;
		EigenMatrix D_ = EigenZero(Fa.rows(), Fa.cols() * 2);
		D_ << Da_, Db_;
		return std::make_tuple(
				std::vector<double>{E},
				std::vector<EigenMatrix>{Fnew_},
				std::vector<EigenMatrix>{G_},
				std::vector<EigenMatrix>{D_}
		);
	};
	EigenMatrix F = EigenZero(Fa.rows(), Fa.cols() * 2);
	F << Fa, Fb;
	std::vector<EigenMatrix> Fs = {F};
	ADIIS adiis(&update_func, 1, 20, 1e-1, 100, output>0);
	if ( T == 0 ) if ( !adiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	CDIIS cdiis(&update_func, 1, 20, 1e-5, 100, output>0);
	cdiis.Steal(adiis);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsa, epsb, occa, occb, Ca, Cb);
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
	EigenVector old_occupations = Occ;
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	std::function<std::tuple<
			std::vector<double>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
			>(std::vector<EigenMatrix>&)
	> update_func = [&](std::vector<EigenMatrix>& Fs_){
		assert(Fs_.size() == 1 && "Only one Fock matrix should be optimized in spin-restricted SCF!");
		const EigenMatrix F_ = Fs_[0];
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z;
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		E = 0;
		if ( T ){
			const EigenArray ns = 1. / ( 1. + ( ( epsilons.array() - Mu ) / T ).exp() );
			old_occupations = occupations;
			occupations = (EigenVector)ns;
			E += 2 * (
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
		E += (D_ * ( Hcore + Fhf_ )).trace() + Exc_;
		EigenMatrix G_ = Fnew_ * D_ * S - S * D_ * Fnew_;
		return std::make_tuple(
				std::vector<double>{E},
				std::vector<EigenMatrix>{Fnew_},
				std::vector<EigenMatrix>{G_},
				std::vector<EigenMatrix>{D_}
		);
	};
	std::vector<EigenMatrix> Fs = {F};
	ADIIS adiis(&update_func, 1, 20, 1e-1, 100, output>0);
	if ( T == 0 ) if ( !adiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	CDIIS cdiis(&update_func, 1, 20, 1e-5, 100, output>0);
	cdiis.Steal(adiis);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilons, occupations, C);
}

void Multiwfn::HartreeFockKohnSham(std::string scf, int output, int nthreads){
	Eigen::setNbThreads(nthreads);

	const double T = this->Temperature;
	const double Mu = this->ChemicalPotential;
	if (output > 0) std::printf("Self-consistent field in %s-canonical ensemble\n", T > 0 ? "grand" : "micro");
	if ( T > 0 && scf.compare("DIIS") != 0 ) throw std::runtime_error("Only DIIS optimization is supported for finite-temperature DFT!");

	const EigenMatrix S = this->Overlap;
	const EigenMatrix Z = this->getCoefficientMatrix(this->Wfntype);

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
		if ( this->Wfntype == 0 ){
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
		}else if ( this->Wfntype == 1 ){
			EigenMatrix Fa = this->getFock(1);
			EigenMatrix Fb = this->getFock(2);
			EigenVector Occa = this->getOccupation(1);
			EigenVector Occb = this->getOccupation(2);
			auto [E, epsa, epsb, occa, occb, Ca, Cb] = UnrestrictedDIIS(
					T, Mu, Occa, Occb,
					Fa, Fb, Z, Z, S, Hcore,
					is, js, ks, ls,
					ints, length,
					output-1, nthreads
			);
			this->E_tot += E;
			this->setEnergy(epsa, 1);
			this->setEnergy(epsb, 2);
			this->setOccupation(occa, 1);
			this->setOccupation(occb, 2);
			this->setCoefficientMatrix(Ca, 1);
			this->setCoefficientMatrix(Cb, 2);
		}
	}
}
