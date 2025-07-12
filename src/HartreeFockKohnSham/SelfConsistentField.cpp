#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
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
#include <libmwfn.h>

#include "../Macro.h"
#include "../Integral/Int2C1E.h"
#include "../Integral/Int4C2E.h"
#include "../Grid/Grid.h"
#include "../ExchangeCorrelation.h"
#include "../DIIS/CDIIS.h"
#include "../DIIS/EDIIS.h"
#include "../DIIS/ADIIS.h"

#define S (int2c1e.Overlap)
#define Hcore (int2c1e.Kinetic + int2c1e.Nuclear )



std::tuple<double, EigenVector, EigenMatrix> RestrictedGrassmann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Dprime, EigenMatrix Z,
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
	> dfunc_newton = [&](EigenMatrix Dprime_, int order){
		const EigenMatrix D_ = Z * Dprime_ * Z.transpose();
		const EigenMatrix Ghf_ = int4c2e.ContractInts(D_, nthreads, 1);
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(Ghf_.rows(), Ghf_.cols());
		if (xc){
			grid.getDensity(2 * D_);
			xc.Evaluate("ev", grid);
			xc.Evaluate("f", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock();
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix F_ = Fhf_+ Gxc_;
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z; // Euclidean gradient
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		const double E_ = 0.5 * (( D_ * ( Hcore + Fhf_ ) ).trace() + Exc_);
		std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix vprime){ return vprime; };
		if ( order == 2 ) He = [Z, &int4c2e, &grid, nthreads](EigenMatrix vprime){
			const EigenMatrix v = Z * vprime * Z.transpose();
			const EigenMatrix FhfU = int4c2e.ContractInts(v, nthreads, 0);
			std::vector<Eigen::Tensor<double, 1>> RhoUss, SigmaUss;
			std::vector<Eigen::Tensor<double, 2>> Rho1Uss;
			grid.getDensityU({2*v}, RhoUss, Rho1Uss, SigmaUss);
			const EigenMatrix FxcU = grid.getFockU(RhoUss, Rho1Uss, SigmaUss)[0];
			const EigenMatrix FU = FhfU + FxcU;
			return (EigenMatrix)(Z.transpose() * FU * Z);
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

std::tuple<double, EigenVector, EigenVector, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Fa, EigenMatrix Fb,
		EigenVector Occa, EigenVector Occb,
		EigenMatrix Za, EigenMatrix Zb,
		int output, int nthreads){
	double oldE = 0;
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
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
		> (std::vector<EigenMatrix>&, std::vector<bool>&)
	> update_func = [&](std::vector<EigenMatrix>& Fs_, std::vector<bool>&){
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
		oldE = E;
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
		auto [Ghfa_, Ghfb_] = int4c2e.ContractInts(Da_, Db_, nthreads, 1);
		const EigenMatrix Fhfa_ = Hcore + Ghfa_;
		const EigenMatrix Fhfb_ = Hcore + Ghfb_;
		const EigenMatrix Fnewa_ = Fhfa_;
		const EigenMatrix Fnewb_ = Fhfb_;
		E += 0.5 * ( Dot(Da_, Hcore + Fnewa_) + Dot(Db_, Hcore + Fnewb_) );
		if (output>0){
			if ( T == 0 ) std::printf("Electronic energy = %.10f\n", E);
			else std::printf("Electronic grand potential = %.10f\n", E);
			std::printf("Changed by %E from the last step\n", E - oldE);
		}
		const EigenMatrix Ga_ = sinvsqrt * (Fnewa_ * Da_ * S - S * Da_ * Fnewa_ ) * sinvsqrt;
		const EigenMatrix Gb_ = sinvsqrt * (Fnewb_ * Db_ * S - S * Db_ * Fnewb_ ) * sinvsqrt;
		EigenMatrix Fnew_ = EigenZero(Fa.rows(), Fa.cols() * 2);
		Fnew_ << Fnewa_, Fnewb_;
		EigenMatrix G_ = EigenZero(Fa.rows(), Fa.cols() * 2);
		G_ << Ga_, Gb_;
		EigenMatrix Aux_ = EigenZero(Fa.rows(), Fa.cols() * 2 + 1);
		Aux_ << Da_, Db_, EigenZero(Fa.rows(), 1);
		Aux_(0, Fa.cols() * 2) = E;
		return std::make_tuple(
				std::vector<EigenMatrix>{Fnew_},
				std::vector<EigenMatrix>{G_},
				std::vector<EigenMatrix>{Aux_}
		);
	};
	EigenMatrix F = EigenZero(Fa.rows(), Fa.cols() * 2);
	F << Fa, Fb;
	std::vector<EigenMatrix> Fs = {F};
	ADIIS adiis(&update_func, 1, 20, 1e-1, 100, output>0 ? 2 : 0);
	if ( T == 0 ) if ( !adiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	CDIIS cdiis(&update_func, 1, 20, 1e-8, 100, output>0 ? 2 : 0);
	if ( T > 0 ) cdiis.Damps.push_back(std::make_tuple(0.1, 100, 0.75));
	cdiis.Steal(adiis);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsa, epsb, occa, occb, Ca, Cb);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix F, EigenVector Occ, EigenMatrix Z,
		int output, int nthreads){
	double oldE = 0;
	double E = 0;
	const int nbasis = F.cols();
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenVector occupations = Occ;
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	std::function<std::tuple<
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
			>(std::vector<EigenMatrix>&, std::vector<bool>&)
	> update_func = [&](std::vector<EigenMatrix>& Fs_, std::vector<bool>&){
		assert(Fs_.size() == 1 && "Only one Fock matrix should be optimized in spin-restricted SCF!");
		const EigenMatrix F_ = Fs_[0];
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z;
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		oldE = E;
		E = 0;
		if ( T ){
			const EigenArray ns = 1. / ( 1. + ( ( epsilons.array() - Mu ) / T ).exp() );
			occupations = (EigenVector)ns;
			E += 2 * (
					T * (
						ns.pow(ns).log() + ( 1. - ns ).pow( 1. - ns ).log()
					).sum()
					- Mu * ns.sum()
			);
		}
		const EigenMatrix D_ = C * occupations.asDiagonal() * C.transpose();
		const EigenMatrix Ghf_ = int4c2e.ContractInts(D_, nthreads, 1);
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(nbasis, nbasis);
		if (xc){
			grid.getDensity(2 * D_);
			xc.Evaluate("ev", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock();
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix Fnew_ = Fhf_ + Gxc_;
		E += (D_ * ( Hcore + Fhf_ )).trace() + Exc_;
		if (output>0){
			if ( T == 0 ) std::printf("Electronic energy = %.10f\n", E);
			else std::printf("Electronic grand potential = %.10f\n", E);
			std::printf("Changed by %E from the last step\n", E - oldE);
		}
		EigenMatrix G_ = Fnew_ * D_ * S - S * D_ * Fnew_;
		EigenMatrix Aux_ = EigenZero(F.rows(), F.cols() + 1);
		Aux_ << D_, EigenZero(F.rows(), 1);
		Aux_(0, F.cols()) = E;
		return std::make_tuple(
				std::vector<EigenMatrix>{Fnew_},
				std::vector<EigenMatrix>{G_},
				std::vector<EigenMatrix>{Aux_}
		);
	};
	std::vector<EigenMatrix> Fs = {F};
	ADIIS adiis(&update_func, 1, 20, 1e-1, 100, output>0 ? 2 : 0);
	if ( T == 0 ) if ( !adiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	CDIIS cdiis(&update_func, 1, 20, 1e-8, 100, output>0 ? 2 : 0);
	if ( T > 0 ) cdiis.Damps.push_back(std::make_tuple(0.1, 100, 0.75));
	cdiis.Steal(adiis);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilons, occupations, C);
}

double HartreeFockKohnSham(Mwfn& mwfn, Environment& env, Int2C1E& int2c1e, Int4C2E& int4c2e, ExchangeCorrelation& xc, Grid& grid, std::string scf, int output, int nthreads){
	Eigen::setNbThreads(nthreads);

	const double T = env.Temperature;
	const double Mu = env.ChemicalPotential;
	if (output > 0) std::printf("Self-consistent field in %s-canonical ensemble\n", T > 0 ? "grand" : "micro");
	if ( T > 0 && scf == "DIIS" ) throw std::runtime_error("Only DIIS optimization is supported for finite-temperature DFT!");

	const EigenMatrix Z = mwfn.getCoefficientMatrix(mwfn.Wfntype);

	double E_scf = 0;
	if ( scf == "GRASSMANN" ){
		EigenMatrix Dprime = (mwfn.getOccupation() / 2).asDiagonal();
		auto [E, epsilons, C] = RestrictedGrassmann(int2c1e, int4c2e, xc, grid, Dprime, Z, output-1, nthreads);
		E_scf = 2 * E;
		mwfn.setEnergy(epsilons);
		mwfn.setCoefficientMatrix(C);
	}else if ( scf == "DIIS" ){
		if ( mwfn.Wfntype == 0 ){
			EigenMatrix F = mwfn.getFock();
			EigenVector Occ = mwfn.getOccupation() / 2;
			auto [E, epsilons, occupations, C] = RestrictedDIIS(
					T, Mu,
					int2c1e, int4c2e,
					xc, grid,
					F, Occ, Z,
					output-1, nthreads
			);
			E_scf = E;
			mwfn.setEnergy(epsilons);
			mwfn.setOccupation(occupations * 2);
			mwfn.setCoefficientMatrix(C);
		}else if ( mwfn.Wfntype == 1 ){
			EigenMatrix Fa = mwfn.getFock(1);
			EigenMatrix Fb = mwfn.getFock(2);
			EigenVector Occa = mwfn.getOccupation(1);
			EigenVector Occb = mwfn.getOccupation(2);
			auto [E, epsa, epsb, occa, occb, Ca, Cb] = UnrestrictedDIIS(
					T, Mu,
					int2c1e, int4c2e,
					Fa, Fb,
					Occa, Occb,
					Z, Z,
					output-1, nthreads
			);
			E_scf = E;
			mwfn.setEnergy(epsa, 1);
			mwfn.setEnergy(epsb, 2);
			mwfn.setOccupation(occa, 1);
			mwfn.setOccupation(occb, 2);
			mwfn.setCoefficientMatrix(Ca, 1);
			mwfn.setCoefficientMatrix(Cb, 2);
		}
	}
	return E_scf;
}
