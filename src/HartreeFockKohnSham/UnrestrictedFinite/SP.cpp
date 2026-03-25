#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <tuple>
#include <cstdio>
#include <Maniverse/Manifold/Grassmann.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <libmwfn.h>

#include "../../Macro.h"
#include "../../Integral.h"
#include "../../Grid.h"
#include "../../ExchangeCorrelation.h"
#include "../../DIIS.h"

#include "../AugmentedRoothaanHall.h"
#include "../UnrestrictedFinite.h"

#define S ( int2c1e.Overlap )
#define Hcore ( int2c1e.Kinetic + int2c1e.Nuclear )

std::tuple<double, EigenVector, EigenVector, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedFiniteDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Fa, EigenMatrix Fb,
		EigenVector Occa, EigenVector Occb,
		EigenMatrix Za, EigenMatrix Zb,
		int output, int nthreads){
	double oldE = 0;
	double E = 0;
	const int nbasis = Fa.rows();
	EigenVector epsa = EigenZero(Za.cols(), 1);
	EigenVector epsb = EigenZero(Zb.cols(), 1);
	EigenVector occa = Occa;
	EigenVector occb = Occb;
	EigenMatrix Ca = EigenZero(Za.rows(), Za.cols());
	EigenMatrix Cb = EigenZero(Zb.rows(), Zb.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
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
		const auto [J_, _, Ka_, Kb_] = int4c2e.ContractInts(EigenZero(0, 0), Da_, Db_, nthreads, 1);
		const EigenMatrix Fhfa_ = Hcore + J_ - Ka_;
		const EigenMatrix Fhfb_ = Hcore + J_ - Kb_;
		double Exc_ = 0;
		EigenMatrix Gxca_ = EigenZero(nbasis, nbasis);
		EigenMatrix Gxcb_ = EigenZero(nbasis, nbasis);
		if (xc){
			grid.getDensity({Da_, Db_});
			xc.Evaluate("ev", grid);
			Exc_ = grid.getEnergy();
			std::vector<EigenMatrix> Gxc_ = grid.getFock();
			Gxca_ = Gxc_[0];
			Gxcb_ = Gxc_[1];
		}
		const EigenMatrix Fnewa_ = Fhfa_ + Gxca_;
		const EigenMatrix Fnewb_ = Fhfb_ + Gxcb_;

		E += 0.5 * ( Dot(Da_, Hcore + Fhfa_) + Dot(Db_, Hcore + Fhfb_) ) + Exc_;
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
	ADIIS adiis(&update_func, 1, 20, 1e-1, 300, output>0 ? 2 : 0);
	if ( T == 0 ) if ( !adiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	CDIIS cdiis(&update_func, 1, 20, 1e-6, 300, output>0 ? 2 : 0);
	if ( T > 0 ) cdiis.Damps.push_back(std::make_tuple(0.1, 100, 0.75));
	cdiis.Steal(adiis);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsa, epsb, occa, occb, Ca, Cb);
}

void UGC_SCF::Calculate0(){
	if ( scftype == "DRY" ) return;
	EigenMatrix Z = mwfn.getCoefficientMatrix(1);
	EigenMatrix Fa = mwfn.getFock(1);
	EigenMatrix Fb = mwfn.getFock(2);
	EigenVector Occa = mwfn.getOccupation(1);
	EigenVector Occb = mwfn.getOccupation(2);
	auto [E, epsa, epsb, occa, occb, Ca, Cb] =
		/* scftype == "DIIS" ? */ UnrestrictedFiniteDIIS(Temperature, ChemicalPotential, int2c1e, int4c2e, xc, grid, Fa, Fb, Occa, Occb, Z, Z, 1, nthreads);
	Energy += E;
	mwfn.setEnergy(epsa, 1);
	mwfn.setEnergy(epsb, 2);
	mwfn.setOccupation(occa, 1);
	mwfn.setOccupation(occb, 2);
	mwfn.setCoefficientMatrix(Ca, 1);
	mwfn.setCoefficientMatrix(Cb, 2);
}
