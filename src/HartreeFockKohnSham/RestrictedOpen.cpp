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
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/TrustRegion.h>
#include <libmwfn.h>

#include "../Macro.h"
#include "../Integral/Int2C1E.h"
#include "../Integral/Int4C2E.h"
#include "../Grid/Grid.h"
#include "../ExchangeCorrelation.h"
#include "AugmentedRoothaanHall.h"

#define S (int2c1e.Overlap)
#define Hcore (int2c1e.Kinetic + int2c1e.Nuclear )

#define DummyFunc [](EigenMatrix v){ return v; }

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenRiemann(
		int nd, int ns,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads){
	double E = 0;
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	// ARH hessian related
	AugmentedRoothaanHall arh;
	if constexpr ( scf_t == arh_t ) arh.Init(20, 1);

	Maniverse::Flag flag(Cprime);
	flag.setBlockParameters({nd, ns});
	Maniverse::Iterate M({flag.Clone()}, 1);
	Maniverse::PreconFunc dfunc_newton = [&](std::vector<EigenMatrix> Cprimes_, int /*order*/){
		const EigenMatrix Cprime_ = Cprimes_[0];
		const EigenMatrix Cdprime_ = Cprime_.leftCols(nd);
		const EigenMatrix Csprime_ = Cprime_.rightCols(ns);
		const EigenMatrix Ddprime_ = Cdprime_ * Cdprime_.transpose();
		const EigenMatrix Dsprime_ = Csprime_ * Csprime_.transpose();
		const EigenMatrix Dd_ = Z * Ddprime_ * Z.transpose();
		const EigenMatrix Ds_ = Z * Dsprime_ * Z.transpose();
		const auto [Gd_, Gs_, _] = int4c2e.ContractInts(Dd_, Ds_, EigenZero(0, 0), nthreads, 1);
		const EigenMatrix Fd_ = Hcore + Gd_;
		const EigenMatrix Fs_ = Hcore + Gs_;
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(Fd_.rows(), Fd_.cols());
		const EigenMatrix Fdprime_ = Z.transpose() * Fd_ * Z;
		const EigenMatrix Fsprime_ = Z.transpose() * Fs_ * Z;

		// ARH hessian related
		if constexpr ( scf_t == arh_t ){
			EigenMatrix Dprime_ = EigenZero(Ddprime_.rows(), 2 * Ddprime_.cols());
			Dprime_ << Ddprime_, Dsprime_;
			EigenMatrix Fprime_ = EigenZero(Fdprime_.rows(), 2 * Fdprime_.cols());
			Fprime_ << Fdprime_, 0.5 * Fsprime_;
			arh.Append(Dprime_, Fprime_);
		}

		//eigensolver.compute(Fprime_);
		//epsilons = eigensolver.eigenvalues();
		//C = Z * eigensolver.eigenvectors();
		const double E_ = 0.5 * Hcore.cwiseProduct( 2 * Dd_ + Ds_ ).sum()
			+ Fd_.cwiseProduct(Dd_).sum()
			+ 0.5 * Fs_.cwiseProduct(Ds_).sum()
			+ Exc_;
		EigenMatrix Grad = EigenZero(Cprime_.rows(), nd + ns);
		Grad << 4 * Fdprime_ * Cdprime_, 2 * Fsprime_ * Csprime_;
		if constexpr ( scf_t == lbfgs_t ){
			return std::make_tuple(
					E_,
					std::vector<EigenMatrix>{Grad},
					std::vector<std::function<EigenMatrix (EigenMatrix)>>{DummyFunc},
					std::vector<std::function<EigenMatrix (EigenMatrix)>>{DummyFunc}
			);
		}else{
			std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix vprime){ return vprime; };
			if constexpr ( scf_t == newton_t || scf_t == arh_t ) He = [nd, ns, Z, Fdprime_, Fsprime_, Cdprime_, Csprime_, &int4c2e, nthreads, &arh](EigenMatrix vprime){
				EigenMatrix vdprime = vprime.leftCols(nd);
				EigenMatrix vsprime = vprime.rightCols(ns);
				EigenMatrix Ddprime = Cdprime_ * vdprime.transpose();
				EigenMatrix Dsprime = Csprime_ * vsprime.transpose();
				Ddprime += Ddprime.transpose().eval();
				Dsprime += Dsprime.transpose().eval();
				EigenMatrix HDd = Ddprime * 0;
				EigenMatrix HDs = Ddprime * 0;
				if constexpr ( scf_t == newton_t ){
					const EigenMatrix Dd = Z * Ddprime * Z.transpose();
					const EigenMatrix Ds = Z * Dsprime * Z.transpose();
					const auto [Gd_, Gs_, _] = int4c2e.ContractInts(Dd, Ds, EigenZero(0, 0), nthreads, 0);
					HDd = Z.transpose() * Gd_ * Z;
					HDs = Z.transpose() * Gs_ * Z;
				}else{
					EigenMatrix Dprime = EigenZero(Ddprime.rows(), 2 * Ddprime.cols());
					Dprime << Ddprime, Dsprime;
					const EigenMatrix HD = arh.Hessian(Dprime);
					HDd = HD.leftCols(Ddprime.cols());
					HDs = HD.rightCols(Dsprime.cols());
				}
				EigenMatrix Hvd = HDd * Cdprime_ + Fdprime_ * vdprime;
				EigenMatrix Hvs = HDs * Csprime_ + Fsprime_ * vsprime;
				EigenMatrix Hv = EigenZero(vprime.rows(), vprime.cols());
				Hv << 4 * Hvd, 2 * Hvs;
				return Hv;
			};
			return std::make_tuple(
					E_,
					std::vector<EigenMatrix>{Grad},
					std::vector<std::function<EigenMatrix (EigenMatrix)>>{He},
					std::vector<std::function<EigenMatrix (EigenMatrix)>>{DummyFunc}
			);
		}
	};

	const std::tuple<double, double, double> tol = {1.e-8, 1.e-5, 1.e-5};
	if constexpr ( scf_t == lbfgs_t ){
		if ( ! Maniverse::LBFGS(
					dfunc_newton, tol,
					20, 300, E, M, output
		) ) throw std::runtime_error("Convergence failed!");
	}else{
		double ratio = 0;
		if constexpr ( scf_t == newton_t ) ratio = 0.001;
		else ratio = 0.01;
		Maniverse::TrustRegionSetting tr_setting;
		if ( ! Maniverse::TrustRegion(
					dfunc_newton, tr_setting, tol,
					ratio, 1, 300, E, M, output
		) ) throw std::runtime_error("Convergence failed!");
	}
	return std::make_tuple(E, epsilons, C);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenLBFGS(
		int nd, int ns,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedOpenRiemann<lbfgs_t>(nd, ns, int2c1e, int4c2e, Cprime, Z, output, nthreads);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenNewton(
		int nd, int ns,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedOpenRiemann<newton_t>(nd, ns, int2c1e, int4c2e, Cprime, Z, output, nthreads);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenARH(
		int nd, int ns,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedOpenRiemann<arh_t>(nd, ns, int2c1e, int4c2e, Cprime, Z, output, nthreads);
}
