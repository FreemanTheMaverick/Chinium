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

template <typename FuncType, bool exact_hess>
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
	if constexpr ( ! exact_hess ) arh.Init(20, 1);

	Maniverse::Flag flag(Cprime);
	flag.setBlockParameters({nd, ns});
	Maniverse::Iterate M({flag.Clone()}, 1);
	FuncType dfunc_newton = [&](std::vector<EigenMatrix> Cprimes_, int order){
		const EigenMatrix Cprime_ = Cprimes_[0];
		const EigenMatrix Cdprime_ = Cprime_.leftCols(nd);
		const EigenMatrix Csprime_ = Cprime_.rightCols(ns);
		const EigenMatrix Ddprime_ = Cdprime_ * Cdprime_.transpose();
		const EigenMatrix Dsprime_ = Csprime_ * Csprime_.transpose();
		const EigenMatrix Dd_ = Z * Ddprime_ * Z.transpose();
		const EigenMatrix Ds_ = Z * Dsprime_ * Z.transpose();
		auto [Jd, Kd] = int4c2e.ContractInts2(Dd_, nthreads, 1);
		auto [Js, Ks] = int4c2e.ContractInts2(Ds_, nthreads, 1);
		const EigenMatrix Fd_ = Hcore + 2 * Jd + Js - Kd - 0.5 * Ks;
		const EigenMatrix Fs_ = 0.5 * ( Hcore + 2 * Jd + Js - Kd - Ks );
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(Fd_.rows(), Fd_.cols());
		const EigenMatrix Fdprime_ = Z.transpose() * Fd_ * Z;
		const EigenMatrix Fsprime_ = Z.transpose() * Fs_ * Z;

		// ARH hessian related
		if constexpr ( ! exact_hess ){
			EigenMatrix Dprime_ = EigenZero(Ddprime_.rows(), 2 * Ddprime_.cols());
			Dprime_ << Ddprime_, Dsprime_;
			EigenMatrix Fprime_ = EigenZero(Fdprime_.rows(), 2 * Fdprime_.cols());
			Fprime_ << Fdprime_, Fsprime_;
			arh.Append(Dprime_, Fprime_);
		}

		//eigensolver.compute(Fprime_);
		//epsilons = eigensolver.eigenvalues();
		//C = Z * eigensolver.eigenvectors();
		const double E_ = Hcore.cwiseProduct( 2 * Dd_ + Ds_ ).sum()
			+ ( 2 * Jd - Kd ).cwiseProduct( Dd_ + Ds_ ).sum()
			+ 0.5 * ( Js - Ks ).cwiseProduct( Ds_ ).sum()
			+ Exc_;
		EigenMatrix Grad = EigenZero(Cprime_.rows(), nd + ns);
		Grad << 4 * Fdprime_ * Cdprime_, 4 * Fsprime_ * Csprime_;
		if constexpr (std::is_same_v<FuncType, Maniverse::UnpreconFirstFunc>){
			return std::make_tuple(
					E_,
					std::vector<EigenMatrix>{Grad}
			);
		}else if constexpr (std::is_same_v<FuncType, Maniverse::UnpreconSecondFunc>){
			std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix vprime){ return vprime; };
			if ( order == 2 ){
				if constexpr ( exact_hess ) He = [nd, ns, Z, Fdprime_, Fsprime_, Cdprime_, Csprime_, &int4c2e, nthreads](EigenMatrix vprime){
					EigenMatrix vdprime = vprime.leftCols(nd);
					EigenMatrix vsprime = vprime.rightCols(ns);
					EigenMatrix Ddprime = Cdprime_ * vdprime.transpose();
					EigenMatrix Dsprime = Csprime_ * vsprime.transpose();
					Ddprime += Ddprime.transpose().eval();
					Dsprime += Dsprime.transpose().eval();
					const EigenMatrix Dd = Z * Ddprime * Z.transpose();
					const EigenMatrix Ds = Z * Dsprime * Z.transpose();
					auto [Jd, Kd] = int4c2e.ContractInts2(Dd, nthreads, 0);
					auto [Js, Ks] = int4c2e.ContractInts2(Ds, nthreads, 0);
					const EigenMatrix Jdprime = Z.transpose() * Jd * Z;
					const EigenMatrix Jsprime = Z.transpose() * Js * Z;
					const EigenMatrix Kdprime = Z.transpose() * Kd * Z;
					const EigenMatrix Ksprime = Z.transpose() * Ks * Z;
					EigenMatrix Hvd = ( 2 * Jdprime + Jsprime - Kdprime - 0.5 * Ksprime ) * Cdprime_ + Fdprime_ * vdprime;
					EigenMatrix Hvs = 0.5 * ( 2 * Jdprime + Jsprime - Kdprime - Ksprime ) * Csprime_ + Fsprime_ * vsprime;
					EigenMatrix Hv = EigenZero(vprime.rows(), vprime.cols());
					Hv << 4 * Hvd, 4 * Hvs;
					return Hv;
				};
				else He = [nd, ns, Cdprime_, Csprime_, Fdprime_, Fsprime_, &arh, nthreads](EigenMatrix vprime){
					EigenMatrix vdprime = vprime.leftCols(nd);
					EigenMatrix vsprime = vprime.rightCols(ns);
					EigenMatrix Ddprime = Cdprime_ * vdprime.transpose();
					EigenMatrix Dsprime = Csprime_ * vsprime.transpose();
					Ddprime += Ddprime.transpose().eval();
					Dsprime += Dsprime.transpose().eval();
					EigenMatrix Dprime = EigenZero(Ddprime.rows(), 2 * Ddprime.cols());
					Dprime << Ddprime, Dsprime;
					EigenMatrix HD = arh.Hessian(Dprime);
					EigenMatrix HDd = HD.leftCols(Ddprime.cols());
					EigenMatrix HDs = HD.rightCols(Dsprime.cols());
					EigenMatrix Hvd = HDd * Cdprime_ + Fdprime_ * vdprime;
					EigenMatrix Hvs = HDs * Csprime_ + Fsprime_ * vsprime;
					EigenMatrix Hv = EigenZero(vprime.rows(), vprime.cols());
					Hv << 4 * Hvd, 4 * Hvs;
					return Hv;
				};
			}
			return std::make_tuple(
					E_,
					std::vector<EigenMatrix>{Grad},
					std::vector<std::function<EigenMatrix (EigenMatrix)>>{He}
			);
		}
	};

	const std::tuple<double, double, double> tol = {1.e-8, 1.e-5, 1.e-5};
	if constexpr (std::is_same_v<FuncType, Maniverse::UnpreconFirstFunc>){
		if ( ! Maniverse::LBFGS(
					dfunc_newton, tol,
					20, 300, E, M, output
		) ) throw std::runtime_error("Convergence failed!");
	}else if constexpr (std::is_same_v<FuncType, Maniverse::UnpreconSecondFunc>){
		double ratio = 0;
		if constexpr ( exact_hess ) ratio = 0.001;
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
	return RestrictedOpenRiemann<Maniverse::UnpreconFirstFunc, 0>(nd, ns, int2c1e, int4c2e, Cprime, Z, output, nthreads);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenNewton(
		int nd, int ns,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedOpenRiemann<Maniverse::UnpreconSecondFunc, 1>(nd, ns, int2c1e, int4c2e, Cprime, Z, output, nthreads);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenARH(
		int nd, int ns,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedOpenRiemann<Maniverse::UnpreconSecondFunc, 0>(nd, ns, int2c1e, int4c2e, Cprime, Z, output, nthreads);
}
