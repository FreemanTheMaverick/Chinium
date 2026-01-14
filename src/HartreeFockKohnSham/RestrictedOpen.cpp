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

#define SafeLowSpin(mat) ( low_spin ? (mat).eval() : EigenZero(0, 0) )

static std::function<EigenMatrix (EigenMatrix)> Preconditioner(EigenMatrix U, EigenMatrix Uperp, EigenMatrix B, EigenMatrix C){
	return [U, Uperp, B, C](EigenMatrix Z){
		EigenMatrix Omega = U.transpose() * Z;
		Omega = Omega.cwiseProduct(B);
		EigenMatrix Kappa = Uperp.transpose() * Z;
		Kappa = Kappa.cwiseProduct(C);
		return ( U * Omega + Uperp * Kappa ).eval();
	};
}

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenRiemann(
		int nd, int na, int nb,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads){
	double E = 0;
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	// ARH hessian related
	AugmentedRoothaanHall arh;
	if constexpr ( scf_t == arh_t ) arh.Init(20, 1);

	const int nbasis = Z.rows();
	Maniverse::Flag flag(Cprime);
	std::vector<int> space = {nd, na};
	bool low_spin = nb > 0;
	if (low_spin) space.push_back(nb);
	flag.setBlockParameters(space);
	auto BlockParameters = flag.BlockParameters;
	Maniverse::Iterate M({flag.Clone()}, 1);
	Maniverse::PreconFunc dfunc_newton = [&](std::vector<EigenMatrix> Cprimes_, int /*order*/){
		const EigenMatrix Cprime_ = Cprimes_[0];
		const EigenMatrix Cdprime_ = FlagGetColumns(Cprime_, 0);
		const EigenMatrix Caprime_ = FlagGetColumns(Cprime_, 1);
		const EigenMatrix Cbprime_ = SafeLowSpin(FlagGetColumns(Cprime_, 2));
		const EigenMatrix Ddprime_ = Cdprime_ * Cdprime_.transpose();
		const EigenMatrix Daprime_ = Caprime_ * Caprime_.transpose();
		const EigenMatrix Dbprime_ = SafeLowSpin(Cbprime_ * Cbprime_.transpose());
		const EigenMatrix Dd_ = Z * Ddprime_ * Z.transpose();
		const EigenMatrix Da_ = Z * Daprime_ * Z.transpose();
		const EigenMatrix Db_ = SafeLowSpin(Z * Dbprime_ * Z.transpose());
		const auto [Gd_, Ga_, Gb_] = int4c2e.ContractInts(Dd_, Da_, Db_, nthreads, 1);
		const EigenMatrix Fhfd_ = Hcore + Gd_;
		const EigenMatrix Fhfa_ = Hcore + Ga_;
		const EigenMatrix Fhfb_ = SafeLowSpin(Hcore + Gb_);
		double Exc_ = 0;
		EigenMatrix Gxcd_ = EigenZero(nbasis, nbasis);
		EigenMatrix Gxca_ = EigenZero(nbasis, nbasis);
		EigenMatrix Gxcb_ = EigenZero(nbasis, nbasis);
		if (xc){
			if ( low_spin ) grid.getDensity({2 * Dd_, Da_, Db_});
			else grid.getDensity({2 * Dd_, Da_});
			xc.Evaluate("ev", grid);
			if constexpr ( scf_t == newton_t ) xc.Evaluate("f", grid);
			Exc_ = grid.getEnergy();
			const std::vector<EigenMatrix> Gxc_ = grid.getFock();
			Gxcd_ = Gxc_[0];
			Gxca_ = Gxc_[1];
			Gxcb_ = SafeLowSpin(Gxc_[2]);
		}
		const EigenMatrix Fd_ = Fhfd_ + Gxcd_;
		const EigenMatrix Fa_ = Fhfa_ + Gxca_;
		const EigenMatrix Fb_ = SafeLowSpin(Fhfb_ + Gxcb_);
		const EigenMatrix Fdprime_ = Z.transpose() * Fd_ * Z;
		const EigenMatrix Faprime_ = Z.transpose() * Fa_ * Z;
		const EigenMatrix Fbprime_ = SafeLowSpin(Z.transpose() * Fb_ * Z);

		// ARH hessian related
		if constexpr ( scf_t == arh_t ){
			EigenMatrix Dprime_ = EigenZero(Ddprime_.rows(), ( low_spin ? 3 : 2 ) * Ddprime_.cols());
			if (low_spin) Dprime_ << Ddprime_, Daprime_, Dbprime_;
			else Dprime_ << Ddprime_, Daprime_;
			EigenMatrix Fprime_ = EigenZero(Fdprime_.rows(), ( low_spin ? 3 : 2 ) * Fdprime_.cols());
			if (low_spin) Fprime_ << Fdprime_, 0.5 * Faprime_, 0.5 * Fbprime_;
			else Fprime_ << Fdprime_, 0.5 * Faprime_;
			arh.Append(Dprime_, Fprime_);
		}

		//eigensolver.compute(Fprime_);
		//epsilons = eigensolver.eigenvalues();
		//C = Z * eigensolver.eigenvectors();
		const double E_ = low_spin ?
			0.5 * Hcore.cwiseProduct( 2 * Dd_ + Da_ + Db_ ).sum()
			+ Fhfd_.cwiseProduct(Dd_).sum()
			+ 0.5 * Fhfa_.cwiseProduct(Da_).sum()
			+ 0.5 * Fhfb_.cwiseProduct(Db_).sum()
			+ Exc_ :
			0.5 * Hcore.cwiseProduct( 2 * Dd_ + Da_ ).sum()
			+ Fhfd_.cwiseProduct(Dd_).sum()
			+ 0.5 * Fhfa_.cwiseProduct(Da_).sum()
			+ Exc_;
		EigenMatrix Grad = EigenZero(Cprime_.rows(), nd + na + nb);
		if (low_spin) Grad << 4 * Fdprime_ * Cdprime_, 2 * Faprime_ * Caprime_, 2 * Fbprime_ * Cbprime_;
		else Grad << 4 * Fdprime_ * Cdprime_, 2 * Faprime_ * Caprime_;

		Eigen::HouseholderQR<EigenMatrix> qr(Cprime_);
		const EigenMatrix Call = qr.householderQ();
		C = Z * Call;
		const EigenMatrix Cperp = Call.rightCols(nbasis - nd - na - nb);
		std::vector<EigenMatrix> Fmos = { Call.transpose() * Fdprime_ * Call, Call.transpose() * Faprime_ * Call, SafeLowSpin(Call.transpose() * Fbprime_ * Call), EigenZero(nbasis, nbasis)};
		EigenMatrix A = EigenMatrix::Ones(nbasis, nbasis);
		for ( int i = 0; i < nbasis; i++ ){
			int I = 0; int Iscale = 4;
			if ( i < nd ){ I = 0; Iscale = 4; }
			else if ( i < nd + na ){ I = 1; Iscale = 2; }
			else if ( i < nd + na + nb ) { I = 2; Iscale = 2; }
			else { I = 3; Iscale = 0; };
			for ( int j = 0; j < nbasis; j++ ){
				int J = 0; int Jscale = 4;
				if ( j < nd ){ J = 0; Jscale = 4; }
				else if ( j < nd + na ){ J = 1; Jscale = 2; }
				else if ( j < nd + na + nb ){ J = 2; Jscale = 2; }
				else{ J = 3; Jscale = 0; }
				const double FIi = Fmos[I](i, i) * Iscale;
				const double FIj = Fmos[I](j, j) * Iscale;
				const double FJi = Fmos[J](i, i) * Jscale;
				const double FJj = Fmos[J](j, j) * Jscale;
				A(i, j) = std::abs(FIj + FJi - FIi - FJj);
				if ( A(i, j) < 0.1 ) A(i, j) = 0.1;
			}
		}
		const EigenMatrix B = A.block(0, 0, nd + na + nb, nd + na + nb);
		const EigenMatrix C = A.block(nd + na + nb, 0, nbasis - nd - na - nb, nd + na + nb);
		if constexpr ( scf_t == lbfgs_t ){
			const EigenMatrix Bsqrt = B.cwiseSqrt();
			const EigenMatrix Bsqrtinv = Bsqrt.cwiseInverse();
			const EigenMatrix Csqrt = C.cwiseSqrt();
			const EigenMatrix Csqrtinv = Csqrt.cwiseInverse();
			const std::function<EigenMatrix (EigenMatrix)> Psqrt = Preconditioner(Cprime_, Cperp, Bsqrtinv, Csqrtinv);
			const std::function<EigenMatrix (EigenMatrix)> Psqrtinv = Preconditioner(Cprime_, Cperp, Bsqrt, Csqrt);
			return std::make_tuple(
					E_,
					std::vector<EigenMatrix>{Grad},
					std::vector<std::function<EigenMatrix (EigenMatrix)>>{Psqrt},
					std::vector<std::function<EigenMatrix (EigenMatrix)>>{Psqrtinv}
			);
		}else if constexpr ( scf_t == newton_t || scf_t == arh_t ){
			std::function<EigenMatrix (EigenMatrix)> He = [low_spin, BlockParameters, Z, Fdprime_, Faprime_, Fbprime_, Cdprime_, Caprime_, Cbprime_, &int4c2e, &xc, &grid, nthreads, &arh](EigenMatrix vprime){
				EigenMatrix vdprime = FlagGetColumns(vprime, 0);
				EigenMatrix vaprime = FlagGetColumns(vprime, 1);
				EigenMatrix vbprime = SafeLowSpin(FlagGetColumns(vprime, 2));
				EigenMatrix Ddprime = Cdprime_ * vdprime.transpose();
				EigenMatrix Daprime = Caprime_ * vaprime.transpose();
				EigenMatrix Dbprime = SafeLowSpin(Cbprime_ * vbprime.transpose());
				Ddprime += Ddprime.transpose().eval();
				Daprime += Daprime.transpose().eval();
				Dbprime += Dbprime.transpose().eval();
				EigenMatrix HDd = Ddprime * 0;
				EigenMatrix HDa = Daprime * 0;
				EigenMatrix HDb = Dbprime * 0;
				if constexpr ( scf_t == newton_t ){
					const EigenMatrix Dd = Z * Ddprime * Z.transpose();
					const EigenMatrix Da = Z * Daprime * Z.transpose();
					const EigenMatrix Db = SafeLowSpin(Z * Dbprime * Z.transpose());
					auto [Gd_, Ga_, Gb_] = int4c2e.ContractInts(Dd, Da, Db, nthreads, 0);
					if (xc){
						if ( low_spin ) grid.getDensityU({{2*Dd}, {Da}, {Db}});
						else grid.getDensityU({{2*Dd}, {Da}});
						const std::vector<std::vector<EigenMatrix>> Gxc_ = grid.getFockU<u_t>();
						Gd_ += Gxc_[0][0];
						Ga_ += Gxc_[1][0];
						if ( low_spin ) Gb_ += SafeLowSpin(Gxc_[2][0]);
					}
					HDd = Z.transpose() * Gd_ * Z;
					HDa = Z.transpose() * Ga_ * Z;
					HDb = SafeLowSpin(Z.transpose() * Gb_ * Z);
				}else if constexpr ( scf_t == arh_t ){
					EigenMatrix Dprime = EigenZero(Ddprime.rows(), ( low_spin ? 3 : 2 ) * Ddprime.cols());
					if (low_spin) Dprime << Ddprime, Daprime, Dbprime;
					else Dprime << Ddprime, Daprime;
					const EigenMatrix HD = arh.Hessian(Dprime);
					HDd = HD(Eigen::placeholders::all, Eigen::seqN(0, Ddprime.cols()));
					HDa = HD(Eigen::placeholders::all, Eigen::seqN(Ddprime.cols(), Ddprime.cols()));;
					HDb = SafeLowSpin(HD(Eigen::placeholders::all, Eigen::seqN(2 * Ddprime.cols(), Ddprime.cols())));
				}
				EigenMatrix Hvd = HDd * Cdprime_ + Fdprime_ * vdprime;
				EigenMatrix Hva = HDa * Caprime_ + Faprime_ * vaprime;
				EigenMatrix Hvb = SafeLowSpin(HDb * Cbprime_ + Fbprime_ * vbprime);
				EigenMatrix Hv = EigenZero(vprime.rows(), vprime.cols());
				if (low_spin) Hv << 4 * Hvd, 2 * Hva, 2 * Hvb;
				else Hv << 4 * Hvd, 2 * Hva;
				return Hv;
			};
			const EigenMatrix Binv = B.cwiseInverse();
			const EigenMatrix Cinv = C.cwiseInverse();
			const std::function<EigenMatrix (EigenMatrix)> Pr = Preconditioner(Cprime_, Cperp, Binv, Cinv);
			return std::make_tuple(
					E_,
					std::vector<EigenMatrix>{Grad},
					std::vector<std::function<EigenMatrix (EigenMatrix)>>{He},
					std::vector<std::function<EigenMatrix (EigenMatrix)>>{Pr}
			);
		}
	};

	const std::tuple<double, double, double> tol = {1.e-8, 1.e-5, 1.e-5};
	if constexpr ( scf_t == lbfgs_t ){
		if ( ! Maniverse::LBFGS(
					dfunc_newton, tol,
					20, 300, 0.1, 0.75, 100,
					E, M, output
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
		int nd, int na, int nb,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedOpenRiemann<lbfgs_t>(nd, na, nb, int2c1e, int4c2e, xc, grid, Cprime, Z, output, nthreads);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenNewton(
		int nd, int na, int nb,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedOpenRiemann<newton_t>(nd, na, nb, int2c1e, int4c2e, xc, grid, Cprime, Z, output, nthreads);
}

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenARH(
		int nd, int na, int nb,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedOpenRiemann<arh_t>(nd, na, nb, int2c1e, int4c2e, xc, grid, Cprime, Z, output, nthreads);
}
