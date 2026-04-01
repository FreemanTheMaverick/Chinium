#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <cstdio>
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <libmwfn.h>

#include "../../Macro.h"
#include "../../DIIS.h"

#include "../Universal.h"
#include "../Unrestricted.h"

#define S ( int2c1e.Overlap)
#define Hcore ( int2c1e.Kinetic + int2c1e.Nuclear )

std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedDIIS(
		int na, int nb,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Fa, EigenMatrix Fb,
		EigenMatrix Za, EigenMatrix Zb,
		int output, int nthreads){
	double oldE = 0;
	double E = 0;
	const int nbasis = Fa.rows();
	EigenVector epsa = EigenZero(Za.cols(), 1);
	EigenVector epsb = EigenZero(Zb.cols(), 1);
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
		const EigenMatrix Da_ = Ca.leftCols(na) * Ca.leftCols(na).transpose();
		const EigenMatrix Db_ = Cb.leftCols(nb) * Cb.leftCols(nb).transpose();
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
			std::printf("Electronic energy = %.10f\n", E);
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
	if ( !adiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	CDIIS cdiis(&update_func, 1, 20, 1e-6, 300, output>0 ? 2 : 0);
	cdiis.Steal(adiis);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsa, epsb, Ca, Cb);
}

namespace{

#undef S
#undef Hcore
#define S (int2c1e->Overlap)
#define Hcore (int2c1e->Kinetic + int2c1e->Nuclear )

class ObjBase: public UniversalObjBase{ public:
	std::vector<EigenMatrix> Cprime_perps = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::vector<EigenMatrix> Ks = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::vector<EigenMatrix> Ls = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };

	using UniversalObjBase::UniversalObjBase;

	virtual void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		Cprimes_.insert(Cprimes_.begin(), EigenZero(0, 0));
		UniversalObjBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Gradient = Gradients;
			Gradient.erase(Gradient.begin());
			for ( int type = 1; type < 3; type++ ){
				Eigen::HouseholderQR<EigenMatrix> qr(Cprimes[type]);
				const EigenMatrix Call = qr.householderQ();
				Cprime_perps[type] = Call.rightCols(nbasis - Norbs[type]);
				EigenMatrix A = EigenMatrix::Ones(nbasis, nbasis);
				const EigenMatrix Fmo = Call.transpose() * Fprimes[type] * Call;
				for ( int o = 0; o < Norbs[type]; o++ ){
					for ( int v = Norbs[type]; v < nbasis; v++ ){
						A(o, v) = A(v, o) = 2 * std::abs( Fmo(v, v) - Fmo(o, o) );
					}
				}
				Ks[type] = A.topLeftCorner(Norbs[type], Norbs[type]);
				Ls[type] = A.bottomLeftCorner(nbasis - Norbs[type], Norbs[type]);
			}
		}
	};
};

class ObjLBFGS: public ObjBase{ public:
	std::vector<EigenMatrix> Ksqrts = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::vector<EigenMatrix> Ksqrtinvs = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::vector<EigenMatrix> Lsqrts = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::vector<EigenMatrix> Lsqrtinvs = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };

	using ObjBase::ObjBase;

	void Calculate(std::vector<EigenMatrix> Dprimes, std::vector<int> derivatives) override{
		ObjBase::Calculate(Dprimes, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			for ( int type = 1; type < 3; type++ ){
				Ksqrts[type] = Ks[type].cwiseSqrt();
				Ksqrtinvs[type] = Ksqrts[type].cwiseInverse();
				Lsqrts[type] = Ls[type].cwiseSqrt();
				Lsqrtinvs[type] = Lsqrts[type].cwiseInverse();
			}
		}
	};

	std::vector<EigenMatrix> PreconditionerSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{
			UniversalPreconditioner(Cprimes[1], Cprime_perps[1], Ksqrtinvs[1], Lsqrtinvs[1], Vs[0]),
			UniversalPreconditioner(Cprimes[2], Cprime_perps[2], Ksqrtinvs[2], Lsqrtinvs[2], Vs[1])
		};
	};

	std::vector<EigenMatrix> PreconditionerInvSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{
			UniversalPreconditioner(Cprimes[1], Cprime_perps[1], Ksqrts[1], Lsqrts[1], Vs[0]),
			UniversalPreconditioner(Cprimes[2], Cprime_perps[2], Ksqrts[2], Lsqrts[2], Vs[1])
		};
	};
};

class ObjNewtonBase: public UniversalObjNewtonBase<ObjBase>{ public:
	std::vector<EigenMatrix> Kinvs = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::vector<EigenMatrix> Linvs = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };

	using UniversalObjNewtonBase<ObjBase>::UniversalObjNewtonBase;

	virtual void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		ObjBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) ){
			for ( int type = 1; type < 3; type++ ){
				Kinvs[type] = Ks[type].cwiseInverse();
				Linvs[type] = Ls[type].cwiseInverse();
			}
		}
	};

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> dCprimes) const override{
		dCprimes.insert(dCprimes.begin(), EigenZero(0, 0));
		std::vector<EigenMatrix> HdCprimes = UniversalObjNewtonBase<ObjBase>::Hessian(dCprimes);
		HdCprimes.erase(HdCprimes.begin());
		return HdCprimes;
	};

	std::vector<EigenMatrix> Preconditioner(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{
			UniversalPreconditioner(Cprimes[1], Cprime_perps[1], Kinvs[1], Linvs[1], Vs[0]),
			UniversalPreconditioner(Cprimes[2], Cprime_perps[2], Kinvs[2], Linvs[2], Vs[1])
		};
	};
};

using ObjNewton = UniversalObjNewton<ObjNewtonBase>;

using ObjARH = UniversalObjARH<ObjNewtonBase>;

} // namespace

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc1, int nocc2,
		EigenMatrix Z1, EigenMatrix Z2,
		int nthreads, int output){
	std::conditional_t< scf_t == lbfgs_t,
				ObjLBFGS,
				std::conditional_t< scf_t == newton_t,
							ObjNewton,
							ObjARH
				>
	> obj(int2c1e, int4c2e, xc, grid, {0, nocc1, nocc2}, 0, {EigenZero(0, 0), Z1, Z2}, nthreads);
	 Maniverse::Flag flag1(EigenOne(Z1.rows(), nocc1)); flag1.setBlockParameters({nocc1});
	 Maniverse::Flag flag2(EigenOne(Z2.rows(), nocc2)); flag2.setBlockParameters({nocc2});
	Maniverse::Iterate M(obj, {flag1.Share(), flag2.Share()}, 1);
	std::tuple<double, double, double> tol = {1.e-8, 1.e-5, 1.e-5};
	if constexpr ( scf_t == lbfgs_t ){
		if ( ! Maniverse::LBFGS(
					M, tol,
					20, 300, 0.1, 0.75, 10, output
		) ) throw std::runtime_error("Convergence failed!");
	}else{
		Maniverse::TrustRegion tr;
		if constexpr ( scf_t == newton_t ){
			if ( ! Maniverse::TruncatedNewton(
						M, tr, tol,
						0.001, 300, output
			) ) throw std::runtime_error("Convergence failed!");
		}else{
			if ( ! Maniverse::TruncatedNewton(
						M, tr, tol,
						0.01, 300, output
			) ) throw std::runtime_error("Convergence failed!");
		}
	}
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(obj.Fprimes[1]);
	const EigenVector eps1 = eigensolver.eigenvalues();
	const EigenMatrix C1 = Z1 * eigensolver.eigenvectors();
	eigensolver.compute(obj.Fprimes[2]);
	const EigenVector eps2 = eigensolver.eigenvalues();
	const EigenMatrix C2 = Z2 * eigensolver.eigenvectors();
	return std::make_tuple(obj.Value, eps1, eps2, C1, C2);
}

void U_SCF::Calculate0(){
	if ( scftype == "DRY" ) return;
	const int nocc1 = mwfn.getNumElec(1); 
	const int nocc2 = mwfn.getNumElec(2); 
	const EigenMatrix Z1 = mwfn.getCoefficientMatrix(1);
	const EigenMatrix Z2 = mwfn.getCoefficientMatrix(2);
	const EigenMatrix F1 = mwfn.getFock(1);
	const EigenMatrix F2 = mwfn.getFock(2);
	auto [E, eps1, eps2, C1, C2] =
		scftype == "DIIS" ? UnrestrictedDIIS(nocc1, nocc2, int2c1e, int4c2e, xc, grid, F1, F2, Z1, Z1, 1, nthreads) :
		scftype == "LBFGS" ? UnrestrictedRiemann<lbfgs_t>(int2c1e, int4c2e, xc, grid, nocc1, nocc2, Z1, Z2, nthreads, 1) :
		scftype == "ARH" ? UnrestrictedRiemann<arh_t>(int2c1e, int4c2e, xc, grid, nocc1, nocc2, Z1, Z2, nthreads, 1) :
		/* scftype == "NEWTON" ? */ UnrestrictedRiemann<newton_t>(int2c1e, int4c2e, xc, grid, nocc1, nocc2, Z1, Z2, nthreads, 1);
	Energy += E;
	mwfn.setEnergy(eps1, 1);
	mwfn.setEnergy(eps2, 2);
	mwfn.setCoefficientMatrix(C1, 1);
	mwfn.setCoefficientMatrix(C2, 2);
}
