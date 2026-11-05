#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cstdio>
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/Optimizer/Newton.h>

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

class ObjBase: public UniversalObjBase{ public:
	std::array<EigenVector, 3> epss = {
		EigenVector::Zero(0), EigenVector::Zero(0), EigenVector::Zero(0)
	};
	std::array<EigenMatrix, 3> Ccano_occs = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::array<EigenMatrix, 3> Ccano_virs = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::array<EigenMatrix, 3> Ks = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::array<EigenMatrix, 3> Ls = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };

	using UniversalObjBase::UniversalObjBase;

	virtual void Calculate(std::vector<EigenMatrix> Cs_, std::vector<int> derivatives) override{
		Cs_.insert(Cs_.begin(), EigenZero(0, 0));
		UniversalObjBase::Calculate(Cs_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Gradient = Gradients;
			Gradient.erase(Gradient.begin());
			Eigen::SelfAdjointEigenSolver<EigenMatrix> es;
			for ( int type = 1; type < 3; type++ ){
				epss[type].resize(nbasis);

				const EigenMatrix Focc = Cs[type].transpose() * Fs[type] * Cs[type];
				es.compute(Focc);
				Ccano_occs[type] = Cs[type] * es.eigenvectors();
				epss[type].head(Norbs[type]) = es.eigenvalues();

				Eigen::HouseholderQR<EigenMatrix> qr(Ssqrt * Cs[type]);
				const EigenMatrix Call = Ssqrtinv * qr.householderQ();
				const EigenMatrix C_perp = Call.rightCols(nbasis - Norbs[type]);

				const EigenMatrix Fvir = C_perp.transpose() * Fs[type] * C_perp;
				es.compute(Fvir);
				Ccano_virs[type] = C_perp * es.eigenvectors();
				epss[type].tail(nbasis - Norbs[type]) = es.eigenvalues();

				EigenMatrix A = EigenMatrix::Ones(nbasis, nbasis);
				for ( int o = 0; o < Norbs[type]; o++ ){
					for ( int v = Norbs[type]; v < nbasis; v++ ){
						A(o, v) = A(v, o) = 2 * std::abs( epss[type](v) - epss[type](o) );
					}
				}
				Ks[type] = A.topLeftCorner(Norbs[type], Norbs[type]);
				Ls[type] = A.bottomLeftCorner(nbasis - Norbs[type], Norbs[type]);
			}
		}
	};
};

class ObjLBFGS: public ObjBase{ public:
	std::array<EigenMatrix, 3> Ksqrts = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::array<EigenMatrix, 3> Ksqrtinvs = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::array<EigenMatrix, 3> Lsqrts = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::array<EigenMatrix, 3> Lsqrtinvs = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };

	using ObjBase::ObjBase;

	void Calculate(std::vector<EigenMatrix> Ds, std::vector<int> derivatives) override{
		ObjBase::Calculate(Ds, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			for ( int type = 1; type < 3; type++ ){
				Ksqrts[type] = Ks[type].cwiseSqrt();
				Ksqrtinvs[type] = Ksqrts[type].cwiseInverse();
				Lsqrts[type] = Ls[type].cwiseSqrt();
				Lsqrtinvs[type] = Lsqrts[type].cwiseInverse();
			}
		}
	};
};

class ObjNewtonBase: public UniversalObjNewtonBase<ObjBase>{ public:
	std::array<EigenMatrix, 3> Kinvs = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
	std::array<EigenMatrix, 3> Linvs = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };

	using UniversalObjNewtonBase<ObjBase>::UniversalObjNewtonBase;

	virtual void Calculate(std::vector<EigenMatrix> Cs_, std::vector<int> derivatives) override{
		ObjBase::Calculate(Cs_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) ){
			for ( int type = 1; type < 3; type++ ){
				Kinvs[type] = Ks[type].cwiseInverse();
				Linvs[type] = Ls[type].cwiseInverse();
			}
		}
	};

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> dCs) const override{
		dCs.insert(dCs.begin(), EigenZero(0, 0));
		std::vector<EigenMatrix> HdCs = UniversalObjNewtonBase<ObjBase>::Hessian(dCs);
		HdCs.erase(HdCs.begin());
		return HdCs;
	};
};

using ObjNewton = UniversalObjNewton<ObjNewtonBase>;

using ObjARH = UniversalObjARH<ObjNewtonBase>;

} // namespace

#define __Precon__(_P_, _K_, _L_){\
	M._P_ = [&obj, bp = M.BlockParameters](EigenVector V){\
		EigenVector PV(V.size());\
		for ( int i = 0; i < 2; i++ ){\
			MV_Block(PV, i, bp) = UniversalPreconditioner(\
				obj.Ccano_occs[i + 1], obj.Ccano_virs[i + 1], obj._K_[i + 1], obj._L_[i + 1],\
				obj.int2c1e->Overlap * MV_constBlock(V, i, bp)\
			);\
		}\
		return PV;\
	};\
}

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc1, int nocc2,
		EigenMatrix C1, EigenMatrix C2,
		int nthreads, int output){
	std::conditional_t< scf_t == lbfgs_t,
				ObjLBFGS,
				std::conditional_t< scf_t == newton_t,
							ObjNewton,
							ObjARH
				>
	> obj(int2c1e, int4c2e, xc, grid, {0, nocc1, nocc2}, 0, nthreads);
	Maniverse::Flag flag1(C1, S); flag1.setBlockParameters({nocc1});
	Maniverse::Flag flag2(C2, S); flag2.setBlockParameters({nocc2});
	Maniverse::Iterate M(obj, {flag1.Share(), flag2.Share()});
	std::array<double, 3> tol = {1.e-8, 1.e-5, 1.e-5};
	if constexpr ( scf_t == lbfgs_t ){
		__Precon__(PreconditionerSqrt, Ksqrtinvs, Lsqrtinvs);
		__Precon__(PreconditionerInvSqrt, Ksqrts, Lsqrts);
		if ( ! Maniverse::LBFGS(
					M, tol,
					20, 300, 0.1, 0.75, 10, output
		) ) throw std::runtime_error("Convergence failed!");
	}else{
		__Precon__(Preconditioner, Kinvs, Linvs);
		Maniverse::TrustRegion tr;
		static constexpr double ls_tol = scf_t == newton_t ? 0.001 : 0.01;
		Maniverse::ConjugateGradient cg(1, {ls_tol, ls_tol}, M.getDimension(), 1);
		Maniverse::initLinearSolverForNewton(cg, M);
		if ( ! Maniverse::Newton(
					M, tr, cg, tol, 300, output
		) ) throw std::runtime_error("Convergence failed!");
	}
	EigenMatrix Ca = EigenZero(obj.nbasis, obj.nbasis);
	Ca << obj.Ccano_occs[1], obj.Ccano_virs[1];
	EigenMatrix Cb = EigenZero(obj.nbasis, obj.nbasis);
	Cb << obj.Ccano_occs[2], obj.Ccano_virs[2];
	return std::make_tuple(obj.Value, obj.epss[1], obj.epss[2], Ca, Cb);
}

void U_SCF::Calculate0(){
	if ( scftype == "DRY" ) return;
	const int nocc1 = mwfn.getNumElec({.Set=0});
	const int nocc2 = mwfn.getNumElec({.Set=1}); 
	EigenMatrix Z1 = EigenZero(mwfn.getNumBasis(), mwfn.getNumBasis());
	EigenMatrix Z2 = EigenZero(mwfn.getNumBasis(), mwfn.getNumBasis());
	Z1 <<
		mwfn.getCoefficientMatrix({.Set=0, .Type=1, .OccUpper=1, .OccLower=1}),
		mwfn.getCoefficientMatrix({.Set=0, .Type=1, .OccUpper=0, .OccLower=0})
	;
	Z2 <<
		mwfn.getCoefficientMatrix({.Set=1, .Type=2, .OccUpper=1, .OccLower=1}),
		mwfn.getCoefficientMatrix({.Set=1, .Type=2, .OccUpper=0, .OccLower=0})
	;
	const EigenMatrix F1 = mwfn.getFock({.Set=0});
	const EigenMatrix F2 = mwfn.getFock({.Set=1});
	auto [E, eps1, eps2, C1, C2] =
		scftype == "DIIS" ? UnrestrictedDIIS(nocc1, nocc2, *int2c1e, *int4c2e, xc, grid, F1, F2, Z1, Z1, 1, nthreads) :
		scftype == "LBFGS" ? UnrestrictedRiemann<lbfgs_t>(*int2c1e, *int4c2e, xc, grid, nocc1, nocc2, Z1.leftCols(nocc1), Z2.leftCols(nocc2), nthreads, 1) :
		scftype == "ARH" ? UnrestrictedRiemann<arh_t>(*int2c1e, *int4c2e, xc, grid, nocc1, nocc2, Z1.leftCols(nocc1), Z2.leftCols(nocc2), nthreads, 1) :
		/* scftype == "NEWTON" ? */ UnrestrictedRiemann<newton_t>(*int2c1e, *int4c2e, xc, grid, nocc1, nocc2, Z1.leftCols(nocc1), Z2.leftCols(nocc2), nthreads, 1);
	Energy += E;
	mwfn.setEnergy(eps1, {.Set=0});
	mwfn.setEnergy(eps2, {.Set=1});
	mwfn.setCoefficientMatrix(C1, {.Set=0});
	mwfn.setCoefficientMatrix(C2, {.Set=1});
}
