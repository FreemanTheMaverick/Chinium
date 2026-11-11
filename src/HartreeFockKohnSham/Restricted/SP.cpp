#include <Eigen/Dense>
#include <vector>
#include <array>
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/Optimizer/AugmentedLagrangian.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/Optimizer/Newton.h>
#include <Maniverse/Diagonalizer/Lanczos.h>

#include "../../Macro.h"
#include "../../DIIS.h"

#include "../Universal.h"
#include "../Restricted.h"
#include "../Determinant.h"

namespace{

#define S ( int2c1e.Overlap)
#define Hcore ( int2c1e.Kinetic + int2c1e.Nuclear )

std::tuple<double, EigenVector, EigenMatrix> RestrictedDIIS(
		int nocc,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix F, EigenMatrix Z,
		int output, int nthreads){
	double oldE = 0;
	double E = 0;
	const int nbasis = F.cols();
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	std::function<std::tuple<
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
			>(std::vector<EigenMatrix>&, std::vector<bool>&)
	> update_func = [&](std::vector<EigenMatrix>& Fs_, std::vector<bool>&){
		const EigenMatrix F_ = Fs_[0];
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z;
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		oldE = E;
		E = 0;
		const EigenMatrix D_ = C.leftCols(nocc) * C.leftCols(nocc).transpose();
		const auto [J_, K_, __, ___] = int4c2e.ContractInts(D_, EigenZero(0, 0), EigenZero(0, 0), nthreads, 1);
		const EigenMatrix Ghf_ = J_ - K_;
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(nbasis, nbasis);
		if (xc){
			grid.getDensity({2 * D_});
			xc.Evaluate("ev", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock()[0];
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix Fnew_ = Fhf_ + Gxc_;
		E += D_.cwiseProduct( Hcore + Fhf_ ).sum() + Exc_;
		if (output>0){
			std::printf("Electronic energy = %.10f\n", E);
			std::printf("Changed by %E from the last step\n", E - oldE);
		}
		EigenMatrix G_ = 2 * ( Fnew_ * D_ * S - S * D_ * Fnew_ );
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
	ADIIS adiis(&update_func, 1, 20, 1e-1, 300, output>0 ? 2 : 0);
	if ( !adiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	CDIIS cdiis(&update_func, 1, 20, 1e-6, 300, output>0 ? 2 : 0);
	cdiis.Steal(adiis);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilons, C);
}

class ObjBase: public UniversalObjBase{ public:
	ObjBase(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		std::vector<int> Norbs, double Coupling,
		int nthreads
	): UniversalObjBase(int2c1e, int4c2e, xc, grid, Norbs, Coupling, nthreads){
		Ccanos.push_back(EigenZero(nbasis, nbasis - Norbs[0] - Norbs[1] - Norbs[2]));
		epss.push_back((EigenZero(nbasis - Norbs[0] - Norbs[1] - Norbs[2], 1)));
	};

	std::vector<EigenMatrix> decouple(EigenMatrix mat) const{
		std::vector<EigenMatrix> mats(3, EigenZero(nbasis, 0));
		for ( int i = 0, col = 0; i < 3; i++ ){
			mats[i] = mat.middleCols(col, Norbs[i]);
			col += Norbs[i];
		}
		return mats;
	};
		
	void Calculate(std::vector<EigenMatrix> Cs_, std::vector<int> derivatives) override{
		Cs = decouple(Cs_[0]);
		UniversalObjBase::CalculateAllCs(derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
		    Gradient = { concatenate_mat(Gradients) };
		}
	};
};

using ObjLBFGS = UniversalObjLBFGS<ObjBase>;

class ObjNewtonBase: public UniversalObjNewtonBase<ObjBase>{ public:
	using UniversalObjNewtonBase<ObjBase>::UniversalObjNewtonBase;

	virtual std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> dCs_) const override{
		const std::vector<EigenMatrix> dCs = decouple(dCs_[0]);
		return { concatenate_mat(UniversalObjNewtonBase::Hessian(dCs)) };
	};
};

using ObjNewton = UniversalObjNewton<ObjNewtonBase>;

using ObjARH = UniversalObjARH<ObjNewtonBase>;

class Cons: public Maniverse::Function{ public:
	std::vector<int> Norbs;
	int nbasis = 0;
	std::vector<ObjDeterminant> lower;

	Cons(
		std::vector<int> Norbs,
		EigenMatrix overlap,
		std::vector<EigenMatrix> lower_
	): Norbs(Norbs), nbasis(overlap.rows()){
		lower.clear();
		for ( EigenMatrix& Clower : lower_ ) lower.push_back({
				ObjDeterminant(overlap * Clower)
		});
		Gradient = { EigenZero(nbasis, Norbs[0] + Norbs[1] + Norbs[2]) };
	};

	virtual void Calculate(std::vector<EigenMatrix> Cs_, std::vector<int> derivatives) override{
		std::array<EigenMatrix, 2> Cs = {
			Cs_[0].leftCols(Norbs[0] + Norbs[1]),
			Cs_[0].leftCols(Norbs[0] + Norbs[2])
		};
		Cs[1].rightCols(Norbs[2]);
		for ( int i = 0; i < (int)lower.size(); i++ ){
			lower[i].Calculate({Cs[i]}, derivatives);
		}
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			if ( lower.size() == 1 ) Value = lower[0].Value;
			else Value = lower[0].Value * lower[1].Value;
		}
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			if ( lower.size() == 1 ) Gradient = lower[0].Gradient;
			else{
				const EigenMatrix Ga = lower[0].Gradient[0] * lower[1].Value;
				const EigenMatrix Gb = lower[0].Value * lower[1].Gradient[0];
				Gradient[0].leftCols(Norbs[0] + Norbs[1]) = Ga;
				Gradient[0].leftCols(Norbs[0]) += Gb.leftCols(Norbs[0]);
				Gradient[0].rightCols(Norbs[2]) = Gb.rightCols(Norbs[2]);
			}
		}
	};

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> dCs_) const override{
		std::array<EigenMatrix, 2> dCs = {
			dCs_[0].leftCols(Norbs[0] + Norbs[1]),
			dCs_[0].leftCols(Norbs[0] + Norbs[2])
		};
		dCs[1].rightCols(Norbs[2]) = dCs_[0].rightCols(Norbs[2]);
		EigenMatrix cons_hess = EigenZero(nbasis, dCs_[0].cols());
		if ( lower.size() == 1 ) cons_hess = lower[0].Hessian({dCs[0]})[0];
		else{
			const EigenMatrix Ca_hess = (
				lower[0].Hessian({dCs[0]})[0] * lower[1].Value
				+ lower[0].Gradient[0] * lower[1].Gradient[0].cwiseProduct(dCs[1]).sum()
			);
			const EigenMatrix Cb_hess = (
				lower[0].Gradient[0].cwiseProduct(dCs[0]).sum() * lower[1].Gradient[0]
				+ lower[0].Value * lower[1].Hessian({dCs[1]})[0]
			);
			cons_hess.leftCols( Norbs[0] + Norbs[1] ) += Ca_hess;
			cons_hess.leftCols( Norbs[0] ) += Cb_hess.leftCols( Norbs[0] );
			cons_hess.rightCols( Norbs[2] ) += Cb_hess.rightCols( Norbs[2] );
		}
		return std::vector<EigenMatrix>{ cons_hess };
	};
};

} // namespace

#define __Precon__(_P_, _K_, _L_){\
	M._P_ = [&obj, bp = M.BlockParameters](EigenVector V){\
		EigenVector PV(V.size());\
		MV_Block(PV, 0, bp) = UniversalPreconditioner(\
			concatenate_mat({obj.Ccanos[0], obj.Ccanos[1], obj.Ccanos[2]}),\
			obj.Ccanos[3],\
			obj._K_[0], obj._L_[0],\
			obj.int2c1e->Overlap * MV_constBlock(V, 0, bp)\
		);\
		return PV;\
	};\
}

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenMatrix> RestrictedRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		std::vector<int> Norbs, double Coupling,
		EigenMatrix C,
		std::vector<std::vector<EigenMatrix>> lowers_,
		int nthreads, int output){
	std::conditional_t< scf_t == lbfgs_t,
				ObjLBFGS,
				std::conditional_t< scf_t == newton_t,
							ObjNewton,
							ObjARH
				>
	> obj(int2c1e, int4c2e, xc, grid, Norbs, Coupling, nthreads);
	Maniverse::Flag flag(C, S);
	std::vector<int> space = {};
	for ( int Norb : Norbs ) if ( Norb > 0 ) space.push_back(Norb);
	flag.setBlockParameters(space);
	std::vector<Cons> lowers;
	for ( std::vector<EigenMatrix>& lower_ : lowers_ ) lowers.emplace_back(Norbs, S, lower_);
	std::vector<Maniverse::Function*> lowers_ptr;
	for ( Cons& lower : lowers ) lowers_ptr.push_back(&lower);
	Maniverse::Iterate M(obj, {flag.Share()}, lowers_ptr);
	const std::array<double, 3> tol = {1.e-8, 1.e-5, 1.e-5};
	const std::vector<double> cons_tol(lowers_.size(), 1e-10);
	if constexpr ( scf_t == lbfgs_t ){
		__Precon__(PreconditionerSqrt, Ksqrtinvs, Lsqrtinvs);
		__Precon__(PreconditionerInvSqrt, Ksqrts, Lsqrts);
		if ( ! lowers_.size() && ! Maniverse::LBFGS(
					M, tol,
					20, 300, 0.1, 0.75, 10, output
		) ) throw std::runtime_error("Convergence failed!");
		if ( lowers_.size() && ! Maniverse::AugmentedLagrangian(1, 3.3, 0.8, cons_tol, 100, output)(Maniverse::LBFGS)(
					M, tol,
					20, 300, 0.1, 0.75, 10, output
		) ) throw std::runtime_error("Convergence failed!");
	}else{
		__Precon__(Preconditioner, Kinvs, Linvs);
		Maniverse::TrustRegion tr;
		static constexpr double ls_tol = scf_t == newton_t ? 0.001 : 0.01;
		Maniverse::ConjugateGradient cg(1, {ls_tol, ls_tol}, M.getDimension(), output);
		Maniverse::initLinearSolverForNewton(cg, M);
		if ( ! lowers_.size() && ! mv::Newton(
					M, tr, cg, tol, 300, output
		) ) throw std::runtime_error("Convergence failed!");
		if ( lowers_.size() && ! Maniverse::AugmentedLagrangian(1, 3.3, 0.8, cons_tol, 100, output)(mv::Newton)(
					M, tr, cg, tol, 300, output
		) ) throw std::runtime_error("Convergence failed!");
	}

	return std::make_tuple(obj.Value, concatenate_vec(obj.epss), concatenate_mat(obj.Ccanos));
}

bool RestrictedStability(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		std::vector<int> Norbs, double Coupling,
		EigenMatrix C,
		std::vector<std::vector<EigenMatrix>> lowers_,
		int stable,
		int nthreads, int output){
	ObjNewton obj(int2c1e, int4c2e, xc, grid, Norbs, Coupling, nthreads);
	Maniverse::Flag flag(C, S);
	std::vector<int> space = {};
	for ( int Norb : Norbs ) if ( Norb > 0 ) space.push_back(Norb);
	flag.setBlockParameters(space);
	std::vector<Cons> lowers;
	for ( std::vector<EigenMatrix>& lower_ : lowers_ ) lowers.emplace_back(Norbs, S, lower_);
	std::vector<Maniverse::Function*> lowers_ptr;
	for ( Cons& lower : lowers ) lowers_ptr.push_back(&lower);
	Maniverse::Iterate M(obj, {flag.Share()}, lowers_ptr);
	M.Calculate(M.getPoint(), {0, 1, 2});
	M.setGradient();
	if ( lowers_.size() > 0 ) M.setLambda(M.calcLambda());
	const auto [Evals, Evecs] = Maniverse::Lanczos(M, stable, 0, lowers_.size() > 0, output);
	if ( output > 0 && Evals[0] < 0 ){
		std::printf("Non-positive curvature:\n");
		const EigenMatrix Y = S * C;
		const EigenMatrix X = Eigen::Map<const Eigen::MatrixXd>(Evecs[0].data(), C.rows(), Norbs[0] + Norbs[1] + Norbs[2]);
		EigenMatrix Omega = X * Y.transpose() - Y * X.transpose() + Y * ( Y.transpose() * X ) * Y.transpose();
		for ( int i = 0; i < 3; i++ ){
			int maxRow, maxCol;
			Omega.cwiseAbs().maxCoeff(&maxRow, &maxCol);
			std::printf("%d <-> %d  %f\n", maxCol, maxRow, Omega(maxRow, maxCol));
			Omega(maxRow, maxCol) = Omega(maxCol, maxRow) = 0;
		}
	}
	return Evals[0] > 0;
}

void R_SCF::Calculate0(){
	if ( scftype == "DRY" ) return;
	EigenMatrix Z = EigenZero(mwfn.getNumBasis(), mwfn.getNumBasis());
	Z <<
		mwfn.getCoefficientMatrix({.Set=0, .OccUpper=2, .OccLower=2}),
		mwfn.getCoefficientMatrix({.Set=0, .Type=1, .OccUpper=1, .OccLower=1}),
		mwfn.getCoefficientMatrix({.Set=0, .Type=2, .OccUpper=1, .OccLower=1}),
		mwfn.getCoefficientMatrix({.Set=0, .OccUpper=0, .OccLower=0})
	;
	const EigenMatrix F = mwfn.getFock({.Set=0});
	auto [E, eps, C] =
		scftype == "DIIS" ? RestrictedDIIS(Np, *int2c1e, *int4c2e, xc, grid, F, Z, 1, nthreads) :
		scftype == "LBFGS" ? RestrictedRiemann<lbfgs_t>(*int2c1e, *int4c2e, xc, grid, {Np, Na, Nb}, Coupling, Z.leftCols(Np + Na + Nb), lowers, nthreads, 1) :
		scftype == "ARH" ? RestrictedRiemann<arh_t>(*int2c1e, *int4c2e, xc, grid, {Np, Na, Nb}, Coupling, Z.leftCols(Np + Na + Nb), lowers, nthreads, 1) :
		/* scftype == "NEWTON" ? */ RestrictedRiemann<newton_t>(*int2c1e, *int4c2e, xc, grid, {Np, Na, Nb}, Coupling, Z.leftCols(Np + Na + Nb), lowers, nthreads, 1);
	Energy += E;
	mwfn.setEnergy(eps, {.Set=0});
	mwfn.setCoefficientMatrix(C, {.Set=0});
	EigenVector occ = EigenZero(mwfn.getNumBasis(), 1);
	occ.head(Np).setConstant(2);
	occ.segment(Np, Na + Nb).setConstant(1);
	mwfn.setOccupation(occ, {.Set=0});
	for ( int iorb = 0; iorb < mwfn.getNumBasis(); iorb++ ) mwfn.Orbitals[0][iorb].Type = ( Np + Na > iorb && iorb >= Np ) ? 1 : ( Np + Na + Nb > iorb && iorb >= Np + Na ) ? 2 : 0;
	if ( stable > 0 ) RestrictedStability(*int2c1e, *int4c2e, xc, grid, {Np, Na, Nb}, Coupling, C.leftCols(Np + Na + Nb), lowers, stable, nthreads, 1);
}
