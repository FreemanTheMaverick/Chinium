#include <Eigen/Dense>
#include <vector>
#include <tuple>
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
	EigenMatrix Cprime;
	EigenMatrix Cprime_perp;
	EigenMatrix C;
	EigenMatrix K;
	EigenMatrix L;
	std::vector<std::array<ObjDeterminant, 2>> lowers;
	std::vector<int> lowers_type;

	ObjBase(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		std::vector<int> Norbs, double Coupling,
		std::vector<EigenMatrix> Zs, int nthreads,
		std::vector<std::array<EigenMatrix, 2>> lowers_,
		std::vector<int> lowers_type
	): UniversalObjBase(int2c1e, int4c2e, xc, grid, Norbs, Coupling, Zs, nthreads), lowers_type(lowers_type){
		Lambda.resize(lowers_.size());
		lowers.clear();
		for ( std::array<EigenMatrix, 2>& lower : lowers_ ) lowers.push_back({
				ObjDeterminant(lower[0]),
				ObjDeterminant(lower[1])
		});
	};

	virtual void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		Cprime = Cprimes_[0];
		Cprimes_.clear();
		for ( int type = 0, col = 0; type < 3; type++ ){
			Cprimes_.push_back( Norbs[type] ? Cprime.middleCols(col, Norbs[type]).eval() : EigenZero(0, 0) );
			col += Norbs[type];
		}
		UniversalObjBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Gradient = { Cprime };
			for ( int type = 0, col = 0; type < 3; type++ ) if ( Norbs[type] ){
				Gradient[0].middleCols(col, Norbs[type]) = Gradients[type];
				col += Norbs[type];
			}

			const int Np = Norbs[0]; const int Na = Norbs[1]; const int Nb = Norbs[2];
			Eigen::HouseholderQR<EigenMatrix> qr(Cprime);
			const EigenMatrix Call = qr.householderQ();
			Cprime_perp = Call.rightCols(nbasis - Np - Na - Nb);
			std::vector<EigenMatrix> Fmos(4, EigenZero(nbasis, nbasis));
			for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
				Fmos[type] = Call.transpose() * Fprimes[type] * Call;
			}
			EigenMatrix A = EigenMatrix::Ones(nbasis, nbasis);
			for ( int i = 0; i < nbasis; i++ ){
				int I = 0; int Iscale = 4;
				if ( i < Np ){ I = 0; Iscale = 4; }
				else if ( i < Np + Na ){ I = 1; Iscale = 2; }
				else if ( i < Np + Na + Nb ) { I = 2; Iscale = 2; }
				else { I = 3; Iscale = 0; };
				for ( int j = 0; j < nbasis; j++ ){
					int J = 0; int Jscale = 4;
					if ( j < Np ){ J = 0; Jscale = 4; }
					else if ( j < Np + Na ){ J = 1; Jscale = 2; }
					else if ( j < Np + Na + Nb ){ J = 2; Jscale = 2; }
					else{ J = 3; Jscale = 0; }
					const double FIi = Fmos[I](i, i) * Iscale;
					const double FIj = Fmos[I](j, j) * Iscale;
					const double FJi = Fmos[J](i, i) * Jscale;
					const double FJj = Fmos[J](j, j) * Jscale;
					A(i, j) = std::abs(FIj + FJi - FIi - FJj);
					if ( A(i, j) < 0.1 ) A(i, j) = 0.1;
				}
			}
			K = A.topLeftCorner(Np + Na + Nb, Np + Na + Nb);
			L = A.bottomLeftCorner(nbasis - Np - Na - Nb, Np + Na + Nb);
		}

		// Orthogonality constraints
		EigenMatrix Ca( nbasis, Norbs[0] + Norbs[1] );
		if ( Norbs[0] ) Ca.leftCols(Norbs[0]) = Cprimes[0];
		if ( Norbs[1] ) Ca.rightCols(Norbs[1]) = Cprimes[1];
		EigenMatrix Cb( nbasis, Norbs[0] + Norbs[2] );
		if ( Norbs[0] ) Cb.leftCols(Norbs[0]) = Cprimes[0];
		if ( Norbs[2] ) Cb.rightCols(Norbs[2]) = Cprimes[2];
		Constraint_Value.resize(lowers.size());
		Constraint_Gradient.resize(lowers.size());

		for ( int icons = 0; icons < (int)lowers.size(); icons++ ){
			std::array<ObjDeterminant, 2>& lower = lowers[icons];
			lower[0].Calculate({Ca}, derivatives);
			lower[1].Calculate({Cb}, derivatives);
			if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
				if ( Norbs[0] && !Norbs[1] && !Norbs[2] && lowers_type[icons] == 0 ){ // The current and the lower states are both closed-shell configurations.
					Constraint_Value[icons] = lower[0].Value;
				// }else if (lowers_type[icons] == 2){ // No need to specify this for two-determinant lower states.
				}else Constraint_Value[icons] = lower[0].Value * lower[1].Value;
				Value += Lambda[icons] * Constraint_Value[icons] + Rho / 2 * std::pow(Constraint_Value[icons], 2);
			}
			if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
				EigenMatrix cons_grad = EigenZero(nbasis, Cprime.cols());
				if ( Norbs[0] && !Norbs[1] && !Norbs[2] && lowers_type[icons] == 0 ){ // The current and the lower states are both closed-shell configurations.
					cons_grad = lower[0].Gradient[0];
				}else{
					const EigenMatrix Ca_grad = lower[0].Gradient[0] * lower[1].Value;
					const EigenMatrix Cb_grad = lower[0].Value * lower[1].Gradient[0];
					cons_grad.leftCols( Norbs[0] + Norbs[1] ) += Ca_grad;
					cons_grad.leftCols( Norbs[0] ) += Cb_grad.leftCols( Norbs[0] );
					cons_grad.rightCols( Norbs[2] ) += Cb_grad.rightCols( Norbs[2] );
				}
				Constraint_Gradient[icons] = { cons_grad };
				Gradient[0] += ( Lambda[icons] + Rho * Constraint_Value[icons] ) * cons_grad;
			}
		}
	};
};

class ObjLBFGS: public ObjBase{ public:
	EigenMatrix Ksqrt, Ksqrtinv, Lsqrt, Lsqrtinv;

	using ObjBase::ObjBase;

	void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		ObjBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Ksqrt = K.cwiseSqrt();
			Ksqrtinv = Ksqrt.cwiseInverse();
			Lsqrt = L.cwiseSqrt();
			Lsqrtinv = Lsqrt.cwiseInverse();
		}
	};

	std::vector<EigenMatrix> PreconditionerSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{ UniversalPreconditioner(Cprime, Cprime_perp, Ksqrtinv, Lsqrtinv, Vs[0]) };
	};

	std::vector<EigenMatrix> PreconditionerInvSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{ UniversalPreconditioner(Cprime, Cprime_perp, Ksqrt, Lsqrt, Vs[0]) };
	};
};

class ObjNewtonBase: public UniversalObjNewtonBase<ObjBase>{ public:
	EigenMatrix Kinv, Linv;

	using UniversalObjNewtonBase<ObjBase>::UniversalObjNewtonBase;

	virtual void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		UniversalObjNewtonBase<ObjBase>::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) ){
			Kinv = K.cwiseInverse();
			Linv = L.cwiseInverse();
		}
	};

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> dCprimes) const override{
		const EigenMatrix dCprime = dCprimes[0];
		dCprimes.clear();
		for ( int type = 0, col = 0; type < 3; type++ ){
			dCprimes.push_back( Norbs[type] ? dCprime.middleCols(col, Norbs[type]).eval() : EigenZero(0, 0) );
			col += Norbs[type];
		}
		const std::vector<EigenMatrix> HdCprimes = UniversalObjNewtonBase<ObjBase>::Hessian(dCprimes);
		EigenMatrix HdCprime = Cprime;
		for ( int type = 0, col = 0; type < 3; type++ ) if ( Norbs[type] ){
			HdCprime.middleCols(col, Norbs[type]) = HdCprimes[type];
			col += Norbs[type];
		}

		EigenMatrix dCa( nbasis, Norbs[0] + Norbs[1] );
		if ( Norbs[0] ) dCa.leftCols(Norbs[0]) = dCprimes[0];
		if ( Norbs[1] ) dCa.rightCols(Norbs[1]) = dCprimes[1];
		EigenMatrix dCb( nbasis, Norbs[0] + Norbs[2] );
		if ( Norbs[0] ) dCb.leftCols(Norbs[0]) = dCprimes[0];
		if ( Norbs[2] ) dCb.rightCols(Norbs[2]) = dCprimes[2];
		for ( int icons = 0; icons < (int)lowers.size(); icons++ ){
			EigenMatrix cons_hess = EigenZero(nbasis, Cprime.cols());
			const std::array<ObjDeterminant, 2>& lower = lowers[icons];
			if ( Norbs[0] && !Norbs[1] && !Norbs[2] && lowers_type[icons] ){
				cons_hess = lower[0].Hessian({dCprime})[0];
			}else{
				const EigenMatrix Ca_hess = lower[0].Hessian({dCa})[0] * lower[1].Value + lower[0].Gradient[0] * lower[1].Gradient[0].cwiseProduct(dCb).sum();
				const EigenMatrix Cb_hess = lower[0].Gradient[0].cwiseProduct(dCa).sum() * lower[1].Gradient[0] + lower[0].Value * lower[1].Hessian({dCb})[0];
				cons_hess.leftCols( Norbs[0] + Norbs[1] ) += Ca_hess;
				cons_hess.leftCols( Norbs[0] ) += Cb_hess.leftCols( Norbs[0] );
				cons_hess.rightCols( Norbs[2] ) += Cb_hess.rightCols( Norbs[2] );
			}
			HdCprime += ( Lambda[icons] + Rho * Constraint_Value[icons] ) * cons_hess + Rho * Constraint_Gradient[icons][0].cwiseProduct(dCprime).sum() * Constraint_Gradient[icons][0];
		}
		return std::vector<EigenMatrix>{ HdCprime };
	};

	std::vector<EigenMatrix> Preconditioner(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{ UniversalPreconditioner(Cprime, Cprime_perp, Kinv, Linv, Vs[0]) };
	};
};

using ObjNewton = UniversalObjNewton<ObjNewtonBase>;

using ObjARH = UniversalObjARH<ObjNewtonBase>;

} // namespace

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenMatrix> RestrictedRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		std::vector<int> Norbs, double Coupling,
		EigenMatrix Z,
		std::vector<std::array<EigenMatrix, 2>> lowers_,
		std::vector<int> lowers_type,
		int nthreads, int output){
	std::conditional_t< scf_t == lbfgs_t,
				ObjLBFGS,
				std::conditional_t< scf_t == newton_t,
							ObjNewton,
							ObjARH
				>
	> obj(int2c1e, int4c2e, xc, grid, Norbs, Coupling, {Z, Z, Z}, nthreads, lowers_, lowers_type);
	Maniverse::Flag flag(EigenOne(Z.rows(), Norbs[0] + Norbs[1] + Norbs[2]));
	std::vector<int> space = {};
	for ( int Norb : Norbs ) if ( Norb > 0 ) space.push_back(Norb);
	flag.setBlockParameters(space);
	Maniverse::Iterate M(obj, {flag.Share()});
	const std::tuple<double, double, double> tol = {1.e-8, 1.e-5, 1.e-5};
	const std::vector<double> cons_tol(lowers_.size(), 1e-8);
	if constexpr ( scf_t == lbfgs_t ){
		if ( ! lowers_.size() && ! Maniverse::LBFGS(
					M, tol,
					20, 300, 0.1, 0.75, 10, output
		) ) throw std::runtime_error("Convergence failed!");
		if ( lowers_.size() && ! Maniverse::AugmentedLagrangian(1, 3.3, 0.8, cons_tol, 100, output)(Maniverse::LBFGS)(
					M, tol,
					20, 300, 0.1, 0.75, 10, output
		) ) throw std::runtime_error("Convergence failed!");
	}else{
		Maniverse::TrustRegion tr;
		static constexpr double ls_tol = scf_t == newton_t ? 0.001 : 0.01;
		Maniverse::ConjugateGradient cg(M, 0, 1, {ls_tol, ls_tol}, M.getDimension(), output);
		if ( ! lowers_.size() && ! mv::Newton(
					M, tr, cg, tol, 300, output
		) ) throw std::runtime_error("Convergence failed!");
		if ( lowers_.size() && ! Maniverse::AugmentedLagrangian(1, 3.3, 0.8, cons_tol, 100, output)(mv::Newton)(
					M, tr, cg, tol, 300, output
		) ) throw std::runtime_error("Convergence failed!");
	}

	EigenVector eps = EigenZero(Z.cols(), 1);
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	if ( Norbs[0] && !Norbs[1] && !Norbs[2] ){
		const EigenMatrix Focc = obj.Cprime.transpose() * obj.Fprimes[0] * obj.Cprime;
		eigensolver.compute(Focc);
		eps.head(Norbs[0]) = eigensolver.eigenvalues();
		C.leftCols(Norbs[0]) = Z * obj.Cprime * eigensolver.eigenvectors();
		const EigenMatrix Fvir = obj.Cprime_perp.transpose() * obj.Fprimes[0] * obj.Cprime_perp;
		eigensolver.compute(Fvir);
		eps.tail(Z.cols() - Norbs[0]) = eigensolver.eigenvalues();
		C.rightCols(Z.cols() - Norbs[0]) = Z * obj.Cprime_perp * eigensolver.eigenvectors();
	}else if ( ( Norbs[1] && !Norbs[2] ) || ( !Norbs[1] && Norbs[2] ) ){ // Guest and Saunders averaged Fock matrix
		const int nd = Norbs[0];
		const int ns = Norbs[1] ? Norbs[1] : Norbs[2];
		const int nv = obj.nbasis - nd - ns;
		EigenMatrix Cp_all = EigenZero(obj.nbasis, obj.nbasis);
		Cp_all << obj.Cprime, obj.Cprime_perp;
		EigenMatrix Fd = Cp_all.transpose() * obj.Fprimes[0] * Cp_all;
		EigenMatrix Fs = 0.5 * Cp_all.transpose() * obj.Fprimes[1] * Cp_all;
		EigenMatrix bigF = EigenZero(obj.nbasis, obj.nbasis);

		// Diagonal
		bigF.block(0, 0, nd, nd) = Fd.block(0, 0, nd, nd);
		bigF.block(nd, nd, ns, ns) = Fd.block(nd, nd, ns, ns);
		bigF.block(nd + ns, nd + ns, nv, nv) = Fd.block(nd + ns, nd + ns, nv, nv);

		// Off-diagonal
		bigF.block(0, nd, nd, ns) = Fd.block(0, nd, nd, ns) - Fs.block(0, nd, nd, ns);
		bigF.block(nd, 0, ns, nd) = bigF.block(0, nd, nd, ns).transpose();
		bigF.block(0, nd + ns, nd, nv) = Fd.block(0, nd + ns, nd, nv);
		bigF.block(nd + ns, 0, nv, nd) = bigF.block(0, nd + ns, nd, nv).transpose();
		bigF.block(nd, nd + ns, ns, nv) = Fs.block(nd, nd + ns, ns, nv);
		bigF.block(nd + ns, nd, nv, ns) = bigF.block(nd, nd + ns, ns, nv).transpose();
		bigF = ( Cp_all * bigF * Cp_all.transpose() ).eval();
		Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

		// Separate diagonalization
		const EigenMatrix Fdocc = obj.Cprime.leftCols(nd).transpose() * bigF * obj.Cprime.leftCols(nd);
		eigensolver.compute(Fdocc);
		eps.head(nd) = eigensolver.eigenvalues();
		C.leftCols(nd) = Z * obj.Cprime.leftCols(nd) * eigensolver.eigenvectors();
		const EigenMatrix Fsocc = obj.Cprime.rightCols(ns).transpose() * bigF * obj.Cprime.rightCols(ns);
		eigensolver.compute(Fsocc);
		eps.segment(nd, ns) = eigensolver.eigenvalues();
		C.middleCols(nd, ns) = Z * obj.Cprime.rightCols(ns) * eigensolver.eigenvectors();
		const EigenMatrix Fvir = obj.Cprime_perp.transpose() * bigF * obj.Cprime_perp;
		eigensolver.compute(Fvir);
		eps.tail(nv) = eigensolver.eigenvalues();
		C.rightCols(nv) = Z * obj.Cprime_perp * eigensolver.eigenvectors();
	}else C << Z * obj.Cprime, Z * obj.Cprime_perp;
	return std::make_tuple(obj.Value, eps, C);
}

bool RestrictedStability(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		std::vector<int> Norbs, double Coupling,
		EigenMatrix Z,
		std::vector<std::array<EigenMatrix, 2>> lowers_,
		std::vector<int> lowers_type,
		int stable,
		int nthreads, int output){
	ObjNewton obj(int2c1e, int4c2e, xc, grid, Norbs, Coupling, {Z, Z, Z}, nthreads, lowers_, lowers_type);
	Maniverse::Flag flag(EigenOne(Z.rows(), Norbs[0] + Norbs[1] + Norbs[2]));
	std::vector<int> space = {};
	for ( int Norb : Norbs ) if ( Norb > 0 ) space.push_back(Norb);
	flag.setBlockParameters(space);
	Maniverse::Iterate M(obj, {flag.Share()});
	M.Func->Calculate(M.getPoint(), {0, 1, 2});
	M.setGradient();
	if ( lowers_.size() ) obj.Lambda = M.getEffectiveLambda();
	const auto [Evals, Evecs] = Maniverse::Lanczos(M, stable, 0, lowers_.size() > 0, output);
	return Evals[0] > 0;
}

void R_SCF::Calculate0(){
	if ( scftype == "DRY" ) return;
	EigenMatrix Z = EigenZero(mwfn.getNumBasis(), mwfn.getNumIndBasis());
	Z <<
		mwfn.getCoefficientMatrix({.Set=0, .OccUpper=2, .OccLower=2}),
		mwfn.getCoefficientMatrix({.Set=0, .Type=1, .OccUpper=1, .OccLower=1}),
		mwfn.getCoefficientMatrix({.Set=0, .Type=2, .OccUpper=1, .OccLower=1}),
		mwfn.getCoefficientMatrix({.Set=0, .OccUpper=0, .OccLower=0})
	;
	const EigenMatrix F = mwfn.getFock({.Set=0});
	auto [E, eps, C] =
		scftype == "DIIS" ? RestrictedDIIS(Np, int2c1e, int4c2e, xc, grid, F, Z, 1, nthreads) :
		scftype == "LBFGS" ? RestrictedRiemann<lbfgs_t>(int2c1e, int4c2e, xc, grid, {Np, Na, Nb}, Coupling, Z, lowers, lowers_type, nthreads, 1) :
		scftype == "ARH" ? RestrictedRiemann<arh_t>(int2c1e, int4c2e, xc, grid, {Np, Na, Nb}, Coupling, Z, lowers, lowers_type, nthreads, 1) :
		/* scftype == "NEWTON" ? */ RestrictedRiemann<newton_t>(int2c1e, int4c2e, xc, grid, {Np, Na, Nb}, Coupling, Z, lowers, lowers_type, nthreads, 1);
	Energy += E;
	mwfn.setEnergy(eps, {.Set=0});
	mwfn.setCoefficientMatrix(C, {.Set=0});
	EigenVector occ = EigenZero(mwfn.getNumIndBasis(), 1);
	occ.head(Np).setConstant(2);
	occ.segment(Np, Na + Nb).setConstant(1);
	mwfn.setOccupation(occ, {.Set=0});
	for ( int iorb = 0; iorb < mwfn.getNumIndBasis(); iorb++ ) mwfn.Orbitals[0][iorb].Type = ( Np + Na > iorb && iorb >= Np ) ? 1 : ( Np + Na + Nb > iorb && iorb >= Np + Na ) ? 2 : 0;
	if ( stable > 0 ) RestrictedStability(int2c1e, int4c2e, xc, grid, {Np, Na, Nb}, Coupling, C, lowers, lowers_type, stable, nthreads, 1);
}
