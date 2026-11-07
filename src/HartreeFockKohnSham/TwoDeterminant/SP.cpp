#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <array>
#include <type_traits>
#include <cstdio>
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/Optimizer/AugmentedLagrangian.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/Optimizer/Newton.h>
#include <Maniverse/Diagonalizer/Lanczos.h>

#include "../../Macro.h"
#include "../../Integral.h"
#include "../../Grid.h"
#include "../../ExchangeCorrelation.h"

#include "../Universal.h"
#include "../AugmentedRoothaanHall.h"
#include "../TwoDeterminant.h"
#include "../Determinant.h"

#define S ( int2c1e->Overlap )
#define Hcore ( int2c1e->Kinetic + int2c1e->Nuclear )

namespace{

#define One ( grid2->SubGridBatches.size() > 0 )
#define __Make_Block_View__(mat, mats){\
	if constexpr (std::is_same_v<std::decay_t<decltype(mat)>, EigenVector>) \
		mats = {mat.segment(0, Np), mat.segment(Np, 1), mat.segment(Np + 1, 1)}; \
	else \
		mats = {mat.middleCols(0, Np), mat.middleCols(Np, 1), mat.middleCols(Np + 1, 1)}; \
}
// #define lambda ( One ? 0 : 1 )
#define lambda 0

class ObjBase: public Maniverse::Function{ public:
	Int2C1E* int2c1e;
	Int4C2E* int4c2e;
	EigenMatrix Ssqrt;
	EigenMatrix Ssqrtinv;
	ExchangeCorrelation* xc;
	Grid* grid;
	Grid* grid2;

	int Np;
	std::array<int, 3> Norbs;
	int nbasis;
	std::array<int, 3> occ = {2, 1, 1};

	int nthreads;

	EigenMatrix C;
	std::vector<Eigen::Ref<EigenMatrix>> Cs;
	std::vector<Eigen::Ref<EigenMatrix>> Gradients;
	std::vector<EigenMatrix> Ds;
	std::vector<EigenMatrix> Fs;
	EigenMatrix K;
	EigenMatrix L;

	EigenMatrix Ccano;
	EigenVector eps;
	std::vector<Eigen::Ref<EigenMatrix>> Ccanos;
	std::vector<Eigen::Ref<EigenVector>> epss;

	ObjBase(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid, Grid& grid2,
		int Np, int nthreads
	): int2c1e(&int2c1e), int4c2e(&int4c2e), xc(&xc), grid(&grid), grid2(&grid2), Np(Np), nthreads(nthreads){
		Eigen::SelfAdjointEigenSolver<EigenMatrix> es(int2c1e.Overlap);
		Ssqrt = es.operatorSqrt();
		Ssqrtinv = es.operatorInverseSqrt();
		nbasis = Ssqrt.rows();
		Norbs = { Np, 1, 1 };
	
		C = EigenZero(nbasis, Np + 2);
		Gradient = {C};
		__Make_Block_View__(C, Cs);
		__Make_Block_View__(Gradient[0], Gradients);
		Ds = Fs = { EigenZero(0, 0), EigenZero(0, 0), EigenZero(0, 0) };
		Ccano.resize(nbasis, nbasis);
		eps.resize(nbasis);
		__Make_Block_View__(Ccano, Ccanos);
		__Make_Block_View__(eps, epss);
		Ccanos.push_back(Ccano.rightCols(nbasis - Np - 2));
		epss.push_back(eps.tail(nbasis - Np - 2));
	};

	virtual void Calculate(std::vector<EigenMatrix> Cs_, std::vector<int> derivatives) override{
		C = Cs_[0];
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			for ( int type = 0; type < 3; type++ ){
				Ds[type] = Cs[type] * Cs[type].transpose();
			}
			const auto [J, Kd, Ka, Kb] = int4c2e->ContractInts(Ds[0], Ds[1], Ds[2], nthreads, 1);
			const std::vector<EigenMatrix> FhfMs = {
				Hcore + J - Kd - 0.5 * ( Ka + Kb ),
				Hcore + J - Kd - Ka + Kb * lambda,
				Hcore + J - Kd + Ka * lambda - Kb
			};
			const std::vector<EigenMatrix> FhfTs = {
				Hcore + J - Kd - 0.5 * ( Ka + Kb ),
				Hcore + J - Kd - Ka - Kb,
				Hcore + J - Kd - Ka - Kb
			};
			Value = 0;
			for ( int type = 0; type < 3; type++ ){
				Fs[type] = 2 * FhfMs[type] - FhfTs[type];
				Value += 0.5 * occ[type] * Ds[type].cwiseProduct( Hcore + Fs[type] ).sum();
			}

			if (*xc){
				if (One){
					grid->getDensity({ Ds[0] + Ds[1], Ds[0] + Ds[2] });
					grid2->getDensity({ Ds[0] + Ds[1] + Ds[2], Ds[0] });
					xc->Evaluate("ev", *grid);
					xc->Evaluate("ev", *grid2);
					Value += 2 * grid->getEnergy() - grid2->getEnergy();
					std::vector<EigenMatrix> GxcMs = grid->getFock();
					GxcMs = {
						0.5 * ( GxcMs[0] + GxcMs[1] ),
						GxcMs[0],
						GxcMs[1]
					};
					std::vector<EigenMatrix> GxcTs = grid2->getFock();
					GxcTs = {
						0.5 * ( GxcTs[0] + GxcTs[1] ),
						GxcTs[0],
						GxcTs[0]
					};
					for ( int type = 0; type < 3; type++ ){
						Fs[type] += 2 * GxcMs[type] - GxcTs[type];
					}
				}else{
					grid->getDensity({ 2 * Ds[0] + Ds[1] + Ds[2] });
					xc->Evaluate("ev", *grid);
					Value += grid->getEnergy();
					const EigenMatrix Gxc = grid->getFock()[0];
					for ( int type = 0; type < 3; type++ ){
						Fs[type] += Gxc;
					}
				}
			}
		}

		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			for ( int type = 0; type < 3; type++ ){
				Gradients[type] = 2 * occ[type] * Fs[type] * Cs[type];
			}

			Eigen::SelfAdjointEigenSolver<EigenMatrix> es;
			for ( int type = 0; type < 3; type++ ){
				const Eigen::Ref<EigenMatrix> Ct = Cs[type];
				const EigenMatrix Ft = Ct.transpose() * Fs[type] * Ct;
				es.compute(Ft);
				Ccanos[type] = Ct * es.eigenvectors();
				epss[type] = es.eigenvalues();
			}
			{
				Eigen::HouseholderQR<EigenMatrix> qr(Ssqrt * C);
				const EigenMatrix C_perp = (Ssqrtinv * qr.householderQ()).rightCols(nbasis - Np - 2);
				const EigenMatrix Fvir = C_perp.transpose() * Fs[0] * C_perp;
				es.compute(Fvir);
				Ccanos[3] = C_perp * es.eigenvectors();
				epss[3] = es.eigenvalues();
			}
			std::vector<EigenVector> Fmodiags(4, EigenZero(nbasis, 1));
			for ( int type = 0; type < 3; type++ ){
				Fmodiags[type] = ( Ccano.transpose() * Fs[type] * Ccano ).diagonal();
			}
			EigenMatrix A = EigenZero(nbasis, nbasis);
			for ( int i = 0; i < nbasis; i++ ){
				int I = 0; int Iscale = 4;
				if ( i < Np ){ I = 0; Iscale = 4; }
				else if ( i < Np + 1 ){ I = 1; Iscale = 2; }
				else if ( i < Np + 2 ) { I = 2; Iscale = 2; }
				else { I = 3; Iscale = 0; };
				for ( int j = 0; j < nbasis; j++ ){
					int J = 0; int Jscale = 4;
					if ( j < Np ){ J = 0; Jscale = 4; }
					else if ( j < Np + 1 ){ J = 1; Jscale = 2; }
					else if ( j < Np + 2 ){ J = 2; Jscale = 2; }
					else{ J = 3; Jscale = 0; }
					if ( I == J ) continue;
					const double FIi = Fmodiags[I](i) * Iscale;
					const double FIj = Fmodiags[I](j) * Iscale;
					const double FJi = Fmodiags[J](i) * Jscale;
					const double FJj = Fmodiags[J](j) * Jscale;
					A(i, j) = std::abs(FIj + FJi - FIi - FJj);
				}
			}
			K = A.topLeftCorner(Np + 2, Np + 2);
			L = A.bottomLeftCorner(nbasis - Np - 2, Np + 2);
		}
	};
};

class ObjLBFGS: public ObjBase{ public:
	EigenMatrix Ksqrt, Ksqrtinv, Lsqrt, Lsqrtinv;

	using ObjBase::ObjBase;

	void Calculate(std::vector<EigenMatrix> Cs_, std::vector<int> derivatives) override{
		ObjBase::Calculate(Cs_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Ksqrt = K.cwiseSqrt();
			Ksqrtinv = Ksqrt.unaryExpr(SafeInverse);
			Lsqrt = L.cwiseSqrt();
			Lsqrtinv = Lsqrt.unaryExpr(SafeInverse);
		}
	};
};

class ObjNewtonBase: public UniversalObjNewtonBase<ObjBase>{ public:
	EigenMatrix Kinv, Linv;

	using UniversalObjNewtonBase<ObjBase>::UniversalObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> Cs_, std::vector<int> derivatives) override{
		ObjBase::Calculate(Cs_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) ){
			Kinv = K.unaryExpr(SafeInverse);
			Linv = L.unaryExpr(SafeInverse);
		}
	};

	virtual std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDprimes) const = 0;

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> dCs_) const override{
		std::vector<Eigen::Ref<EigenMatrix>> dCs;
		__Make_Block_View__(dCs_[0], dCs);
		std::vector<EigenMatrix> dDs(3, EigenZero(nbasis, nbasis));
		for ( int type = 0; type < 3; type++ ){
			dDs[type] = Cs[type] * dCs[type].transpose();
			dDs[type] += dDs[type].transpose().eval();
		}

		const std::vector<EigenMatrix> HdDs = DensityHessian(dDs);

		EigenMatrix HdC = EigenZero(nbasis, Np + 2);
		std::vector<Eigen::Ref<EigenMatrix>> HdCs;
		__Make_Block_View__(HdC, HdCs);
		for ( int type = 0; type < 3; type++ ){
			HdCs[type] = 2 * occ[type] * ( HdDs[type] * Cs[type] + Fs[type] * dCs[type] );
		}

		return std::vector<EigenMatrix>{ HdC };
	};
};

class ObjNewton: public ObjNewtonBase{ public:
	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> Cs_, std::vector<int> derivatives) override{
		ObjNewtonBase::Calculate(Cs_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) && *xc ){
			xc->Evaluate("f", *grid);
			if (One) xc->Evaluate("f", *grid2);
		}
	};

	std::vector<EigenMatrix> DensityHessian(std::vector<EigenMatrix> dDs) const override{
		std::vector<EigenMatrix> dFs(3);

		const auto [J, Kd, Ka, Kb] = int4c2e->ContractInts(dDs[0], dDs[1], dDs[2], nthreads, 0);
		const std::vector<EigenMatrix> dFhfMs = {
			J - Kd - 0.5 * ( Ka + Kb ),
			J - Kd - Ka + Kb * lambda,
			J - Kd + Ka * lambda - Kb
		};
		const std::vector<EigenMatrix> dFhfTs = {
			J - Kd - 0.5 * ( Ka + Kb ),
			J - Kd - Ka - Kb,
			J - Kd - Ka - Kb
		};
		for ( int type = 0; type < 3; type++ ){
			dFs[type] = 2 * dFhfMs[type] - dFhfTs[type];
		}

		if (*xc){
			if (One){
				grid->getDensityU({ { dDs[0] + dDs[1] }, { dDs[0] + dDs[2] } });
				grid2->getDensityU({ { dDs[0] + dDs[1] + dDs[2] }, { dDs[0] } });
				std::vector<std::vector<EigenMatrix>> dGxcMs = grid->getFockU<u_t>();
				dGxcMs = {
					{ 0.5 * ( dGxcMs[0][0] + dGxcMs[1][0] ) },
					{ dGxcMs[0][0] },
					{ dGxcMs[1][0] }
				};
				std::vector<std::vector<EigenMatrix>> dGxcTs = grid2->getFockU<u_t>();
				dGxcTs = {
					{ 0.5 * ( dGxcTs[0][0] + dGxcTs[1][0] ) },
					{ dGxcTs[0][0] },
					{ dGxcTs[0][0] }
				};
				for ( int type = 0; type < 3; type++ ){
					dFs[type] += 2 * dGxcMs[type][0] - dGxcTs[type][0];
				}
			}else{
				grid->getDensityU({{ 2 * dDs[0] + dDs[1] + dDs[2] }});
				const EigenMatrix dGxcs = grid->getFockU<u_t>()[0][0];
				for ( int type = 0; type < 3; type++ ){
					dFs[type] += dGxcs;
				}
			}
		}
		return dFs;
	};
};

using ObjARH = UniversalObjARH<ObjNewtonBase>;

class Cons: public Maniverse::Function{ public:
	int Np;
	int nbasis = 0;
	std::array<std::array<ObjDeterminant, 2>, 2> lower;
	int lower_type = 0;

	Cons(
		int Np, EigenMatrix overlap,
		std::vector<EigenMatrix> lower_
	): Np(Np), nbasis(overlap.rows()), lower_type(lower_.size()){
		lower = std::array<std::array<ObjDeterminant, 2>, 2>{
				std::array<ObjDeterminant, 2>{ ObjDeterminant(overlap * lower_[0]), ObjDeterminant(overlap * lower_[0]) },
				std::array<ObjDeterminant, 2>{ ObjDeterminant(overlap * lower_.back()), ObjDeterminant(overlap * lower_.back()) }
		};
		Gradient = { EigenZero(nbasis, Np + 2) };
	};

	virtual void Calculate(std::vector<EigenMatrix> Cs_, std::vector<int> derivatives) override{
		EigenMatrix Ca = Cs_[0].leftCols( Np + 1 );
		EigenMatrix Cb = Ca; Cb.rightCols(1) = Cs_[0].rightCols(1);
		lower[0][0].Calculate({Ca}, derivatives);
		lower[1][1].Calculate({Cb}, derivatives);
		if ( lower_type == 2 ){
			lower[1][0].Calculate({Ca}, derivatives);
			lower[0][1].Calculate({Cb}, derivatives);
		}
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			Value = lower[0][0].Value * lower[1][1].Value;
			if ( lower_type == 2 ) Value += lower[1][0].Value * lower[0][1].Value;
		}
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			EigenMatrix Ga = lower[0][0].Gradient[0] * lower[1][1].Value;
			EigenMatrix Gb = lower[0][0].Value * lower[1][1].Gradient[0];
			if ( lower_type == 2 ){
				Ga += lower[0][1].Value * lower[1][0].Gradient[0];
				Gb += lower[0][1].Gradient[0] * lower[1][0].Value;
			}
			Gradient[0].leftCols(Np + 1) = Ga;
			Gradient[0].leftCols(Np) += Gb.leftCols(Np);
			Gradient[0].rightCols(1) = Gb.rightCols(1);
		}
	};

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> dCs_) const override{
		std::array<EigenMatrix, 2> dCs = {
			dCs_[0].leftCols(Np + 1),
			dCs_[0].leftCols(Np + 1)
		};
		dCs[1].rightCols(1) = dCs_[0].rightCols(1);
		EigenMatrix cons_hess = EigenZero(nbasis, Np + 2);
		EigenMatrix Ga = (
			lower[0][0].Hessian({dCs[0]})[0] * lower[1][1].Value
			+ lower[0][0].Gradient[0] * lower[1][1].Gradient[0].cwiseProduct(dCs[1]).sum()
		);
		EigenMatrix Gb = (
			lower[0][0].Gradient[0].cwiseProduct(dCs[0]).sum() * lower[1][1].Gradient[0]
			+ lower[0][0].Value * lower[1][1].Hessian({dCs[1]})[0]
		);
		if ( lower_type == 2 ){
			Ga += (
				lower[0][1].Gradient[0].cwiseProduct(dCs[1]).sum() * lower[1][0].Gradient[0]
				+ lower[0][1].Value * lower[1][0].Hessian({dCs[0]})[0]
			);
			Gb += (
				lower[0][1].Hessian({dCs[1]})[0] * lower[1][0].Value
				+ lower[0][1].Gradient[0] * lower[1][0].Gradient[0].cwiseProduct(dCs[0]).sum()
			);
		}
		cons_hess.leftCols(Np + 1) = Ga;
		cons_hess.leftCols(Np) += Gb.leftCols(Np);
		cons_hess.rightCols(1) += Gb.rightCols(1);
		return std::vector<EigenMatrix>{ cons_hess };
	};
};

} // namespace

#define __Precon__(_P_, _K_, _L_){\
	M._P_ = [&obj, bp = M.BlockParameters](EigenVector V){\
		EigenVector PV(V.size());\
		MV_Block(PV, 0, bp) = UniversalPreconditioner(\
			obj.Ccano.leftCols(obj.Np + 2), obj.Ccanos[3],\
			obj._K_, obj._L_,\
			obj.int2c1e->Overlap * MV_constBlock(V, 0, bp)\
		);\
		return PV;\
	};\
}

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenMatrix> TwoDeterminantRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid, Grid& grid2,
		int Np, EigenMatrix C,
		std::vector<std::vector<EigenMatrix>> lowers_,
		int nthreads, int output){
	std::conditional_t< scf_t == lbfgs_t,
				ObjLBFGS,
				std::conditional_t< scf_t == newton_t,
							ObjNewton,
							ObjARH
				>
	> obj(int2c1e, int4c2e, xc, grid, grid2, Np, nthreads);
	Maniverse::Flag flag(C, int2c1e.Overlap); flag.setBlockParameters({Np, 1, 1});
	std::vector<Cons> lowers;
	for ( std::vector<EigenMatrix>& lower_ : lowers_ ) lowers.emplace_back(Np, int2c1e.Overlap, lower_);
	std::vector<Maniverse::Function*> lowers_ptr;
	for ( Cons& lower : lowers ) lowers_ptr.push_back(&lower);
	Maniverse::Iterate M(obj, {flag.Share()}, lowers_ptr);
	std::array<double, 3> tol = {1.e-8, 1.e-5, 1.e-5};
	const std::vector<double> cons_tol(lowers_.size(), 1e-10);
	if constexpr ( scf_t == lbfgs_t ){
		__Precon__(PreconditionerSqrt, Ksqrtinv, Lsqrtinv);
		__Precon__(PreconditionerInvSqrt, Ksqrt, Lsqrt);
		if ( ! lowers_.size() && ! Maniverse::LBFGS(
					M, tol,
					20, 300, 0.1, 0.75, 10, output
		) ) throw std::runtime_error("Convergence failed!");
		if ( lowers_.size() && ! Maniverse::AugmentedLagrangian(1, 3.3, 0.8, cons_tol, 100, output)(Maniverse::LBFGS)(
					M, tol,
					20, 300, 0.1, 0.75, 10, output
		) ) throw std::runtime_error("Convergence failed!");
	}else{
		__Precon__(Preconditioner, Kinv, Linv);
		Maniverse::TrustRegion tr;
		static constexpr double ls_tol = scf_t == newton_t ? 0.001 : 0.01;
		Maniverse::ConjugateGradient cg(1, {ls_tol, ls_tol}, M.getDimension(), output);
		Maniverse::initLinearSolverForNewton(cg, M);
		if ( ! lowers_.size() && ! Maniverse::Newton(
					M, tr, cg, tol, 300, output
		) ) throw std::runtime_error("Convergence failed!");
		if ( lowers_.size() && ! Maniverse::AugmentedLagrangian(1, 3.3, 0.8, cons_tol, 100, output)(Maniverse::Newton)(
					M, tr, cg, tol, 300, output
		) ) throw std::runtime_error("Convergence failed!");
	}

	return std::make_tuple(obj.Value, obj.eps, obj.Ccano);
}

bool TwoDetStability(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid, Grid& grid2,
		int Np, EigenMatrix C,
		std::vector<std::vector<EigenMatrix>> lowers_,
		int stable,
		int nthreads, int output){
	ObjNewton obj(int2c1e, int4c2e, xc, grid, grid2, Np, nthreads);
	Maniverse::Flag flag(C, int2c1e.Overlap); flag.setBlockParameters({Np, 1, 1});
	std::vector<Cons> lowers;
	for ( std::vector<EigenMatrix>& lower_ : lowers_ ) lowers.emplace_back(Np, int2c1e.Overlap, lower_);
	std::vector<Maniverse::Function*> lowers_ptr;
	for ( Cons& lower : lowers ) lowers_ptr.push_back(&lower);
	Maniverse::Iterate M(obj, {flag.Share()}, lowers_ptr);
	M.Calculate(M.getPoint(), {0, 1, 2});
	M.setGradient();
	if ( lowers_.size() > 0 ) M.setLambda(M.calcLambda());
	const auto [Evals, Evecs] = Maniverse::Lanczos(M, stable, 0, lowers_.size() > 0, output);
	if ( output > 0 && Evals[0] < 0 ){
		std::printf("Non-positive curvature:\n");
		const EigenMatrix Y = int2c1e.Overlap * C;
		const EigenMatrix X = Eigen::Map<const Eigen::MatrixXd>(Evecs[0].data(), C.rows(), C.cols());
		EigenMatrix Omega = X * Y.transpose() - Y * X.transpose() + Y * ( Y.transpose() * X ) * Y.transpose();
		for ( int i = 0; i < 3; i++ ){
			int maxRow, maxCol;
			const double maxcoeff = Omega.cwiseAbs().maxCoeff(&maxRow, &maxCol);
			std::printf("%d <-> %d  %f\n", maxCol, maxRow, maxcoeff);
			Omega(maxRow, maxCol) = Omega(maxCol, maxRow) = 0;
		}
	}
	return Evals[0] > 0;
}

void TwoDet::Calculate0(){
	if ( scftype == "DRY" ) return;
	EigenMatrix Z = EigenZero(mwfn.getNumBasis(), mwfn.getNumIndBasis());
	Z <<
		mwfn.getCoefficientMatrix({.Set=0, .OccUpper=2, .OccLower=2}),
		mwfn.getCoefficientMatrix({.Set=0, .Type=1, .OccUpper=1, .OccLower=1}),
		mwfn.getCoefficientMatrix({.Set=0, .Type=2, .OccUpper=1, .OccLower=1}),
		mwfn.getCoefficientMatrix({.Set=0, .OccUpper=0, .OccLower=0})
	;
	auto [E, eps, C] =
		scftype == "LBFGS" ? TwoDeterminantRiemann<lbfgs_t>(*int2c1e, *int4c2e, xc, grid, grid2, Np, Z.leftCols(Np + 2), lowers, nthreads, 1) :
		scftype == "ARH" ? TwoDeterminantRiemann<arh_t>(*int2c1e, *int4c2e, xc, grid, grid2, Np, Z.leftCols(Np + 2), lowers, nthreads, 1) :
		/* scftype == "NEWTON" ? */ TwoDeterminantRiemann<newton_t>(*int2c1e, *int4c2e, xc, grid, grid2, Np, Z.leftCols(Np + 2), lowers, nthreads, 1);
	Energy += E;
	mwfn.setCoefficientMatrix(C, {.Set=0});
	mwfn.setEnergy(eps, {.Set=0});
	EigenVector occ = EigenZero(mwfn.getNumBasis(), 1);
	occ.head(Np).setConstant(2);
	occ.segment(Np, 2).setConstant(1);
	mwfn.setOccupation(occ, {.Set=0});
	for ( int iorb = 0; iorb < mwfn.getNumBasis(); iorb++ ) mwfn.Orbitals[0][iorb].Type = iorb == Np ? 1 : iorb == Np + 1 ? 2 : 0;
	if ( stable > 0 ) TwoDetStability(*int2c1e, *int4c2e, xc, grid, grid2, Np, C, lowers, stable, nthreads, 1);
}
