#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <libmwfn.h>

#include "../../Macro.h"
#include "../../DIIS.h"

#include "../Universal.h"
#include "../Restricted.h"

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

	using UniversalObjBase::UniversalObjBase;

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
		int nthreads, int output){
	std::conditional_t< scf_t == lbfgs_t,
				ObjLBFGS,
				std::conditional_t< scf_t == newton_t,
							ObjNewton,
							ObjARH
				>
	> obj(int2c1e, int4c2e, xc, grid, Norbs, Coupling, {Z, Z, Z}, nthreads);
	Maniverse::Flag flag(EigenOne(Z.rows(), Norbs[0] + Norbs[1] + Norbs[2]));
	std::vector<int> space = {};
	for ( int Norb : Norbs ) if ( Norb > 0 ) space.push_back(Norb);
	flag.setBlockParameters(space);
	Maniverse::Iterate M(obj, {flag.Share()}, 1);
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

	EigenVector eps = EigenZero(Z.cols(), 1);
	EigenMatrix C = EigenZero(Z.rows(), Z.cols());
	if ( Norbs[0] && !Norbs[1] && !Norbs[2] ){
		Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
		eigensolver.compute(obj.Fprimes[0]);
		eps = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
	}else C << Z * obj.Cprime, Z * obj.Cprime_perp;
	return std::make_tuple(obj.Value, eps, C);
}

void R_SCF::Calculate0(){
	if ( scftype == "DRY" ) return;
	const EigenMatrix Z = mwfn.getCoefficientMatrix(1);
	const EigenMatrix F = mwfn.getFock(1);
	auto [E, eps, C] =
		scftype == "DIIS" ? RestrictedDIIS(Np, int2c1e, int4c2e, xc, grid, F, Z, 1, nthreads) :
		scftype == "LBFGS" ? RestrictedRiemann<lbfgs_t>(int2c1e, int4c2e, xc, grid, {Np, Na, Nb}, Coupling, Z, nthreads, 1) :
		scftype == "ARH" ? RestrictedRiemann<arh_t>(int2c1e, int4c2e, xc, grid, {Np, Na, Nb}, Coupling, Z, nthreads, 1) :
		/* scftype == "NEWTON" ? */ RestrictedRiemann<newton_t>(int2c1e, int4c2e, xc, grid, {Np, Na, Nb}, Coupling, Z, nthreads, 1);
	Energy += E;
	mwfn.setEnergy(eps, 1);
	mwfn.setCoefficientMatrix(C, 1);
}
