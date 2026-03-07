#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <tuple>
#include <map>
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <libmwfn.h>

#include "../../Macro.h"
#include "../../Integral.h"
#include "../../Grid.h"
#include "../../ExchangeCorrelation.h"

#include "../AugmentedRoothaanHall.h"
#include "../RestrictedOpen.h"

#define S ( int2c1e->Overlap )
#define Hcore ( int2c1e->Kinetic + int2c1e->Nuclear )

namespace{

#define ntypes (int)occ.size()
#define __Make_Block_View__(Mat, Mats){\
	int _ncols_ = 0;\
	if (nd) Mats.emplace(0, Mat.middleCols(_ncols_, nd));\
	_ncols_ += nd;\
	if (na) Mats.emplace(1, Mat.middleCols(_ncols_, na));\
	_ncols_ += na;\
	if (nb) Mats.emplace(2, Mat.middleCols(_ncols_, nb));\
	_ncols_ += nb;\
}

class ObjBase: public Maniverse::Objective{ public:
	Int2C1E* int2c1e;
	Int4C2E* int4c2e;
	ExchangeCorrelation* xc;
	Grid* grid;
	int nd;
	int na;
	int nb;
	EigenMatrix Z;
	int nthreads;

	int nbasis;

	EigenMatrix Cprime;
	EigenMatrix Cprime_perp;
	std::map<int, int> occ;
	std::map<int, Eigen::Ref<EigenMatrix>> Cprimes;
	std::map<int, Eigen::Ref<EigenMatrix>> Gradients;
	std::map<int, EigenMatrix> Dprimes;
	std::map<int, EigenMatrix> Fprimes;
	EigenVector epsilons;
	EigenMatrix C;
	EigenMatrix K;
	EigenMatrix L;

	ObjBase(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nd, int na, int nb, EigenMatrix Z,
		int nthreads
	): int2c1e(&int2c1e), int4c2e(&int4c2e), xc(&xc), grid(&grid), nd(nd), na(na), nb(nb), Z(Z), nthreads(nthreads){
		nbasis = Z.rows();
		Cprime = EigenZero(nbasis, nd + na + nb);
		Gradient = {Cprime};
		int ncols = 0;
		if (nd){
			occ[0] = 2;
			Dprimes[0] = Fprimes[0] = EigenZero(nbasis, nbasis);
			ncols += nd;
		}
		if (na){
			occ[1] = 1;
			Dprimes[1] = Fprimes[1] = EigenZero(nbasis, nbasis);
			ncols += na;
		}
		if (nb){
			occ[2] = 1;
			Dprimes[2] = Fprimes[2] = EigenZero(nbasis, nbasis);
			ncols += nb;
		}
		__Make_Block_View__(Cprime, Cprimes);
		__Make_Block_View__(Gradient[0], Gradients);
		epsilons = EigenZero(nbasis, 1);
	};

	virtual void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			Cprime = Cprimes_[0];
			std::vector<EigenMatrix> Ds(3, EigenZero(0, 0));
			for ( int type = 0; type < 3; type++ ) if ( occ.count(type) ){
				Dprimes[type] = Cprimes.at(type) * Cprimes.at(type).transpose();
				Ds[type].resize(nbasis, nbasis);
				Ds[type] = Z * Dprimes[type] * Z.transpose();
			}
			std::vector<EigenMatrix> Ghfs(3, EigenZero(nbasis, nbasis));
			std::tie(Ghfs[0], Ghfs[1], Ghfs[2]) = int4c2e->ContractInts(Ds[0], Ds[1], Ds[2], nthreads, 1);
			Value = 0;
			for ( int type = 0; type < 3; type++ ) if ( occ.count(type) ){
				const EigenMatrix Fhf = Hcore + Ghfs[type];
				Value += 0.5 * occ[type] * Ds[type].cwiseProduct( Hcore + Fhf ).sum();
				Fprimes[type] = Z.transpose() * Fhf * Z;
			}

			if (nd) Ds[0] *= 2;
			Ds.erase(
					std::remove_if(
						Ds.begin(), Ds.end(),
						[](const EigenMatrix& D){ return D.size() == 0; }
					), Ds.end()
			);
			if (*xc){
				grid->getDensity(Ds);
				xc->Evaluate("ev", *grid);
				Value += grid->getEnergy();
				const std::vector<EigenMatrix> Gxcs = grid->getFock();
				for ( int type = 0, itype = 0; type < 3; type++ ) if ( occ.count(type) ){
					Fprimes[type] += Z.transpose() * Gxcs[itype++] * Z;
				}
			}
		}

		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			for ( int type = 0; type < 3; type++ ) if ( occ.count(type) ){
				Gradients.at(type) = 2 * occ[type] * Fprimes[type] * Cprimes.at(type);
			}

			Eigen::HouseholderQR<EigenMatrix> qr(Cprime);
			const EigenMatrix Call = qr.householderQ();
			C = Z * Call;
			Cprime_perp = Call.rightCols(nbasis - nd - na - nb);
			std::map<int, EigenMatrix> Fmos;
			for ( int type = 0; type < 3; type++ ) if ( occ.count(type) ){
				Fmos[type] = Call.transpose() * Fprimes[type] * Call;
			}
			Fmos[3] = EigenZero(nbasis, nbasis);
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
			K = A.block(0, 0, nd + na + nb, nd + na + nb);
			L = A.block(nd + na + nb, 0, nbasis - nd - na - nb, nd + na + nb);
		}
	};
};

EigenMatrix Preconditioner(EigenMatrix U, EigenMatrix Uperp, EigenMatrix K, EigenMatrix L, EigenMatrix V){
	EigenMatrix Omega = U.transpose() * V;
	Omega = Omega.cwiseProduct(K);
	EigenMatrix Kappa = Uperp.transpose() * V;
	Kappa = Kappa.cwiseProduct(L);
	return ( U * Omega + Uperp * Kappa ).eval();
}

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
		return std::vector<EigenMatrix>{ ::Preconditioner(Cprime, Cprime_perp, Ksqrtinv, Lsqrtinv, Vs[0]) };
	};

	std::vector<EigenMatrix> PreconditionerInvSqrt(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{ ::Preconditioner(Cprime, Cprime_perp, Ksqrt, Lsqrt, Vs[0]) };
	};
};

class ObjNewtonBase: public ObjBase{ public:
	EigenMatrix Kinv, Linv;

	using ObjBase::ObjBase;

	void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		ObjBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) ){
			Kinv = K.cwiseInverse();
			Linv = L.cwiseInverse();
		}
	};

	virtual std::map<int, EigenMatrix> DensityHessian(std::map<int, EigenMatrix> dDprimes) const = 0;

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> Vprimes) const override{
		std::map<int, Eigen::Ref<EigenMatrix>> dCprimes;
		__Make_Block_View__(Vprimes[0], dCprimes);
		std::map<int, EigenMatrix> dDprimes;
		for ( int type = 0; type < 3; type++ ) if ( occ.count(type) ){
			dDprimes[type] = Cprimes.at(type) * dCprimes.at(type).transpose();
			dDprimes[type] += dDprimes[type].transpose().eval();
		}

		const std::map<int, EigenMatrix> HdDprimes = DensityHessian(dDprimes);

		EigenMatrix HdCprime = EigenZero(nbasis, nd + na + nb);
		std::map<int, Eigen::Ref<EigenMatrix>> HdCprimes;
		__Make_Block_View__(HdCprime, HdCprimes);
		for ( int type = 0; type < 3; type++ ) if ( occ.count(type) ){
			HdCprimes.at(type) = 2 * occ.at(type) * ( HdDprimes.at(type) * Cprimes.at(type) + Fprimes.at(type) * dCprimes.at(type) );
		}
		return std::vector<EigenMatrix>{ HdCprime };
	};

	std::vector<EigenMatrix> Preconditioner(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{ ::Preconditioner(Cprime, Cprime_perp, Kinv, Linv, Vs[0]) };
	};
};

class ObjNewton: public ObjNewtonBase{ public:
	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		ObjNewtonBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 2) && *xc ){
			xc->Evaluate("f", *grid);
		}
	};

	std::map<int, EigenMatrix> DensityHessian(std::map<int, EigenMatrix> dDprimes) const override{
		std::vector<std::vector<EigenMatrix>> dDs(3, {EigenZero(0, 0)});
		for ( int type = 0; type < 3; type++ ) if ( occ.count(type) ){
			dDs[type][0] = Z * dDprimes[type] * Z.transpose();
		}
		std::vector<EigenMatrix> dGs(3, EigenZero(nbasis, nbasis));
		std::tie(dGs[0], dGs[1], dGs[2]) = int4c2e->ContractInts(dDs[0][0], dDs[1][0], dDs[2][0], nthreads, 0);

		dDs[0][0] *= 2;
		dDs.erase(
				std::remove_if(
					dDs.begin(), dDs.end(),
					[](const std::vector<EigenMatrix>& dD){ return dD[0].size() == 0; }
				), dDs.end()
		);
		if (*xc){
			grid->getDensityU(dDs);
			const std::vector<std::vector<EigenMatrix>> dGxcs = grid->getFockU<u_t>();
			for ( int type = 0, itype = 0; type < 3; type++ ) if ( occ.count(type) ){
				dGs[type] += dGxcs[itype++][0];
			}
		}
		std::map<int, EigenMatrix> HdDprimes;
		for ( int type = 0; type < 3; type++ ) if ( occ.count(type) ){
			HdDprimes[type] = Z.transpose() * dGs[type] * Z;
		}
		return HdDprimes;
	};
};

class ObjARH: public ObjNewtonBase{ public:
	AugmentedRoothaanHall arh = AugmentedRoothaanHall(20, 1);

	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> Cprimes_, std::vector<int> derivatives) override{
		ObjNewtonBase::Calculate(Cprimes_, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			EigenMatrix Dprime = EigenZero(nbasis, nbasis * ntypes);
			EigenMatrix Fprime = EigenZero(nbasis, nbasis * ntypes);
			for ( int itype = 0; itype < ntypes; itype++ ){
				Dprime(Eigen::placeholders::all, Eigen::seqN(itype * nbasis, nbasis)) = Dprimes[itype];
				Fprime(Eigen::placeholders::all, Eigen::seqN(itype * nbasis, nbasis)) = Fprimes[itype] / 2;
			}
			if (nd) Fprime.leftCols(nbasis) *= 2;
			arh.Append(Dprime, Fprime);
		}
	};

	std::map<int, EigenMatrix> DensityHessian(std::map<int, EigenMatrix> dDprimes) const override{
		EigenMatrix dDprime = EigenZero(nbasis, nbasis * ntypes);
		for ( int type = 0, itype = 0; type < 3; type++ ) if ( occ.count(type) ){
			dDprime.middleCols((itype++) * nbasis, nbasis) = dDprimes[type];
		}
		const EigenMatrix HdDprime = arh.Hessian(dDprime);
		std::map<int, EigenMatrix> HdDprimes = dDprimes;
		for ( int type = 0, itype = 0; type < 3; type++ ) if ( occ.count(type) ){
			HdDprimes[type] = HdDprime.middleCols((itype++) * nbasis, nbasis);
		}
		return HdDprimes;
	};
};

} // namespace

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nd, int na, int nb, EigenMatrix Z,
		int nthreads, int output){
	std::conditional_t< scf_t == lbfgs_t,
				ObjLBFGS,
				std::conditional_t< scf_t == newton_t,
							ObjNewton,
							ObjARH
				>
	> obj(int2c1e, int4c2e, xc, grid, nd, na, nb, Z, nthreads);
	std::vector<int> space = {};
	if (nd) space.push_back(nd);
	if (na) space.push_back(na);
	if (nb) space.push_back(nb);
	Maniverse::Flag flag(EigenOne(Z.rows(), nd + na + nb)); flag.setBlockParameters(space);
	Maniverse::Iterate M(obj, {flag.Share()}, 1);
	std::tuple<double, double, double> tol = {1.e-8, 1.e-5, 1.e-5};
	if constexpr ( scf_t == lbfgs_t ){
		if ( ! Maniverse::LBFGS(
					M, tol,
					20, 300, 0.1, 0.75, 100, output
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

	#define nv obj.nbasis - nd - na - nb
	/*if ( nd && na ){ // Guest and Saunders averaged Fock matrix
		EigenMatrix Cp_all = EigenZero(obj.nbasis, obj.nbasis);
		Cp_all << obj.Cprime, obj.Cprime_perp;
		EigenMatrix Fd = Cp_all.transpose() * obj.Fprimes[0] * Cp_all;
		EigenMatrix Fa = 0.5 * Cp_all.transpose() * obj.Fprimes[1] * Cp_all;
		EigenMatrix bigF = EigenZero(obj.nbasis, obj.nbasis);
		// Diagonal
		bigF.block(0, 0, nd, nd) = Fd.block(0, 0, nd, nd);
		bigF.block(nd, nd, na, na) = Fd.block(nd, nd, na, na);
		bigF.block(nd + na, nd + na, nv, nv) = Fd.block(nd + na, nd + na, nv, nv);
		// Off-diagonal
		bigF.block(0, nd, nd, na) = Fd.block(0, nd, nd, na) - Fa.block(0, nd, nd, na);
		bigF.block(nd, 0, na, nd) = bigF.block(0, nd, nd, na).transpose();
		bigF.block(0, nd + na, nd, nv) = Fd.block(0, nd + na, nd, nv);
		bigF.block(nd + na, 0, nv, nd) = bigF.block(0, nd + na, nd, nv).transpose();
		bigF.block(nd, nd + na, na, nv) = Fa.block(nd, nd + na, na, nv);
		bigF.block(nd + na, nd, nv, na) = bigF.block(nd, nd + na, na, nv).transpose();
		bigF = ( Cp_all * bigF * Cp_all.transpose() ).eval();
		Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
		eigensolver.compute(bigF);
		const EigenVector epsilons = eigensolver.eigenvalues();
		const EigenMatrix C = Z * eigensolver.eigenvectors();
		return std::make_tuple(obj.Value, epsilons, C);
	}else */return std::make_tuple(obj.Value, obj.epsilons, obj.C);
}

void RO_SCF::Calculate0(){
	int nd = 0; int na = 0; int nb = 0;
	for ( auto& orbital : mwfn.Orbitals ){
		if ( orbital.Occ > 0 ) switch (orbital.Type){
			case 0: nd++; break;
			case 1: na++; break;
			case 2: nb++; break;
		}
	}
	const EigenMatrix Z = mwfn.getCoefficientMatrix(1);
	auto [E, epsilons, C] =
		scftype == "LBFGS" ? RestrictedOpenRiemann<lbfgs_t>(int2c1e, int4c2e, xc, grid, nd, na, nb, Z, nthreads, 1) :
		scftype == "ARH" ? RestrictedOpenRiemann<arh_t>(int2c1e, int4c2e, xc, grid, nd, na, nb, Z, nthreads, 1) :
		/* scftype == "NEWTON" ? */ RestrictedOpenRiemann<newton_t>(int2c1e, int4c2e, xc, grid, nd, na, nb, Z, nthreads, 1);
	Energy += E;
	mwfn.setEnergy(epsilons, 1);
	mwfn.setCoefficientMatrix(C, 1);
}
