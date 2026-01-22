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
#include <Maniverse/Manifold/Euclidean.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <libmwfn.h>

#include "../Macro.h"
#include "../Integral/Int2C1E.h"
#include "../Integral/Int4C2E.h"
#include "../Grid/Grid.h"
#include "../ExchangeCorrelation.h"
#include "../DIIS/CDIIS.h"
#include "AugmentedRoothaanHall.h"

#define S (int2c1e.Overlap)
#define Hcore (int2c1e.Kinetic + int2c1e.Nuclear )

static EigenVector FermiDirac(EigenVector epsilons, double T, double Mu, int order){
	EigenArray ns = 1. / ( 1. + ( ( epsilons.array() - Mu ) / T ).exp() );
	EigenArray res = ns / std::pow( - T, order );
	for ( int i = 1; i <= order; i++ ) res *= 1. - i * ns;
	return res.matrix();
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix F, EigenVector Occ, EigenMatrix Z,
		int output, int nthreads){
	double oldE = 0;
	double E = 0;
	const int nbasis = F.cols();
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenVector occupations = Occ;
	EigenMatrix C = EigenOne(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	bool first_iter = 1;

	std::function<std::tuple<
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
			>(std::vector<EigenMatrix>&, std::vector<bool>&)
	> update_func = [&](std::vector<EigenMatrix>& Fs_, std::vector<bool>&){
		assert(Fs_.size() == 1 && "Only one Fock matrix should be optimized in spin-restricted SCF!");
		const EigenMatrix F_ = Fs_[0];
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z;
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		oldE = E;
		EigenArray ns = FermiDirac(epsilons, T, Mu, 0).array();
		if (first_iter){
			first_iter = 0;
			ns = occupations.array();
		}else occupations = ns.matrix();
		if (output>0) std::printf("Total number of electrons = %.10f\n", 2 * ns.sum());
		E = 2 * (
				T * (
					ns.pow(ns).log() + ( 1. - ns ).pow( 1. - ns ).log()
				).sum()
				- Mu * ns.sum()
		);
		const EigenMatrix D_ = C * occupations.asDiagonal() * C.transpose();
		const auto [Ghf_, _, __] = int4c2e.ContractInts(D_, EigenZero(0, 0), EigenZero(0, 0), nthreads, 1);
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
		/*{
			EigenMatrix Cprime_old = eigensolver.eigenvectors();
			EigenMatrix Dprime_old = Cprime_old * occupations.asDiagonal() * Cprime_old.transpose();
			EigenMatrix Fprime_new = Z.transpose() * Fnew_ * Z;
			eigensolver.compute(Fprime_new);
			EigenVector occ_new = FermiDirac(eigensolver.eigenvalues(), T, Mu, 0);
			EigenMatrix Cprime_new = eigensolver.eigenvectors();
			EigenMatrix Dprime_new = Cprime_new * occ_new.asDiagonal() * Cprime_new.transpose();
			std::printf("Residual: %E\n", (Dprime_new - Dprime_old).norm());
		}*/
		E += (D_ * ( Hcore + Fhf_ )).trace() + Exc_;
		if (output>0){
			std::printf("Electronic grand potential = %.10f\n", E);
			std::printf("Changed by %E from the last step\n", E - oldE);
		}
		EigenMatrix G_ = Fnew_ * D_ * S - S * D_ * Fnew_;
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
	CDIIS cdiis(&update_func, 1, 20, 1e-6, 300, output>0 ? 2 : 0);
	cdiis.Damps.push_back(std::make_tuple(0.1, 100, 0.75));
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilons, occupations, C);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteLoopDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix F, EigenVector Occ, EigenMatrix Z,
		int output, int nthreads){
	double oldE = 0;
	double E = 0;
	const int nbasis = F.cols();
	EigenVector epsilons = EigenZero(Z.cols(), 1);
	EigenVector occupations = Occ;
	EigenMatrix C = EigenOne(Z.rows(), Z.cols());
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	bool first_iter = 1;

	std::function<std::tuple<
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
			>(std::vector<EigenMatrix>&, std::vector<bool>&)
	> update_func = [&](std::vector<EigenMatrix>& Fs_, std::vector<bool>&){
		assert(Fs_.size() == 1 && "Only one Fock matrix should be optimized in spin-restricted SCF!");
		const EigenMatrix F_ = Fs_[0];
		const EigenMatrix Fprime_ = Z.transpose() * F_ * Z;
		eigensolver.compute(Fprime_);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();
		oldE = E;
		EigenArray ns = FermiDirac(epsilons, T, Mu, 0).array();
		if (first_iter || 1){
			first_iter = 0;
			ns = occupations.array();
		}//else occupations = ns.matrix();
		if (output>0) std::printf("Total number of electrons = %.10f\n", 2 * ns.sum());
		E = 2 * (
				T * (
					ns.pow(ns).log() + ( 1. - ns ).pow( 1. - ns ).log()
				).sum()
				- Mu * ns.sum()
		);
		const EigenMatrix D_ = C * occupations.asDiagonal() * C.transpose();
		const auto [Ghf_, _, __] = int4c2e.ContractInts(D_, EigenZero(0, 0), EigenZero(0, 0), nthreads, 1);
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(nbasis, nbasis);
		if (xc){
			grid.getDensity({2 * D_});
			xc.Evaluate("ev", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock()[0];
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix Fnew_ = F = Fhf_ + Gxc_;
		{
			EigenMatrix Cprime_old = eigensolver.eigenvectors();
			EigenMatrix Dprime_old = Cprime_old * occupations.asDiagonal() * Cprime_old.transpose();
			EigenMatrix Fprime_new = Z.transpose() * Fnew_ * Z;
			eigensolver.compute(Fprime_new);
			EigenVector occ_new = FermiDirac(eigensolver.eigenvalues(), T, Mu, 0);
			EigenMatrix Cprime_new = eigensolver.eigenvectors();
			EigenMatrix Dprime_new = Cprime_new * occ_new.asDiagonal() * Cprime_new.transpose();
			std::printf("Residual: %E\n", (Dprime_new - Dprime_old).norm());
		}
		E += (D_ * ( Hcore + Fhf_ )).trace() + Exc_;
		if (output>0){
			std::printf("Electronic grand potential = %.10f\n", E);
			std::printf("Changed by %E from the last step\n", E - oldE);
		}
		EigenMatrix G_ = Fnew_ * D_ * S - S * D_ * Fnew_;
		EigenMatrix Aux_ = EigenZero(F.rows(), F.cols() + 1);
		Aux_ << D_, EigenZero(F.rows(), 1);
		Aux_(0, F.cols()) = E;
		return std::make_tuple(
				std::vector<EigenMatrix>{Fnew_},
				std::vector<EigenMatrix>{G_},
				std::vector<EigenMatrix>{Aux_}
		);
	};
	hello:
	std::vector<EigenMatrix> Fs = {F};
	CDIIS cdiis(&update_func, 1, 20, 1e-6, 300, output>0 ? 2 : 0);
	cdiis.Damps.push_back(std::make_tuple(0.1, 100, 0.75));
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	EigenVector occupations_new = FermiDirac(epsilons, T, Mu, 0);
	if ( ( occupations - occupations_new ).norm() > 1e-6 ){
		occupations = occupations_new;
		goto hello;
	}
	return std::make_tuple(E, epsilons, occupations, C);
}

#undef S
#undef Hcore

namespace{

#define S (int2c1e->Overlap)
#define Hcore (int2c1e->Kinetic + int2c1e->Nuclear )
#define Eta 1e-8
#define Eta2 1e-6

class OneMoreOccupied: public std::exception{ public:
	const char* what() const throw(){
		return "One active orbital is set to occupied!";
	}
};

class OneMoreVirtual: public std::exception{ public:
	const char* what() const throw(){
		return "One active orbital is set to virtual!";
	}
};

class ObjBase: public Maniverse::Objective{ public:
	Int2C1E* int2c1e;
	Int4C2E* int4c2e;
	ExchangeCorrelation* xc;
	Grid* grid;
	double T, Mu;
	EigenMatrix Z;
	int nthreads;
	int output;

	int nbasis;
	int No;
	int Na;
	int Nv;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	EigenMatrix Cprime_oa;
	EigenMatrix Cprime_o;
	EigenMatrix Cprime_a;
	EigenMatrix Cprime_v;
	EigenMatrix Dprime;
	EigenMatrix Fprime;
	EigenVector Eps_oav;
	EigenVector Occ_a;
	EigenVector Occ_oa;
	EigenMatrix K, L;

	ObjBase(
			Int2C1E& int2c1e, Int4C2E& int4c2e,
			ExchangeCorrelation& xc, Grid& grid,
			double T, double Mu, EigenMatrix Z,
			int nthreads, int output
	): int2c1e(&int2c1e), int4c2e(&int4c2e), xc(&xc), grid(&grid), T(T), Mu(Mu), Z(Z), nthreads(nthreads), output(output){
		nbasis = Z.rows();
	};

	#define Occ_a_arr Occ_a.array()
	#define Occ_oa_arr Occ_oa.array()
	void Calculate(std::vector<EigenMatrix> X, int /*derivative*/){
		Cprime_oa = X[0];
		Occ_a = X.size() == 2 ? X[1] : Eigen::VectorXd::Zero(0);
		Na = Occ_a.size(); No = Cprime_oa.cols() - Na; Nv = nbasis - No - Na;
		Cprime_o = Cprime_oa.leftCols(No);
		Cprime_a = Cprime_oa.rightCols(Na);
		Eigen::HouseholderQR<EigenMatrix> qr(Cprime_oa);
		Cprime_v = EigenMatrix(qr.householderQ()).rightCols(Nv);
		Occ_oa = EigenMatrix::Ones(No + Na, 1);
		Occ_oa.tail(Na) = Occ_a;

		if (output>0){
			std::printf("Fractional occupation:");
			for ( int i = 0; i < Na; i++ ) std::printf(" %f", Occ_a(i));
			std::printf("\n");
		}
		for ( int i = 0; i < Na; i++ ) if ( Occ_a(i) > 1. - Eta ) throw OneMoreOccupied();
		for ( int i = 0; i < Na; i++ ) if ( Occ_a(i) < Eta ) throw OneMoreVirtual();
		if (output>0) std::printf("Total number of electrons = %.10f\n", 2 * Occ_oa.sum() );

		Dprime = Cprime_oa * Occ_oa.asDiagonal() * Cprime_oa.transpose();
		const EigenMatrix D = Z * Dprime * Z.transpose();
		const auto [Ghf, _, __] = int4c2e->ContractInts(D, EigenZero(0, 0), EigenZero(0, 0), nthreads, 1);
		const EigenMatrix Fhf = Hcore + Ghf;
		double Exc = 0;
		EigenMatrix Gxc = EigenZero(nbasis, nbasis);
		if (*xc){
			grid->getDensity({ 2 * D });
			xc->Evaluate("ev", *grid);
			Exc = grid->getEnergy();
			Gxc = grid->getFock()[0];
		}
		const EigenMatrix F = Fhf + Gxc;
		Fprime = Z.transpose() * F * Z;

		eigensolver.compute(Fprime);
		Eps_oav = eigensolver.eigenvalues();

		Value = D.cwiseProduct(Hcore + Fhf).sum() + Exc
			+ 2 * T * (
					Occ_a_arr.pow(Occ_a_arr).log() +
					( 1. - Occ_a_arr ).pow( 1. - Occ_a_arr ).log()
			).sum()
			- 2 * Mu * Occ_oa_arr.sum();
		EigenMatrix GradC = 4 * Fprime * Cprime_oa * Occ_oa.asDiagonal();
		Gradient = { GradC };
		if ( Na ){
			const EigenVector GradOcc1 = ( Cprime_oa.transpose() * Fprime * Cprime_oa ).diagonal().tail(Na) * 2;
			const EigenVector GradOcc2 = - 2 * ( T * ( 1. / Occ_a_arr - 1. ).log() + Mu ).matrix();
			Gradient.push_back( GradOcc1 + GradOcc2 );
		}

		// Preconditioner
		// https://doi.org/10.1016/j.cpc.2016.06.023
		EigenMatrix A = EigenMatrix::Ones(nbasis, nbasis);
		for ( int i = 0; i < nbasis; i++ ){
			const double occ_i = i < No + Na ? Occ_oa(i) : 0;
			for ( int j = i; j < nbasis; j++ ){
				const double occ_j = j < No + Na ? Occ_oa(j) : 0;
				const double occ_diff = occ_i - occ_j;
				if ( std::abs(occ_diff) > Eta ) A(i, j) = A(j, i) = - 2 * ( Eps_oav(i) - Eps_oav(j) ) * occ_diff;
			}
		}
		K = A.topLeftCorner(No + Na, No + Na);
		L = A.bottomLeftCorner(Nv, No + Na);
	};
};

EigenMatrix Preconditioner(EigenMatrix U, EigenMatrix Uperp, EigenMatrix B, EigenMatrix C, EigenMatrix V){
	EigenMatrix Omega = U.transpose() * V;
	Omega = Omega.cwiseProduct(B);
	EigenMatrix Kappa = Uperp.transpose() * V;
	Kappa = Kappa.cwiseProduct(C);
	return ( U * Omega + Uperp * Kappa ).eval();
}

class ObjLBFGS: public ObjBase{ public:
	EigenMatrix Ksqrt, Ksqrtinv, Lsqrt, Lsqrtinv;

	using ObjBase::ObjBase;

	void Calculate(std::vector<EigenMatrix> X, int /*derivative*/) override{
		ObjBase::Calculate(X, 2);
		Ksqrt = K.cwiseSqrt();
		Ksqrtinv = Ksqrt.cwiseInverse();
		Lsqrt = L.cwiseSqrt();
		Lsqrtinv = Lsqrt.cwiseInverse();
	};

	std::vector<EigenMatrix> PreconditionerSqrt(std::vector<EigenMatrix> Vs) const override{
		const EigenMatrix PCC = ::Preconditioner(Cprime_oa, Cprime_v, Ksqrtinv, Lsqrtinv, Vs[0]);
		if ( Na ) return std::vector<EigenMatrix>{ PCC, Vs[1] };
		else return std::vector<EigenMatrix>{ PCC };
	};

	std::vector<EigenMatrix> PreconditionerInvSqrt(std::vector<EigenMatrix> Vs) const override{
		const EigenMatrix PCC = ::Preconditioner(Cprime_oa, Cprime_v, Ksqrt, Lsqrt, Vs[0]);
		if ( Na ) return std::vector<EigenMatrix>{ PCC, Vs[1] };
		else return std::vector<EigenMatrix>{ PCC };
	};
};

class ObjNewtonBase: public ObjBase{ public:
	EigenMatrix Kinv, Linv;

	using ObjBase::ObjBase;

	void Calculate(std::vector<EigenMatrix> X, int /*derivative*/) override{
		ObjBase::Calculate(X, 2);
		Kinv = K.cwiseInverse();
		Linv = L.cwiseInverse();
	};

	virtual EigenMatrix DensityHessian(EigenMatrix dDprime) const = 0;

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> dX) const override{
		const EigenMatrix& dCprime_oa = dX[0];

		// HCC
		EigenMatrix dDprime = Cprime_oa * Occ_oa.asDiagonal() * dCprime_oa.transpose();
		dDprime += dDprime.transpose().eval();
		const EigenMatrix FoverC = DensityHessian(dDprime);
		const EigenMatrix HCC = 4 * (
				FoverC * Cprime_oa +
				Fprime * dCprime_oa
		) * Occ_oa.asDiagonal();
		if ( Na == 0 ) return std::vector<EigenMatrix>{ HCC };
		else{
			const EigenMatrix& dNprime_a = dX[1];

			// HCN
			const EigenMatrix dDprime = Cprime_a * dNprime_a.asDiagonal() * Cprime_a.transpose();
			const EigenMatrix FoverN = DensityHessian(dDprime);
			EigenMatrix HCN = FoverN * Cprime_oa * Occ_oa.asDiagonal();
			HCN.rightCols(Na) += Fprime * Cprime_a * dNprime_a.asDiagonal();
			HCN *= 4;

			// HNC
			EigenVector HNC = 2 * (
					2 * dCprime_oa.rightCols(Na).transpose() * Fprime * Cprime_a
					+ Cprime_a.transpose() * FoverC * Cprime_a
			).diagonal();

			// HNN
			EigenVector HNN = 2 * (
					( Cprime_a.transpose() * FoverN * Cprime_a ).diagonal()
					- ( T / Occ_a_arr / ( Occ_a_arr - 1. ) ).matrix().cwiseProduct(dNprime_a)
			);

			return std::vector<EigenMatrix>{
							HCC + HCN,
							HNC + HNN
			};
		}
	};

	std::vector<EigenMatrix> Preconditioner(std::vector<EigenMatrix> Vs) const override{
		const EigenMatrix PCC = ::Preconditioner(Cprime_oa, Cprime_v, Kinv, Linv, Vs[0]);
		if ( Na ) return std::vector<EigenMatrix>{ PCC, Vs[1] };
		else return std::vector<EigenMatrix>{ PCC };
	};
};

class ObjNewton: public ObjNewtonBase{ public:
	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> X, int /*derivative*/) override{
		ObjNewtonBase::Calculate(X, 2);
		if (*xc) xc->Evaluate("f", *grid);
	};

	EigenMatrix DensityHessian(EigenMatrix dDprime) const override{
		const EigenMatrix D = Z * dDprime * Z.transpose();
		const auto [FhfU, _, __] = int4c2e->ContractInts(D, EigenZero(0, 0), EigenZero(0, 0), nthreads, 0);
		EigenMatrix FxcU = EigenZero(nbasis, nbasis);
		if (*xc){
			grid->getDensityU({{ 2 * D }});
			FxcU = grid->getFockU<u_t>()[0][0];
		}
		const EigenMatrix FU = FhfU + FxcU;
		return Z.transpose() * FU * Z;
	};
};

class ObjARH: public ObjNewtonBase{ public:
	AugmentedRoothaanHall arh = AugmentedRoothaanHall(20, 1);

	using ObjNewtonBase::ObjNewtonBase;

	void Calculate(std::vector<EigenMatrix> X, int /*derivative*/) override{
		ObjNewtonBase::Calculate(X, 2);
		arh.Append(Dprime, Fprime);
	};

	EigenMatrix DensityHessian(EigenMatrix dDprime) const override{
		return arh.Hessian(dDprime);
	};
};

#undef Nv

} // namespace

std::tuple<int, int, int> Regularize(EigenVector& occ, double threshold){
	int ni = 0; int na = 0; int nv = 0;
	for ( int i = 0; i < occ.size(); i++ ){
		if ( occ(i) > 1. - threshold ){
			ni++;
			occ(i) = 1;
		}else if ( occ(i, 0) > threshold ) na++;
		else{
		   	nv++;
			occ(i) = 0;
		}
	}
	return std::make_tuple(ni, na, nv);
}

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		double T, double Mu,
		EigenVector all_occ, EigenMatrix Z,
		int nthreads, int output){
	int No, Na, Nv;
	std::tuple<double, double, double> tol = {1.e-3, 1.e-2, 1.e-2};
	EigenMatrix Cprime = EigenOne(Z.rows(), Z.cols());
	AugmentedRoothaanHall arh(20, 1);

	RESTART:
	std::tie(No, Na, Nv) = Regularize(all_occ, Eta);
	Maniverse::Flag flag(Cprime.leftCols(No + Na));
	std::vector<int> spaces;
	if ( No > 0 ) spaces.push_back(No);
	for ( int i = 0; i < Na; i++ ) spaces.push_back(1);
	flag.setBlockParameters(spaces);
	std::vector<std::shared_ptr<Maniverse::Manifold>> ms = {flag.Share()};
	if ( Na > 0 ){
		Maniverse::Euclidean euclidean(all_occ(Eigen::seqN(No, Na)));
		ms.push_back(euclidean.Share());
	}
	std::conditional_t< scf_t == lbfgs_t,
				ObjLBFGS,
				std::conditional_t< scf_t == newton_t,
							ObjNewton,
							ObjARH
				>
	> obj(int2c1e, int4c2e, xc, grid, T, Mu, Z, nthreads, output);
	if constexpr ( scf_t == arh_t ) obj.arh = arh;
	Maniverse::Iterate M(obj, ms, 1);

	try{
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
				arh = obj.arh;
			}
		}
	}catch (OneMoreOccupied&){
		if (output) std::printf("One active orbital is set to occupied!\n");
		Cprime << obj.Cprime_oa, obj.Cprime_v;
		all_occ = EigenZero(Z.cols(), 1);
		all_occ.head(No + Na) = obj.Occ_oa;
		all_occ(No) = 1;
		No++; Na--;
		goto RESTART;
	}catch (OneMoreVirtual&){
		if (output) std::printf("One active orbital is set to virtual!\n");
		Cprime << obj.Cprime_oa, obj.Cprime_v;
		all_occ = EigenZero(Z.cols(), 1);
		all_occ.head(No + Na) = obj.Occ_oa;
		all_occ(No + Na) = 0;
		Na--; Nv++;
		goto RESTART;
	}
	Cprime << obj.Cprime_oa, obj.Cprime_v;
	all_occ = EigenZero(Z.cols(), 1);
	all_occ.head(No + Na) = obj.Occ_oa;
	EigenVector all_occ_new = FermiDirac(obj.Eps_oav, T, Mu, 0);
	auto [No_new, Na_new, Nv_new] = Regularize(all_occ_new, Eta);
	if ( No != No_new || Na != Na_new || Nv != Nv_new ){
		Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
		if ( No > No_new ){
			const EigenMatrix Co = Cprime.leftCols(No);
			const EigenMatrix Fo = Co.transpose() * obj.Fprime * Co;
			eigensolver.compute(Fo);
			Cprime.leftCols(No) = Co * eigensolver.eigenvectors();
			all_occ(No - 1) = 1. - Eta2;
			No--;
		}
		if ( Nv > Nv_new ){
			const EigenMatrix Cv = Cprime.rightCols(Nv);
			const EigenMatrix Fv = Cv.transpose() * obj.Fprime * Cv;
			eigensolver.compute(Fv);
			Cprime.rightCols(Nv) = Cv * eigensolver.eigenvectors();
			all_occ(Cprime.rows() - Nv) = Eta2;
			Nv--;
		}
		Na = Cprime.rows() - No - Nv;
		if (output) std::printf("Switching manifold due to inconsistent orbital energy and occupation!\n");
		goto RESTART;
	}
	if ( tol != std::make_tuple(1e-8, 1e-5, 1e-5) ){
		if (output) std::printf("Switching to higher convergence precision!\n");
		tol = {1e-8, 1e-5, 1e-5};
		if constexpr ( scf_t == arh_t ){
			arh.Ps.pop_back();
			arh.Gs.pop_back();
		}
		goto RESTART;
	}

	const EigenMatrix C = Z * Cprime;
	return std::make_tuple(obj.Value, obj.Eps_oav, all_occ, C);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteLBFGS(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		double T, double Mu,
		EigenVector all_occ, EigenMatrix Z,
		int nthreads, int output){
	return RestrictedFiniteRiemann<lbfgs_t>(int2c1e, int4c2e, xc, grid, T, Mu, all_occ, Z, nthreads, output);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteNewton(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		double T, double Mu,
		EigenVector all_occ, EigenMatrix Z,
		int nthreads, int output){
	return RestrictedFiniteRiemann<newton_t>(int2c1e, int4c2e, xc, grid, T, Mu, all_occ, Z, nthreads, output);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		double T, double Mu,
		EigenVector all_occ, EigenMatrix Z,
		int nthreads, int output){
	return RestrictedFiniteRiemann<arh_t>(int2c1e, int4c2e, xc, grid, T, Mu, all_occ, Z, nthreads, output);
}
