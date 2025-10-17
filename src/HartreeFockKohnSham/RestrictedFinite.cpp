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
#include <Maniverse/Optimizer/TrustRegion.h>
#include <Maniverse/Optimizer/Anderson.h>
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
		EigenArray ns = 1. / ( 1. + ( ( epsilons.array() - Mu ) / T ).exp() );
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
			grid.getDensity(2 * D_);
			xc.Evaluate("ev", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock();
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix Fnew_ = Fhf_ + Gxc_;
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
	CDIIS cdiis(&update_func, 1, 20, 1e-8, 100, output>0 ? 2 : 0);
	cdiis.Damps.push_back(std::make_tuple(0.1, 100, 0.75));
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");
	return std::make_tuple(E, epsilons, occupations, C);
}

static EigenVector FermiDirac(EigenVector epsilons, double T, double Mu, int order){
	EigenArray ns = 1. / ( 1. + ( ( epsilons.array() - Mu ) / T ).exp() );
	EigenArray res = ns / std::pow( - T, order );
	for ( int i = 1; i <= order; i++ ) res *= 1. - i * ns;
	return res.matrix();
}

class SwitchingManifold : public std::exception{ public:
	const char* what() const throw(){
		return "Switching manifold!";
	}
};

static std::tuple<int, int, int> Regularize(EigenVector& occ, double threshold){
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

#define IdentityFunc [](EigenMatrix v){ return v; }

enum SCF_t{ lbfgs_t, newton_t, arh_t };
template <SCF_t scf_t>
std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteRiemann(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenVector occ_guess, EigenMatrix Z,
		int output, int nthreads){
	double E = 0;
	const int nbasis = Z.cols();
	EigenMatrix C = EigenZero(nbasis, nbasis);
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;

	// ARH hessian related
	AugmentedRoothaanHall arh;
	if constexpr ( scf_t == arh_t ) arh.Init(20, 1);

	// Exact hessian related
	EigenMatrix FoverC = EigenZero(0, 0);
	EigenMatrix FoverOcc = EigenZero(0, 0);
	if constexpr ( scf_t == newton_t ){
		FoverC.resize(nbasis, nbasis);
		FoverOcc.resize(nbasis, nbasis);
	}

	EigenVector all_occ = occ_guess;
	all_occ.resize(nbasis, 1);
	std::tuple<int, int, int> n = Regularize(all_occ, 1e-6);
	auto& [ni, na, nv] = n;

	EigenVector epsilons = EigenZero(nbasis, 1);
	EigenMatrix Fprime = EigenZero(nbasis, nbasis);

	Maniverse::PreconFunc cfunc_newton = [&](std::vector<EigenMatrix> Xs, int /*order*/){
		Cprime = Xs[0];
		const EigenMatrix Cprime_thin = Cprime(Eigen::placeholders::all, Eigen::seqN(ni, na));
		const EigenMatrix Occ = na != 0 ? Xs[1] : EigenZero(0, 0);
		if (output>0){
			std::printf("Fractional occupation:");
			for ( int i = 0; i < na; i++ ) std::printf(" %f", Occ(i, 0));
			std::printf("\n");
		}
		for ( int i = 0; i < na; i++ ) if ( Occ(i) > 1. - 1e-6 ){
				all_occ(ni + i) = 1;
				throw SwitchingManifold();
		}
		for ( int i = na - 1; i >= 0; i-- ) if ( Occ(i) < 1e-6 ){
				all_occ(ni + i) = 0;
				throw SwitchingManifold();
		}

		if ( na != 0 ) all_occ(Eigen::seqN(ni, na)) = Occ;
		if (output>0) std::printf("Total number of electrons = %.10f\n", 2 * all_occ.sum());

		const EigenMatrix Dprime_ = Cprime * all_occ.asDiagonal() * Cprime.transpose();
		const EigenMatrix D_ = Z * Dprime_ * Z.transpose();
		const auto [Ghf_, _, __] = int4c2e.ContractInts(D_, EigenZero(0, 0), EigenZero(0, 0), nthreads, 1);
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(D_.rows(), D_.cols());
		if (xc){
			grid.getDensity(2 * D_);
			xc.Evaluate("ev", grid);
			if constexpr ( scf_t == newton_t ) xc.Evaluate("f", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock();
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix F_ = Fhf_ + Gxc_;
		Fprime = Z.transpose() * F_ * Z;
		
		eigensolver.compute(Fprime);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();

		// ARH hessian related
		if constexpr ( scf_t == arh_t ) arh.Append(Dprime_, Fprime);

		const EigenArray ns = Occ.array();
		const double E_ =
			( D_ * ( Hcore + Fhf_ ) ).trace() + Exc_
			+ 2 * T * ( ns.pow(ns).log() + ( 1. - ns ).pow( 1. - ns ).log() ).sum()
			- 2 * Mu * ( ns.sum() + ni );

		const EigenMatrix GradC = 4 * Fprime * Cprime * all_occ.asDiagonal();
		const EigenVector CtFC = ( Cprime.transpose() * Fprime * Cprime ).diagonal();
		const EigenMatrix GradOcc1 = 2 * CtFC(Eigen::seqN(ni, na));
		const EigenMatrix GradOcc2 = ( - 2 * T * ( 1. / ns - 1. ).log() - 2 * Mu ).matrix();
		std::vector<EigenMatrix> Grad = {GradC};
		if ( na != 0 ) Grad.push_back(GradOcc1 + GradOcc2);

		// Preconditioner
		// https://doi.org/10.1016/j.cpc.2016.06.023
		EigenMatrix A = EigenMatrix::Ones(nbasis, nbasis);
		for ( int i = 0; i < nbasis; i++ ){
			for ( int j = i; j < nbasis; j++ ){
				const double occ_diff = all_occ(i) - all_occ(j);
				if ( std::abs(occ_diff) > 1e-6 ) A(i, j) = A(j, i) = - 2 * ( epsilons(i) - epsilons(j) ) * occ_diff;
			}
		}
		std::function<EigenMatrix (EigenMatrix)> DummyC = [nbasis](EigenMatrix /*Z*/){ return EigenZero(nbasis, nbasis); };
		std::function<EigenMatrix (EigenMatrix)> DummyOcc = [na](EigenMatrix /*Z*/){ return EigenZero(na, 1); };
		if constexpr ( scf_t == lbfgs_t ){
			const EigenMatrix Asqrt = A.cwiseSqrt();
			const EigenMatrix Asqrtinv = Asqrt.cwiseInverse();
			std::function<EigenMatrix (EigenMatrix)> Psqrt = [Cprime, Asqrtinv](EigenMatrix Z){
				EigenMatrix Omega = Cprime.transpose() * Z;
				Omega = Omega.cwiseProduct(Asqrtinv);
				return ( Cprime * Omega ).eval();
			};
			std::function<EigenMatrix (EigenMatrix)> Psqrtinv = [Cprime, Asqrt](EigenMatrix Z){
				EigenMatrix Omega = Cprime.transpose() * Z;
				Omega = Omega.cwiseProduct(Asqrt);
				return ( Cprime * Omega ).eval();
			};
			std::vector<std::function<EigenMatrix (EigenMatrix)>> precon = {Psqrt};
			std::vector<std::function<EigenMatrix (EigenMatrix)>> inv_precon = {Psqrtinv};
			if ( na != 0 ){
				precon.push_back(DummyC);       inv_precon.push_back(DummyC);
				precon.push_back(DummyOcc);     inv_precon.push_back(DummyOcc);
				precon.push_back(IdentityFunc); inv_precon.push_back(IdentityFunc);
			}
			return std::make_tuple(E_, Grad, precon, inv_precon);
		}else if constexpr ( scf_t == newton_t || scf_t == arh_t){
			std::vector<std::function<EigenMatrix (EigenMatrix)>> He;
			He.push_back([Z, all_occ, Fprime, Cprime, &int4c2e, &xc, &grid, nthreads, &arh, &FoverC](EigenMatrix vprime){
					EigenMatrix Dprime = Cprime * all_occ.asDiagonal() * vprime.transpose();
					Dprime += Dprime.transpose().eval();
					if constexpr ( scf_t == newton_t ){
						const EigenMatrix D = Z * Dprime * Z.transpose();
						const auto [FhfU, _, __] = int4c2e.ContractInts(D, EigenZero(0, 0), EigenZero(0, 0), nthreads, 0);
						EigenMatrix FxcU = EigenZero(D.rows(), D.cols());
						if (xc){
							std::vector<Eigen::Tensor<double, 1>> RhoUss, SigmaUss;
							std::vector<Eigen::Tensor<double, 2>> Rho1Uss;
							grid.getDensityU({2*D}, RhoUss, Rho1Uss, SigmaUss);
							FxcU = grid.getFockU(RhoUss, Rho1Uss, SigmaUss)[0];
						}
						const EigenMatrix FU = FhfU + FxcU;
						FoverC = Z.transpose() * FU * Z;
					}else{
						FoverC = arh.Hessian(Dprime);
					}
					const EigenMatrix res = 4 * ( 
							FoverC * Cprime
							+ Fprime * vprime
					) * all_occ.asDiagonal();
					return res;
			});
			if ( na != 0 ) He.push_back([ni, na, Z, Cprime, Cprime_thin, all_occ, Fprime, &int4c2e, &xc, &grid, nthreads, &arh, &FoverOcc](EigenMatrix gamma){
					const EigenMatrix Dprime = Cprime_thin * gamma.asDiagonal() * Cprime_thin.transpose();
					if constexpr ( scf_t == newton_t ){
						const EigenMatrix D = Z * Dprime * Z.transpose();
						const auto [FhfU, _, __] = int4c2e.ContractInts(D, EigenZero(0, 0), EigenZero(0, 0), nthreads, 0);
						EigenMatrix FxcU = EigenZero(D.rows(), D.cols());
						if (xc){
							std::vector<Eigen::Tensor<double, 1>> RhoUss, SigmaUss;
							std::vector<Eigen::Tensor<double, 2>> Rho1Uss;
							grid.getDensityU({2*D}, RhoUss, Rho1Uss, SigmaUss);
							FxcU = grid.getFockU(RhoUss, Rho1Uss, SigmaUss)[0];
						}
						const EigenMatrix FU = FhfU + FxcU;
						FoverOcc = Z.transpose() * FU * Z;
					}else{
						FoverOcc = arh.Hessian(Dprime);
					}
					EigenMatrix all_gamma = EigenZero(Dprime.rows(), 1);
					all_gamma(Eigen::seqN(ni, na), 0) = gamma;
					const EigenMatrix res = 4 * (
							Fprime * Cprime * all_gamma.asDiagonal()
							+ FoverOcc * Cprime * all_occ.asDiagonal()
					);
					return res;
			});
			if ( na != 0 ) He.push_back([ni, na, Cprime, Fprime, &FoverC](EigenMatrix Delta){
					const EigenVector res = 2 * (
							2 * Delta.transpose() * Fprime * Cprime
							+ Cprime.transpose() * FoverC * Cprime
					).diagonal();
					return res(Eigen::seqN(ni, na)).eval();
			});
			if ( na != 0 ) He.push_back([T, ni, na, Cprime_thin, ns, Fprime, &FoverOcc](EigenMatrix gamma){
					const EigenVector res = 2 * (
							( Cprime_thin.transpose() * FoverOcc * Cprime_thin ).diagonal()
							- ( T / ns / ( ns - 1. ) ).matrix().cwiseProduct(gamma)
					);
					return res;
			});
			const EigenMatrix Ainv = A.cwiseInverse();
			const std::function<EigenMatrix (EigenMatrix)> Pr = [Cprime, Ainv](EigenMatrix Z){
				EigenMatrix Omega = Cprime.transpose() * Z;
				Omega = Omega.cwiseProduct(Ainv);
				return ( Cprime * Omega ).eval();
			};
			std::vector<std::function<EigenMatrix (EigenMatrix)>> precon = {Pr};
			if ( na != 0 ){
				precon.push_back(DummyC);
				precon.push_back(DummyOcc);
				precon.push_back(IdentityFunc);
			}
			return std::make_tuple(E_, Grad, He, precon);
		}
	};

	std::tuple<double, double, double> tol = {1.e-3, 1.e-2, 1.e-2};

	label:
	Maniverse::Flag flag(Cprime);
	std::vector<int> spaces;
	n = Regularize(all_occ, 1e-6);
	if ( ni > 0 ) spaces.push_back(ni);
	for ( int i = 0; i < na; i++ ) spaces.push_back(1);
	if ( nv > 0 ) spaces.push_back(nv);
	flag.setBlockParameters(spaces);
	std::vector<std::shared_ptr<Maniverse::Manifold>> ms = {flag.Clone()};
	if ( na > 0 ){
		Maniverse::Euclidean euclidean(all_occ(Eigen::seqN(ni, na)));
		ms.push_back(euclidean.Clone());
	}
	Maniverse::Iterate M(ms, 1);
	std::tuple<int, int, int> n_old = n;
	auto& [ni_old, na_old, nv_old] = n_old;

	try{
		if constexpr ( scf_t == lbfgs_t ){
			if ( ! Maniverse::LBFGS(
						cfunc_newton, tol,
						20, 300, E, M, output
			) ) throw std::runtime_error("Convergence failed!");
		}else{
			double ratio = 0;
			if constexpr ( scf_t == newton_t ) ratio = 0.001;
			else ratio = 0.01;
			Maniverse::TrustRegionSetting tr_setting;
			if ( ! Maniverse::TrustRegion(
						cfunc_newton, tr_setting, tol,
						ratio, 1, 300, E, M, output
			) ) throw std::runtime_error("Convergence failed!");
		}
	}catch (SwitchingManifold&){
		if (output) std::printf("Switching manifold for smaller active space!\n");
		goto label;
	}
	EigenVector all_occ_new = FermiDirac(epsilons, T, Mu, 0);
	n = Regularize(all_occ_new, 1e-6);
	if ( n_old != n ){
		if ( ni_old > ni ){
			const EigenMatrix Ci = Cprime.leftCols(ni_old);
			const EigenMatrix Fi = Ci.transpose() * Fprime * Ci;
			eigensolver.compute(Fi);
			Cprime.leftCols(ni_old) = Ci * eigensolver.eigenvectors();
			all_occ(ni_old - 1) = 0.99;
		}
		if ( nv_old > nv ){
			const EigenMatrix Cv = Cprime.rightCols(nv_old);
			const EigenMatrix Fv = Cv.transpose() * Fprime * Cv;
			eigensolver.compute(Fv);
			Cprime.rightCols(nv_old) = Cv * eigensolver.eigenvectors();
			all_occ(nbasis - nv_old) = 0.01;
		}
		if (output) std::printf("Switching manifold due to inconsistent orbital energy and occupation!\n");
		goto label;
	}
	if ( tol != std::make_tuple(1e-8, 1e-5, 1e-5) ){
		if (output) std::printf("Switching to higher convergence precision!\n");
		tol = {1e-8, 1e-5, 1e-5};
		goto label;
	}

	return std::make_tuple(E, epsilons, all_occ, C);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteLBFGS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenVector occ_guess, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedFiniteRiemann<lbfgs_t>(T, Mu, int2c1e, int4c2e, xc, grid, Cprime, occ_guess, Z, output, nthreads);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteNewton(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenVector occ_guess, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedFiniteRiemann<newton_t>(T, Mu, int2c1e, int4c2e, xc, grid, Cprime, occ_guess, Z, output, nthreads);
}

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteARH(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenVector occ_guess, EigenMatrix Z,
		int output, int nthreads){
	return RestrictedFiniteRiemann<arh_t>(T, Mu, int2c1e, int4c2e, xc, grid, Cprime, occ_guess, Z, output, nthreads);
}
