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

static std::function<EigenMatrix (EigenMatrix)> Preconditioner(EigenMatrix U, EigenMatrix Uperp, EigenMatrix B, EigenMatrix C){
	return [U, Uperp, B, C](EigenMatrix Z){
		EigenMatrix Omega = U.transpose() * Z;
		Omega = Omega.cwiseProduct(B);
		EigenMatrix Kappa = Uperp.transpose() * Z;
		Kappa = Kappa.cwiseProduct(C);
		return ( U * Omega + Uperp * Kappa ).eval();
	};
}

#define Eta 1e-8
#define Eta2 1e-6

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
	std::tuple<int, int, int> n = Regularize(all_occ, Eta);
	auto& [ni, na, nv] = n;

	EigenVector epsilons = EigenZero(nbasis, 1);
	EigenMatrix Fprime = EigenZero(nbasis, nbasis);
	EigenMatrix& Cprime_out = Cprime;

	Maniverse::PreconFunc cfunc_newton = [&](std::vector<EigenMatrix> Xs, int /*order*/){
		const EigenMatrix Cprime = Cprime_out.leftCols(ni + na) = Xs[0];
		const EigenMatrix Cprime_thin = Cprime.rightCols(na);
		Eigen::HouseholderQR<EigenMatrix> qr(Cprime);
		const EigenMatrix Cprime_perp = Cprime_out.rightCols(nv) = EigenMatrix(qr.householderQ()).rightCols(nv);
		const EigenMatrix thin_occ = na != 0 ? Xs[1] : EigenZero(0, 0);
		EigenMatrix occ = EigenMatrix::Ones(ni + na, 1);
		if ( na != 0 ) occ.bottomRows(na) = thin_occ;
		if (output>0){
			std::printf("Fractional occupation:");
			for ( int i = 0; i < na; i++ ) std::printf(" %f", thin_occ(i, 0));
			std::printf("\n");
		}
		for ( int i = 0; i < na; i++ ) if ( thin_occ(i) > 1. - Eta ){
			all_occ(ni + i) = 1;
			throw SwitchingManifold();
		}
		for ( int i = na - 1; i >= 0; i-- ) if ( thin_occ(i) < Eta ){
			all_occ(ni + i) = 0;
			throw SwitchingManifold();
		}

		if ( na != 0 ) all_occ.head(ni + na) = occ;
		if (output>0) std::printf("Total number of electrons = %.10f\n", 2 * all_occ.sum());

		const EigenMatrix Dprime_ = Cprime * occ.asDiagonal() * Cprime.transpose();
		const EigenMatrix D_ = Z * Dprime_ * Z.transpose();
		const auto [Ghf_, _, __] = int4c2e.ContractInts(D_, EigenZero(0, 0), EigenZero(0, 0), nthreads, 1);
		double Exc_ = 0;
		EigenMatrix Gxc_ = EigenZero(D_.rows(), D_.cols());
		if (xc){
			grid.getDensity({2 * D_});
			xc.Evaluate("ev", grid);
			if constexpr ( scf_t == newton_t ) xc.Evaluate("f", grid);
			Exc_ = grid.getEnergy();
			Gxc_ = grid.getFock()[0];
		}
		const EigenMatrix Fhf_ = Hcore + Ghf_;
		const EigenMatrix F_ = Fhf_ + Gxc_;
		Fprime = Z.transpose() * F_ * Z;

		eigensolver.compute(Fprime);
		epsilons = eigensolver.eigenvalues();
		C = Z * eigensolver.eigenvectors();

		/*{
			eigensolver.compute(Fprime);
			EigenVector occ_new = FermiDirac(eigensolver.eigenvalues(), T, Mu, 0);
			EigenMatrix Cprime_new = eigensolver.eigenvectors();
			EigenMatrix Dprime_new = Cprime_new * occ_new.asDiagonal() * Cprime_new.transpose();
			std::printf("Residual: %E\n", (Dprime_ - Dprime_new).norm());
		}*/

		// ARH hessian related
		if constexpr ( scf_t == arh_t ) arh.Append(Dprime_, Fprime);

		const EigenArray ns = na != 0 ? thin_occ.array() : EigenArray{0};
		const double E_ =
			( D_ * ( Hcore + Fhf_ ) ).trace() + Exc_
			+ 2 * T * ( ns.pow(ns).log() + ( 1. - ns ).pow( 1. - ns ).log() ).sum()
			- 2 * Mu * ( ns.sum() + ni );

		const EigenMatrix GradC = 4 * Fprime * Cprime * occ.asDiagonal();
		const EigenVector CtFC = ( Cprime.transpose() * Fprime * Cprime ).diagonal();
		const EigenMatrix GradOcc1 = 2 * CtFC.tail(na);
		const EigenMatrix GradOcc2 = ( - 2 * T * ( 1. / ns - 1. ).log() - 2 * Mu ).matrix();
		std::vector<EigenMatrix> Grad = {GradC};
		if ( na != 0 ) Grad.push_back(GradOcc1 + GradOcc2);

		// Preconditioner
		// https://doi.org/10.1016/j.cpc.2016.06.023
		EigenMatrix A = EigenMatrix::Ones(nbasis, nbasis);
		for ( int i = 0; i < nbasis; i++ ){
			for ( int j = i; j < nbasis; j++ ){
				const double occ_diff = all_occ(i) - all_occ(j);
				if ( std::abs(occ_diff) > Eta ) A(i, j) = A(j, i) = - 2 * ( epsilons(i) - epsilons(j) ) * occ_diff;
			}
		}
		const EigenMatrix B = A.topLeftCorner(ni + na, ni + na);
		const EigenMatrix C = A.bottomLeftCorner(nv, ni + na);
		const std::function<EigenMatrix (EigenMatrix)> DummyC = [nbasis, ni, na](EigenMatrix /*Z*/){ return EigenZero(nbasis, ni + na); };
		const std::function<EigenMatrix (EigenMatrix)> DummyOcc = [na](EigenMatrix /*Z*/){ return EigenZero(na, 1); };
		const std::function<EigenMatrix (EigenMatrix)> IdentityFunc = [](EigenMatrix v){ return v; };
		if constexpr ( scf_t == lbfgs_t ){
			const EigenMatrix Bsqrt = B.cwiseSqrt();
			const EigenMatrix Bsqrtinv = Bsqrt.cwiseInverse();
			const EigenMatrix Csqrt = C.cwiseSqrt();
			const EigenMatrix Csqrtinv = Csqrt.cwiseInverse();
			const std::function<EigenMatrix (EigenMatrix)> Psqrt = Preconditioner(Cprime, Cprime_perp, Bsqrtinv, Csqrtinv);
			const std::function<EigenMatrix (EigenMatrix)> Psqrtinv = Preconditioner(Cprime, Cprime_perp, Bsqrt, Csqrt);
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
			He.push_back([Z, occ, Fprime, Cprime, &int4c2e, &xc, &grid, nthreads, &arh, &FoverC](EigenMatrix vprime){
					EigenMatrix Dprime = Cprime * occ.asDiagonal() * vprime.transpose();
					Dprime += Dprime.transpose().eval();
					if constexpr ( scf_t == newton_t ){
						const EigenMatrix D = Z * Dprime * Z.transpose();
						const auto [FhfU, _, __] = int4c2e.ContractInts(D, EigenZero(0, 0), EigenZero(0, 0), nthreads, 0);
						EigenMatrix FxcU = EigenZero(D.rows(), D.cols());
						if (xc){
							grid.getDensityU({{2*D}});
							FxcU = grid.getFockU<u_t>()[0][0];
						}
						const EigenMatrix FU = FhfU + FxcU;
						FoverC = Z.transpose() * FU * Z;
					}else{
						FoverC = arh.Hessian(Dprime);
					}
					EigenMatrix res = 4 * ( 
							FoverC * Cprime
							+ Fprime * vprime
					) * occ.asDiagonal();
					return res;
			});
			if ( na != 0 ) He.push_back([na, Z, Cprime, Cprime_thin, occ, Fprime, &int4c2e, &xc, &grid, nthreads, &arh, &FoverOcc](EigenMatrix gamma){
					const EigenMatrix Dprime = Cprime_thin * gamma.asDiagonal() * Cprime_thin.transpose();
					if constexpr ( scf_t == newton_t ){
						const EigenMatrix D = Z * Dprime * Z.transpose();
						const auto [FhfU, _, __] = int4c2e.ContractInts(D, EigenZero(0, 0), EigenZero(0, 0), nthreads, 0);
						EigenMatrix FxcU = EigenZero(D.rows(), D.cols());
						if (xc){
							grid.getDensityU({{2*D}});
							FxcU = grid.getFockU<u_t>()[0][0];
						}
						const EigenMatrix FU = FhfU + FxcU;
						FoverOcc = Z.transpose() * FU * Z;
					}else{
						FoverOcc = arh.Hessian(Dprime);
					}
					EigenMatrix res = FoverOcc * Cprime * occ.asDiagonal();
					res.rightCols(na) += Fprime * Cprime_thin * gamma.asDiagonal();
					res *= 4;
					return res;
			});
			if ( na != 0 ) He.push_back([na, Cprime_thin, Fprime, &FoverC](EigenMatrix Delta){
					const EigenVector res = 2 * (
							2 * Delta.rightCols(na).transpose() * Fprime * Cprime_thin
							+ Cprime_thin.transpose() * FoverC * Cprime_thin
					).diagonal();
					return res;
			});
			if ( na != 0 ) He.push_back([T, Cprime_thin, ns, Fprime, &FoverOcc](EigenMatrix gamma){
					const EigenVector res = 2 * (
							( Cprime_thin.transpose() * FoverOcc * Cprime_thin ).diagonal()
							- ( T / ns / ( ns - 1. ) ).matrix().cwiseProduct(gamma)
					);
					return res;
			});
			const EigenMatrix Binv = B.cwiseInverse();
			const EigenMatrix Cinv = C.cwiseInverse();
			const std::function<EigenMatrix (EigenMatrix)> Pr = Preconditioner(Cprime, Cprime_perp, Binv, Cinv);
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
	std::vector<int> spaces;
	n = Regularize(all_occ, Eta);
	Maniverse::Flag flag(Cprime.leftCols(ni + na));
	if ( ni > 0 ) spaces.push_back(ni);
	for ( int i = 0; i < na; i++ ) spaces.push_back(1);
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
					20, 300, 0.1, 0.75, 100,
					E, M, output
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
	n = Regularize(all_occ_new, Eta);
	if ( n_old != n ){
		if ( ni_old > ni ){
			const EigenMatrix Ci = Cprime.leftCols(ni_old);
			const EigenMatrix Fi = Ci.transpose() * Fprime * Ci;
			eigensolver.compute(Fi);
			Cprime.leftCols(ni_old) = Ci * eigensolver.eigenvectors();
			all_occ(ni_old - 1) = 1. - Eta2;
		}
		if ( nv_old > nv ){
			const EigenMatrix Cv = Cprime.rightCols(nv_old);
			const EigenMatrix Fv = Cv.transpose() * Fprime * Cv;
			eigensolver.compute(Fv);
			Cprime.rightCols(nv_old) = Cv * eigensolver.eigenvectors();
			all_occ(nbasis - nv_old) = Eta2;
		}
		if (output) std::printf("Switching manifold due to inconsistent orbital energy and occupation!\n");
		goto label;
	}
	if ( tol != std::make_tuple(1e-8, 1e-5, 1e-5) ){
		if (output) std::printf("Switching to higher convergence precision!\n");
		tol = {1e-8, 1e-5, 1e-5};
		if constexpr ( scf_t == arh_t ){
			arh.Ps.pop_back();
			arh.Gs.pop_back();
		}
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
