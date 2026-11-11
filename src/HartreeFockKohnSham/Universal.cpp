#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <Maniverse/Manifold/Manifold.h>

#include "../Macro.h"
#include "../Integral.h"
#include "../Grid.h"
#include "../ExchangeCorrelation.h"

#include "Universal.h"

#define Hcore ( int2c1e->Kinetic + int2c1e->Nuclear )

std::function<double (double)> SafeInverse = [](double x){
	return x == 0 ? 0 : 1. / x;
};

UniversalObjBase::UniversalObjBase(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		std::vector<int> Norbs, double Coupling,
		int nthreads
): int2c1e(&int2c1e), int4c2e(&int4c2e), xc(&xc), grid(&grid), Norbs(Norbs), Coupling(Coupling), nthreads(nthreads){
	Eigen::SelfAdjointEigenSolver<EigenMatrix> es(int2c1e.Overlap);
	Ssqrt = es.operatorSqrt();
	Ssqrtinv = es.operatorInverseSqrt();
	nbasis = Ssqrt.rows();
	for ( int type = 0; type < 3; type++ ){
		Cs.push_back(EigenZero(nbasis, Norbs[type]));
		Gradients.push_back(EigenZero(nbasis, Norbs[type]));
		Ds.push_back(EigenZero(0, 0));
		Fs.push_back(EigenZero(0, 0));
		Ccanos.push_back(EigenZero(nbasis, Norbs[type]));
		epss.push_back((EigenZero(Norbs[type], 1)));
	}
}

void UniversalObjBase::calcInternal(std::vector<int> derivatives){ // Used by R and U; Overridden by two-det.
	if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
		const auto [J, Kd, Ka, Kb] = int4c2e->ContractInts(Ds[0], Ds[1], Ds[2], nthreads, 1);
		Value = 0;
		for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
			const EigenMatrix Fhf =
				type == 0 ? ( Hcore + J - Kd - 0.5 * Ka - 0.5 * Kb ).eval() :
				type == 1 ? ( Hcore + J - Kd - Ka + Coupling * Kb ).eval() :
				/* type == 2 ? */ ( Hcore + J - Kd - Kb + Coupling * Ka ).eval();
			Value += 0.5 * occ[type] * Ds[type].cwiseProduct( Hcore + Fhf ).sum();
			Fs[type] = Fhf;
		}

		if (*xc){
			std::vector<EigenMatrix> Dab(2, EigenZero(nbasis, nbasis));
			for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
				if ( type == 0 || type == 1 ) Dab[0] += Ds[type];
				if ( type == 0 || type == 2 ) Dab[1] += Ds[type];
			}
			if ( Norbs[0] && !Norbs[1] && !Norbs[2] ) Dab = { 2 * Dab[0] };
			grid->getDensity(Dab);
			xc->Evaluate("ev", *grid);
			Value += grid->getEnergy();
		}
	}

	if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
		if (*xc){
			const std::vector<EigenMatrix> Fxcab = grid->getFock();
			if ( Norbs[0] && !Norbs[1] && !Norbs[2] ){
				Fs[0] += Fxcab[0];
			}else for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
				Fs[type] += (
					type == 0 ? ( ( Fxcab[0] + Fxcab[1] ) / 2 ).eval() :
					type == 1 ? Fxcab[0] :
					/* type == 2 ? */ Fxcab[1]
				);
			}
		}
	}
}

void UniversalObjBase::makePrecon(){ // Used by R and two-det; Overridden by U
	const int Np = Norbs[0]; const int Na = Norbs[1]; const int Nb = Norbs[2];
	const int Nv = nbasis - Np - Na - Nb;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> es;
	for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
		const EigenMatrix& Ct = Cs[type];
		const EigenMatrix Ft = Ct.transpose() * Fs[type] * Ct;
		es.compute(Ft);
		Ccanos[type] = Ct * es.eigenvectors();
		epss[type] = es.eigenvalues();
	}
	EigenMatrix Ccano = concatenate_mat({Ccanos[0], Ccanos[1], Ccanos[2]});
	{
		Eigen::HouseholderQR<EigenMatrix> qr(Ssqrt * Ccano);
		const EigenMatrix C_perp = (Ssqrtinv * qr.householderQ()).rightCols(Nv);
		const EigenMatrix Fv = C_perp.transpose() * Fs[0] * C_perp;
		es.compute(Fv);
		Ccanos[3] = C_perp * es.eigenvectors();
		Ccano = concatenate_mat({Ccano, Ccanos[3]});
		epss[3] = es.eigenvalues();
	}
	std::vector<EigenVector> Fmodiags(4, EigenZero(nbasis, 1));
	for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
		Fmodiags[type] = ( Ccano.transpose() * Fs[type] * Ccano ).diagonal();
	}
	EigenMatrix A = EigenZero(nbasis, nbasis);
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
			if ( I == J ) continue;
			const double FIi = Fmodiags[I](i) * Iscale;
			const double FIj = Fmodiags[I](j) * Iscale;
			const double FJi = Fmodiags[J](i) * Jscale;
			const double FJj = Fmodiags[J](j) * Jscale;
			A(i, j) = std::abs(FIj + FJi - FIi - FJj);
		}
	}
	Ks = { A.topLeftCorner(Np + Na + Nb, Np + Na + Nb) };
	Ls = { A.bottomLeftCorner(Nv, Np + Na + Nb) };
}

void UniversalObjBase::CalculateAllCs(std::vector<int> derivatives){
	if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
		for ( int type = 0; type < 3; type++ ){
			if ( Norbs[type] ) Ds[type] = Cs[type] * Cs[type].transpose();
			else Ds[type] = EigenZero(0, 0);
		}
	}
	calcInternal(derivatives);
	if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
		makePrecon();
		for ( int type = 0; type < 3; type++ ) if ( Norbs[type] ){
			Gradients[type] = 2 * occ[type] * Fs[type] * Cs[type];
		}
	}
}

EigenMatrix UniversalPreconditioner(EigenMatrix U, EigenMatrix Uperp, EigenMatrix K, EigenMatrix L, EigenMatrix SV){
	EigenMatrix Omega = U.transpose() * SV;
	Omega = Omega.cwiseProduct(K);
	EigenMatrix Kappa = Uperp.transpose() * SV;
	Kappa = Kappa.cwiseProduct(L);
	return ( U * Omega + Uperp * Kappa ).eval();
}

EigenMatrix concatenate_mat(std::vector<EigenMatrix> mats){
	const int nrows = mats[0].rows();
	const bool same_rows = std::all_of(
		mats.begin(), mats.end(),
		[nrows](const EigenMatrix& mat){ return mat.rows() == nrows; }
	);
	if ( !same_rows ) throw std::runtime_error("concatenate_mat: Matrices must have the same number of rows");
	int ncols = 0;
	for ( EigenMatrix& mat : mats ) ncols += mat.cols();
	EigenMatrix bigmat(nrows, ncols);
	for ( int i = 0, j = 0; i < (int)mats.size(); i++ ){
		bigmat.middleCols(j, mats[i].cols()) = mats[i];
		j += mats[i].cols();
	}
	return bigmat;
}

EigenVector concatenate_vec(std::vector<EigenVector> vecs){
	Eigen::Index size = 0;
	for (const EigenVector& vec : vecs) size += vec.size();
	EigenVector result(size);
	Eigen::Index offset = 0;
	for (const EigenVector& vec : vecs){
		result.segment(offset, vec.size()) = vec;
		offset += vec.size();
	}
	return result;
}