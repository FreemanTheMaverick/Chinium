#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <string>
#include <cstdio>
#include <chrono>
#include <memory>
#include <Maniverse/Manifold/Simplex.h>
#include <Maniverse/Optimizer/TrustRegion.h>

#include "../Macro.h"
#include "EDIIS.h"


EDIIS::EDIIS(
		std::function<std::tuple<
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
			> (std::vector<EigenMatrix>&, std::vector<bool>&)
		>* update_func,
		int nmatrices, int max_size, double tolerance,
		int max_iter, int verbose): DIIS(update_func, nmatrices, max_size, tolerance, max_iter, verbose){
	this->Name = "Energy-DIIS (EDIIS)";
}

#define Es(i) this->Auxiliariess[index][i](0, ncols - 1)
#define Ds(i) this->Auxiliariess[index][i].leftCols(ncols - 1)
#define Fs(i) this->Updatess[index][i]

EigenVector EDIIS::Extrapolate(int index){
	const int size = this->getCurrentSize();
	EigenMatrix A = EigenZero(size, 1);
	EigenMatrix B = EigenZero(size, size);
	for ( int i = 0; i < size; i++ ){
		const int ncols = this->Auxiliariess[index][i].cols();
		A(i) = Es(i) / 2;
		for ( int j = 0; j < size; j++ )
			B(i, j) = - Dot(Ds(i) - Ds(j), Fs(i) - Fs(j));
	}
	EigenMatrix p = EigenZero(size, 1); p(size - 1) = 0.9;
	for ( int i = 0; i < size - 1; i++ )
		p(i) = ( 1. - p(size - 1) ) / ( size - 1 );
	Simplex M = Simplex(p, 1);

	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		> (EigenMatrix, int)
	> func = [A, B](EigenMatrix x_, int order){
		const double L = Dot(x_, A + 0.5 * B * x_);
		const EigenMatrix G = A + B * x_;
		std::function<EigenMatrix (EigenMatrix)> H = [](EigenMatrix v__){ return v__; };
		if ( order == 2) H = [B](EigenMatrix v__){
			return (EigenMatrix)(B * v__);
		};
		return std::make_tuple(L, G, H);
	};

	if( this->Verbose > 1 ) std::printf("| Calling Maniverse for optimization on the Simplex manifold\n");
	double L = 0;
	TrustRegionSetting tr_setting;
	const bool converged = TrustRegion(
				func, tr_setting, {1.e-4, 1.e-6, 1e-2},
				0.001, 1, 1000, L, M, 0
	);
	if ( this->Verbose > 1 && !converged ){
		std::printf("| Warning: Optimization of EDIIS weights did not fully converged!\n");
		std::printf("| Warning: This is probably fine if SCF convergence is met later.\n");
	}

	return (EigenVector)M.P;
}
