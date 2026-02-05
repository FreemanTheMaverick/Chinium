#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <functional>
#include <string>
#include <cstdio>
#include <chrono>
#include <memory>
#include <Maniverse/Manifold/Simplex.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>

#include "../Macro.h"
#include "ADIIS.h"
#include "QuadraticObj.h"

ADIIS::ADIIS(
		std::function<std::tuple<
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
			> (std::vector<EigenMatrix>&, std::vector<bool>&)
		>* update_func,
		int nmatrices, int max_size, double tolerance,
		int max_iter, int verbose): DIIS(update_func, nmatrices, max_size, tolerance, max_iter, verbose){
	this->Name = "Augmented-DIIS (ADIIS)";
}

#define Es(i) this->Auxiliariess[index][i](0, ncols - 1)
#define Ds(i) this->Auxiliariess[index][i].leftCols(ncols - 1)
#define Fs(i) this->Updatess[index][i]
#define E this->Auxiliariess[index].back()(0, ncols - 1)
#define D this->Auxiliariess[index].back().leftCols(ncols - 1)
#define F this->Updatess[index].back()

EigenVector ADIIS::Extrapolate(int index){
	const int size = this->getCurrentSize();
	EigenMatrix A = EigenZero(size, 1);
	EigenMatrix B = EigenZero(size, size);
	for ( int i = 0; i < size; i++ ){
		const int ncols = this->Auxiliariess[index][i].cols();
		A(i) = Dot(Ds(i) - D, F);
		for ( int j = 0; j < size; j++ )
			B(i, j) = Dot(Ds(i) - D, Fs(j) - F);
	}
	QuadraticObj obj(A, B);
	EigenMatrix p = EigenZero(size, 1); p(size - 1) = 0.9;
	for ( int i = 0; i < size - 1; i++ )
		p(i) = ( 1. - p(size - 1) ) / ( size - 1 );
	Maniverse::Simplex simplex(p);
	Maniverse::Iterate M(obj, {simplex.Share()}, 1);

	if ( this->Verbose > 1 ) std::printf("| Calling Maniverse for optimization on the Simplex manifold\n");
	Maniverse::TrustRegion tr;
	const bool converged = Maniverse::TruncatedNewton(
				M, tr, {1.e-4, 1.e-6, 1e-2},
				0.001, 1000, 0
	);
	if ( this->Verbose > 1 && !converged ){
		std::printf("| Warning: Optimization of ADIIS weights did not fully converged!\n");
		std::printf("| Warning: This is probably fine if SCF convergence is met later.\n");
	}

	return (EigenVector)M.Point;
}
