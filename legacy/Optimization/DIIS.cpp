#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <string>
#include <cstdio>

#include "../Macro.h"
#include "../Manifold/Simplex.h"

#include "Ox.h"

#include <iostream>

EigenMatrix CDIIS(std::deque<EigenMatrix>& Gs, std::deque<EigenMatrix>& Fs){
	const int size = Fs.size();
	EigenMatrix B = EigenZero(size + 1, size + 1);
	EigenVector b(size + 1); b.setZero();
	b[size] = -1;
	for ( int i = 0; i < size; i++ ){
		for ( int j = 0; j <= i; j++ ){
			const double bij = (Gs[i].transpose() * Gs[j]).trace();
			B(i, j) = bij;
			B(j, i) = bij;
		}
		B(i, size) = -1;
		B(size, i) = -1;
		b[i] = 0;
	}
	EigenVector x = B.colPivHouseholderQr().solve(b);
	EigenMatrix F = EigenZero(Fs[0].rows(), Fs[0].cols());
	for ( int i = 0; i < size; i++ )
		F += x(i) * Fs[i];
	return F;
}

EigenMatrix ADIIS(EigenVector A, EigenMatrix B, std::deque<EigenMatrix>& Fs, int output){
	if (output > 0) std::printf("Augmented DIIS\n");
	const int n = A.size();
	EigenMatrix p = EigenZero(n, 1); p(n - 1) = 0.9;
	for ( int i = 0; i < n - 1; i++ )
		p(i) = ( 1. - p(n - 1) ) / ( n - 1 );
	Simplex M = Simplex(p);

	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		> (EigenMatrix)
	> func = [A, B](EigenMatrix x_){
		const double L = ( x_.transpose() * ( A + 0.5 * B * x_ ) ).trace();
		const EigenMatrix G = A + B * x_;
		const std::function<EigenMatrix (EigenMatrix)> H = [B](EigenMatrix v__){
			return EigenMatrix(B * v__);
		};
		return std::make_tuple(L, G, H);
	};

	double L = 0;
	//assert( Ox( func, {1.e2, 1.e-6, 1e-2}, 100000, L, M, output-1) && "Convergence failed!" );
	Ox( func, {1.e2, 1.e-6, 1e-2}, 100000, L, M, output-1);

	if (output > 0) std::printf("Converged!\n");
	EigenMatrix F = EigenZero(Fs[0].rows(), Fs[0].cols());
	for ( int i = 0; i < n; i++ )
		F += M.P(i) * Fs[i];
	return F;
}


