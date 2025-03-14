#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <string>
#include <cstdio>
#include <memory>
#include <chrono>
#include <Maniverse/Manifold/Simplex.h>
#include <Maniverse/Optimizer/TrustRegion.h>

#include "Macro.h"

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
	Simplex M = Simplex(p, 1);

	std::function<
		std::tuple<
			double,
			EigenMatrix,
			std::function<EigenMatrix (EigenMatrix)>
		> (EigenMatrix, int)
	> func = [A, B](EigenMatrix x_, int order){
		const double L = ( x_.transpose() * ( A + 0.5 * B * x_ ) ).trace();
		const EigenMatrix G = A + B * x_;
		std::function<EigenMatrix (EigenMatrix)> H = [](EigenMatrix v__){ return v__; };
		if ( order == 2) H = [B](EigenMatrix v__){
			return EigenMatrix(B * v__);
		};
		return std::make_tuple(L, G, H);
	};

	double L = 0;
	TrustRegionSetting tr_setting;
	assert(
			TrustRegion(
				func, tr_setting, {1.e-4, 1.e-6, 1e-2},
				0.001, 1, 100000, L, M, output-1
			) && "Convergence Failed!"
	);

	EigenMatrix F = EigenZero(Fs[0].rows(), Fs[0].cols());
	for ( int i = 0; i < n; i++ )
		F += M.P(i) * Fs[i];
	return F;
}

bool GeneralizedDIIS(
		std::function<
			std::vector<std::tuple< // Multiple variables
					double, EigenMatrix, EigenMatrix // Objective, Residual, Next step
			>>
			(std::vector<EigenMatrix>&)
		>& RawUpdate,
		double tolerance, int max_size, int max_iter,
		double& E, std::vector<EigenMatrix>& M, int output,
		std::function<
			std::vector<EigenMatrix> // Multiple variables
			(
				std::vector<EigenVector>, // DIIS weights
				std::deque<std::vector<EigenMatrix>>&
			)
		>& DiisUpdate){

	if (output > 0){
		std::printf("********* Generalized Direct Inversion in the Iterative Subspace *********\n\n");
		std::printf("Maximum number of iterations: %d\n", max_iter);
		std::printf("Maximum size of DIIS space: %d\n", max_size);
		std::printf("Number of co-optimized matrices: %d\n", (int)M.size());
		std::printf("Convergence threshold for residual : %E\n\n", tolerance);
	}

	std::deque<std::vector<double>> Objective;
	std::deque<std::vector<EigenMatrix>> Residual;
	std::deque<std::vector<EigenMatrix>> After;
	std::vector<EigenVector> Weight;
	const int nmatrices = M.size();

	E = 0;
	double ResiNorm = 0;
	bool converged = 0;
	const auto all_start = __now__;

	for ( int iiter = 0; iiter < max_iter && ( ! converged ); iiter++ ){

		const auto iter_start = __now__;
		if (output > 0) std::printf("Iteration %d\n", iiter);

		if ( (int)Objective.size() == max_size ){
			Objective.pop_front();
			Residual.pop_front();
			After.pop_front();
		}

		std::vector<double> objective;
		std::vector<EigenMatrix> residual;
		std::vector<EigenMatrix> after;
		std::vector<std::tuple<double, EigenMatrix, EigenMatrix>> tups = RawUpdate(M);
		for ( auto& tup : tups ){
			objective.push_back(std::get<0>(tup));
			residual.push_back(std::get<1>(tup));
			after.push_back(std::get<2>(tup));
		}
		Objective.push_back(objective);
		Residual.push_back(residual);
		After.push_back(after);
		const int size = Objective.size();

		const double oldE = E;
		E = objective[0];
		for ( double obj : objective ) E = E > obj ? E : obj;
		const double deltaE = E - oldE;
		ResiNorm = residual[0].norm();
		for ( EigenMatrix& res : residual )
			ResiNorm = ResiNorm > res.norm() ? ResiNorm : res.norm();
		if (output > 0){
			std::printf("Target = %.10f\n", E);
			std::printf("Difference in target = %E\n", deltaE);
			std::printf("Maximal residual = %E\n", ResiNorm);
		}

		converged = ResiNorm < tolerance;
		if (converged){
			if (output > 0) std::printf("Converged!\n");
		}else{
			if (output > 0) std::printf("Not converged yet!\n");
			if ( size < 2 ){
				if (output > 0) std::printf("Update: Naive\n");
				M = after;
			}else{
				if (output > 0) std::printf("Update: CDIIS\n");
				Weight.clear();
				for ( int mat = 0; mat < nmatrices; mat++ ){
					EigenMatrix H = EigenZero(size + 1, size + 1);
					EigenVector b = EigenZero(size + 1, 1); b(size) = 1;
					for ( int i = 0; i < size; i++ ){
						H(i, size) = H(size, i) = 1;
						for ( int j = i; j < size; j++ )
							H(i, j) = H(j, i) = Residual[i][mat].cwiseProduct(Residual[j][mat]).sum();
					}
					const EigenVector weight = H.colPivHouseholderQr().solve(b).head(size);
					Weight.push_back(weight);
					if (output > 0){
						std::printf("DIIS weight:");
						for ( int i = 0; i < weight.size(); i++ )
							std::printf(" %f", weight[i]);
						std::printf("\n");
					}
				}
				M = DiisUpdate(Weight, After);
			}
		}

		if (output > 0) std::printf("Elapsed time: %f seconds for current iteration; %f seconds in total\n\n", __duration__(iter_start, __now__), __duration__(all_start, __now__));
	}

	return converged;
}
