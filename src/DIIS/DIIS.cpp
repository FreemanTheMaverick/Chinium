#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <string>
#include <cstdio>
#include <chrono>

#include "../Macro.h"
#include "DIIS.h"

DIIS::DIIS(
		std::function<std::tuple<
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
			> (std::vector<EigenMatrix>&, std::vector<bool>&)
		>* update_func,
		int nmatrices, int max_size, double tolerance,
		int max_iter, int verbose){
	this->Name = "DIIS";
	this->Verbose = verbose;
	this->MaxSize = max_size;
	this->Tolerance = tolerance;
	this->MaxIter = max_iter;
	this->UpdateFunc = update_func;
	this->Updatess.resize(nmatrices);
	this->Residualss.resize(nmatrices);
	this->Auxiliariess.resize(nmatrices);
}

int DIIS::getNumMatrices(){
	return (int)this->Updatess.size();
}

int DIIS::getCurrentSize(){
	return (int)this->Updatess[0].size();
}

void DIIS::Steal(DIIS& another_diis){
	assert(this->getNumMatrices() == another_diis.getNumMatrices() && "Different numbers of co-optimized matrices!");
	this->Updatess = another_diis.Updatess;
	this->Residualss = another_diis.Residualss;
	this->Auxiliariess = another_diis.Auxiliariess;
	if ( this->MaxSize < another_diis.getCurrentSize() ){
		const int remove = another_diis.getCurrentSize() - this->MaxSize;
		for ( int imat = 0; imat < this->getNumMatrices(); imat++ ){
			this->Updatess[imat].erase(this->Updatess[imat].begin(), this->Updatess[imat].begin() + remove);
			this->Residualss[imat].erase(this->Residualss[imat].begin(), this->Residualss[imat].begin() + remove);
			this->Auxiliariess[imat].erase(this->Auxiliariess[imat].begin(), this->Auxiliariess[imat].begin() + remove);
		}
	}
}

void DIIS::Append(
		std::vector<EigenMatrix>& updates,
		std::vector<EigenMatrix>& residuals,
		std::vector<EigenMatrix>& auxiliaries){
	const int nmatrices = this->getNumMatrices();
	assert(
			(int)updates.size() == nmatrices &&
			(int)residuals.size() == nmatrices &&
			(int)auxiliaries.size() == nmatrices &&
			"Inconsistent numbers of matrices!"
	);
	if ( this->getCurrentSize() == this->MaxSize ){
		for ( int i = 0; i < nmatrices; i++ ){
			this->Updatess[i].pop_front();
			this->Residualss[i].pop_front();
			this->Auxiliariess[i].pop_front();
		}
	}
	for ( int i = 0; i < nmatrices; i++ ){
		this->Updatess[i].push_back(updates[i]);
		this->Residualss[i].push_back(residuals[i]);
		this->Auxiliariess[i].push_back(auxiliaries[i]);
	}
}

static double FindDamp(double max_residual, std::vector<std::tuple<double, double, double>> damps){
	double this_mat_damp = 0;
	for ( auto [lower, upper, damp] : damps ){
		if ( lower < max_residual && max_residual < upper )
			this_mat_damp = damp;
	}
	return this_mat_damp;
}

bool DIIS::Run(std::vector<EigenMatrix>& Ms){
	const int nmatrices = this->getNumMatrices();
	assert(
			nmatrices == (int)Ms.size() &&
			"Inconsistent numbers of matrices!"
	);
	if (this->Verbose > 0){
		std::printf("************** Direct Inversion in the Iterative Subspace **************\n\n");
		std::printf("DIIS type: %s\n", this->Name.c_str());
		std::printf("Maximum number of iterations: %d\n", this->MaxIter);
		std::printf("Maximum size of DIIS space: %d\n", this->MaxSize);
		std::printf("Number of co-optimized matrices: %d\n", nmatrices);
		std::printf("Convergence threshold for residual : %E\n\n", this->Tolerance);
	}

	std::vector<bool> dones(nmatrices);
	bool converged = 0;
	const auto all_start = __now__;

	for ( int iiter = 0; iiter < this->MaxIter && ( ! converged ); iiter++ ){

		const auto iter_start = __now__;
		if (this->Verbose > 0) std::printf("Iteration %d\n", iiter);

		std::vector<EigenMatrix> updates(nmatrices);
		std::vector<EigenMatrix> residuals(nmatrices);
		std::vector<EigenMatrix> auxiliaries(nmatrices);
		if ( iiter == 0 && this->getCurrentSize() > 0 ){
			if (this->Verbose > 0) std::printf("Hot restart -> Skipping computation in the first iteration\n");
			for ( int imat = 0; imat < nmatrices; imat++ ){
				updates[imat] = this->Updatess[imat].back();
				residuals[imat] = this->Residualss[imat].back();
				auxiliaries[imat] = this->Auxiliariess[imat].back();
			}
		}else{
			std::tie(updates, residuals, auxiliaries) = (*this->UpdateFunc)(Ms, dones);
			this->Append(updates, residuals, auxiliaries);
		}

		int num_dones = 0;
		std::vector<double> max_residuals(nmatrices);
		double max_max_residual = 0;
		for ( int i = 0; i < nmatrices; i++ ){
			const double max_residual = max_residuals[i] = residuals[i].cwiseAbs().maxCoeff();
			if ( i == 0 ) max_max_residual = max_residual;
			else max_max_residual = max_max_residual > max_residual ? max_max_residual : max_residual;
			dones[i] = max_residual < this->Tolerance;
			if ( dones[i] ) num_dones++;
		}
		converged = max_max_residual < this->Tolerance;
		if (this->Verbose > 0){
			std::printf("%d out of %d matrices have converged.\n", num_dones, nmatrices);
			std::printf("Maximal residual element = %E\n", max_max_residual);
		}

		if (converged){
			if (this->Verbose > 0) std::printf("Done!\n");
		}else{
			const int current_size = this->getCurrentSize();
			if (this->Verbose > 0)
				std::printf("Current DIIS space size: %d -> %s update\n", current_size, current_size < 2 ? "Naive" : "DIIS");
			if ( current_size < 2 ){
				for ( int mat = 0; mat < nmatrices; mat++ ){
					const double max_residual = max_residuals[mat];
					const double damp = FindDamp(max_residual, this->Damps);
					if (this->Verbose > 1){
						std::printf("Naive update on Matrix %d:\n", mat);
						std::printf("| Maximal residual element: %E\n", max_residual);
						std::printf("| Damping factor: %f\n", damp);
					}
					Ms[mat] *= damp;
					Ms[mat] += ( 1. - damp ) * updates[mat];
				}
			}else{
				for ( int mat = 0; mat < nmatrices; mat++ ) if ( !dones[mat] ){
					const double max_residual = max_residuals[mat];
					const double damp = FindDamp(max_residual, this->Damps);
					if (this->Verbose > 1){
						std::printf("%s extrapolation on Matrix %d:\n", this->Name.c_str(), mat);
						std::printf("| Maximal residual element: %E\n", max_residual);
						std::printf("| Damping factor: %f\n", damp);
					}
					Ms[mat] *= damp;
					const EigenVector Ws = this->Extrapolate(mat);
					for ( int i = 0; i < current_size; i++ )
						Ms[mat] += ( 1. - damp ) * Ws[i] * Updatess[mat][i];
					if (this->Verbose > 1){
						std::printf("| DIIS weight:");
						for ( int i = 0; i < current_size; i++ )
							std::printf(" %f", Ws[i]);
						std::printf("\n");
					}
				}
			}
		}

		if (this->Verbose) std::printf("Elapsed time: %f seconds for current iteration; %f seconds in total\n\n", __duration__(iter_start, __now__), __duration__(all_start, __now__));
	}

	return converged;
}
