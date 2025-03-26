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
		int max_iter, bool verbose){
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

bool DIIS::Run(std::vector<EigenMatrix>& Ms){
	const int nmatrices = this->getNumMatrices();
	assert(
			nmatrices == (int)Ms.size() &&
			"Inconsistent numbers of matrices!"
	);
	if (this->Verbose){
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
		if (this->Verbose) std::printf("Iteration %d\n", iiter);

		std::vector<EigenMatrix> updates(nmatrices);
		std::vector<EigenMatrix> residuals(nmatrices);
		std::vector<EigenMatrix> auxiliaries(nmatrices);
		if ( iiter == 0 && this->getCurrentSize() > 0 ){
			if (this->Verbose) std::printf("Hot restart -> Skipping computation in the first iteration\n");
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
		double max_max_residual = 0;
		for ( int i = 0; i < nmatrices; i++ ){
			const double max_residual = residuals[i].cwiseAbs().maxCoeff();
			if ( i == 0 ) max_max_residual = max_residual;
			else max_max_residual = max_max_residual > max_residual ? max_max_residual : max_residual;
			dones[i] = max_residual < this->Tolerance;
			if ( dones[i] ) num_dones++;
		}
		converged = max_max_residual < this->Tolerance;
		if (this->Verbose){
			std::printf("Maximal residual element = %E\n", max_max_residual);
			std::printf("%d out of %d matrices have converged.\n", num_dones, nmatrices);
		}

		if (converged){
			if (this->Verbose) std::printf("Done!\n");
		}else{
			const int current_size = this->getCurrentSize();
			if (this->Verbose){
				std::printf("Current DIIS space size: %d\n", current_size);
			}
			if ( current_size < 2 ){
				if (this->Verbose) std::printf("Naive update\n");
				Ms = updates;
			}else{
				for ( int mat = 0; mat < nmatrices; mat++ ) if ( !dones[mat] ){
					if (this->Verbose) std::printf("%s extrapolation on Matrix %d\n", this->Name.c_str(), mat);
					const EigenVector Ws = this->Extrapolate(mat);
					Ms[mat].setZero();
					for ( int i = 0; i < current_size; i++ )
						Ms[mat] += Ws[i] * Updatess[mat][i];
					if (this->Verbose){
						std::printf("DIIS weight:");
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
