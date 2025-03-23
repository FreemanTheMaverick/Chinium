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
			std::vector<double>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
			> (std::vector<EigenMatrix>&)
		>* update_func,
		int nmatrices, int max_size, double tolerance,
		int max_iter, bool verbose){
	this->Name = "DIIS";
	this->Verbose = verbose;
	this->MaxSize = max_size;
	this->Tolerance = tolerance;
	this->MaxIter = max_iter;
	this->UpdateFunc = update_func;
	this->Objectivess.resize(nmatrices);
	this->Updatess.resize(nmatrices);
	this->Residualss.resize(nmatrices);
	this->Auxiliariess.resize(nmatrices);
}

int DIIS::getNumMatrices(){
	return (int)this->Objectivess.size();
}

int DIIS::getCurrentSize(){
	return (int)this->Objectivess[0].size();
}

void DIIS::Steal(DIIS& another_diis){
	assert(this->getNumMatrices() == another_diis.getNumMatrices() && "Different numbers of co-optimized matrices!");
	this->Objectivess = another_diis.Objectivess;
	this->Updatess = another_diis.Updatess;
	this->Residualss = another_diis.Residualss;
	this->Auxiliariess = another_diis.Auxiliariess;
	if ( this->MaxSize < another_diis.getCurrentSize() ){
		const int remove = another_diis.getCurrentSize() - this->MaxSize;
		for ( int imat = 0; imat < this->getNumMatrices(); imat++ ){
			this->Objectivess[imat].erase(this->Objectivess[imat].begin(), this->Objectivess[imat].begin() + remove);
			this->Updatess[imat].erase(this->Updatess[imat].begin(), this->Updatess[imat].begin() + remove);
			this->Residualss[imat].erase(this->Residualss[imat].begin(), this->Residualss[imat].begin() + remove);
			this->Auxiliariess[imat].erase(this->Auxiliariess[imat].begin(), this->Auxiliariess[imat].begin() + remove);
		}
	}
}

void DIIS::Append(
		std::vector<double>& objectives,
		std::vector<EigenMatrix>& updates,
		std::vector<EigenMatrix>& residuals,
		std::vector<EigenMatrix>& auxiliaries){
	const int nmatrices = this->getNumMatrices();
	assert(
			(int)objectives.size() == nmatrices &&
			(int)updates.size() == nmatrices &&
			(int)residuals.size() == nmatrices &&
			(int)auxiliaries.size() == nmatrices &&
			"Inconsistent numbers of matrices!"
	);
	if ( this->getCurrentSize() == this->MaxSize ){
		for ( int i = 0; i < nmatrices; i++ ){
			this->Objectivess[i].pop_front();
			this->Updatess[i].pop_front();
			this->Residualss[i].pop_front();
			this->Auxiliariess[i].pop_front();
		}
	}
	for ( int i = 0; i < nmatrices; i++ ){
		this->Objectivess[i].push_back(objectives[i]);
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
		std::printf("Number of co-optimized matrices: %d\n", this->getNumMatrices());
		std::printf("Convergence threshold for residual : %E\n\n", this->Tolerance);
	}

	double E = 0;
	double ResiNorm = 0;
	bool converged = 0;
	const auto all_start = __now__;

	for ( int iiter = 0; iiter < this->MaxIter && ( ! converged ); iiter++ ){

		const auto iter_start = __now__;
		if (this->Verbose) std::printf("Iteration %d\n", iiter);

		std::vector<double> objectives(nmatrices);
		std::vector<EigenMatrix> updates(nmatrices);
		std::vector<EigenMatrix> residuals(nmatrices);
		std::vector<EigenMatrix> auxiliaries(nmatrices);
		if ( iiter == 0 && this->getCurrentSize() >= 2 ){
			std::printf("Hot restart -> Skipping computation in the first iteration\n");
			for ( int imat = 0; imat < nmatrices; imat++ ){
				objectives[imat] = this->Objectivess[imat].back();
				updates[imat] = this->Updatess[imat].back();
				residuals[imat] = this->Residualss[imat].back();
				auxiliaries[imat] = this->Auxiliariess[imat].back();
			}
		}else{
			std::tie(objectives, updates, residuals, auxiliaries) = (*this->UpdateFunc)(Ms);
			this->Append(objectives, updates, residuals, auxiliaries);
		}

		const double oldE = E;
		E = objectives[0];
		for ( double obj : objectives ) E = E > obj ? E : obj;
		const double deltaE = E - oldE;
		ResiNorm = residuals[0].cwiseAbs().maxCoeff();
		for ( EigenMatrix& res : residuals )
			ResiNorm = ResiNorm > res.norm() ? ResiNorm : res.cwiseAbs().maxCoeff();
		if (this->Verbose){
			std::printf("Target = %.10f\n", E);
			std::printf("Difference in target = %E\n", deltaE);
			std::printf("Maximal residual = %E\n", ResiNorm);
		}

		converged = ResiNorm < this->Tolerance;
		if (converged){
			if (this->Verbose) std::printf("Converged!\n");
		}else{
			if (this->Verbose) std::printf("Not converged yet!\n");
			if ( this->getCurrentSize() < 2 ){
				if (this->Verbose) std::printf("Update: Naive\n");
				Ms = updates;
			}else{
				if (this->Verbose){
					std::printf("Update: DIIS\n");
					std::printf("Current space size: %d\n", this->getCurrentSize());
				}
				for ( int mat = 0; mat < nmatrices; mat++ ){
					const EigenVector Ws = this->Extrapolate(mat);
					Ms[mat].setZero();
					for ( int i = 0; i < this->getCurrentSize(); i++ )
						Ms[mat] += Ws[i] * Updatess[mat][i];
					if (this->Verbose){
						std::printf("DIIS weight:");
						for ( int i = 0; i < this->getCurrentSize(); i++ )
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
