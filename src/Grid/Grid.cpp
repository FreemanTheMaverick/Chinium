#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <vector>
#include <array>
#include <chrono>
#include <cstdio>
#include <string>
#include <libmwfn.h>

#include "../Macro.h"
#include "Grid.h"

void Grid::setType(int type){
	this->Type = type;
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->Type = type;
		}
	}
}
	
void Grid::getAO(int derivative, int output){
	const int order = derivative + this->Type;
	if (output) std::printf("Generating grids to order %d of basis functions ... ", order);
	auto start = __now__;
	#pragma omp parallel for schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getAO(derivative);
		}
	}
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
}

void Grid::getDensity(EigenMatrix D){
	#pragma omp parallel for schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getDensity(D);
		}
	}
}

double Grid::getNumElectrons(){
	double n = 0;
	#pragma omp parallel for reduction(+:n) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getNumElectrons(n);
		}
	}
	return n;
}

void Grid::getDensityU(std::vector<EigenMatrix> Ds){
	#pragma omp parallel for schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getDensityU(Ds);
		}
	}
}

void Grid::getDensitySkeleton(EigenMatrix D){
	#pragma omp parallel for schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getDensitySkeleton(D);
		}
	}
}

void Grid::getDensitySkeleton2(EigenMatrix D){
	#pragma omp parallel for schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getDensitySkeleton2(D);
		}
	}
}

double Grid::getEnergy(){
	double e = 0;
	#pragma omp parallel for reduction(+:e) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getEnergy(e);
		}
	}
	return e;
}

template <typename T>
void VectorSum(
		std::vector<T>& omp_out,
		std::vector<T>& omp_in){
	for ( int i = 0; i < (int)omp_out.size(); i++ )
		omp_out[i] += omp_in[i];
}

std::vector<double> Grid::getEnergyGrad(){
	std::vector<double> e(3 * this->MWFN->getNumCenters(), 0);
	#pragma omp declare reduction(VectorSumReduction: std::vector<double>: VectorSum<double>(omp_out, omp_in)) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(VectorSumReduction: e) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getEnergyGrad(e);
		}
	}
	return e;
}

template <typename T>
void VectorVectorSum(
		std::vector<std::vector<T>>& omp_out,
		std::vector<std::vector<T>>& omp_in){
	for ( int i = 0; i < (int)omp_out.size(); i++ )
		for ( int j = 0; j < (int)omp_out[i].size(); j++ )
			omp_out[i][j] += omp_in[i][j];
}

std::vector<std::vector<double>> Grid::getEnergyHess(){
	std::vector<std::vector<double>> e(3 * this->MWFN->getNumCenters(), std::vector<double>(3 * this->MWFN->getNumCenters(), 0));
	#pragma omp declare reduction(VectorVectorSumReduction: std::vector<std::vector<double>>: VectorVectorSum<double>(omp_out, omp_in)) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(VectorVectorSumReduction: e) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getEnergyHess(e);
		}
	}
	return e;
}

EigenMatrix Grid::getFock(){
	EigenMatrix F = EigenZero(this->MWFN->getNumBasis(), this->MWFN->getNumBasis());
	#pragma omp declare reduction(MatrixSumReduction: EigenMatrix: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(MatrixSumReduction: F) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getFock(F);
		}
	}
	return F;
}

std::vector<EigenMatrix> Grid::getFockSkeleton(){
	std::vector<EigenMatrix> Fs(this->MWFN->getNumCenters() * 3, EigenZero(this->MWFN->getNumBasis(), this->MWFN->getNumBasis()));
	#pragma omp declare reduction(VectorSumReduction: std::vector<EigenMatrix>: VectorSum<EigenMatrix>(omp_out, omp_in)) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(VectorSumReduction: Fs) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getFockSkeleton(Fs);
		}
	}
	return Fs;
}

// enum D_t{ s_t, u_t };
template <D_t d_t>
std::vector<EigenMatrix> Grid::getFockU(){
	std::vector<EigenMatrix> Fs( [this]() -> int {
			if constexpr ( d_t == s_t ) return this->MWFN->getNumCenters() * 3;
			else return this->SubGridBatches[0][0]->RhoU.dimension(1); // Crap. I have to go inside a SubGrid.
	}(), EigenZero(this->MWFN->getNumBasis(), this->MWFN->getNumBasis()) );
	#pragma omp declare reduction(VectorSumReduction: std::vector<EigenMatrix>: VectorSum<EigenMatrix>(omp_out, omp_in)) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(VectorSumReduction: Fs) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getFockU<d_t>(Fs);
		}
	}
	return Fs;
}
template std::vector<EigenMatrix> Grid::getFockU<s_t>(); // Using skeleton derivatives of density
template std::vector<EigenMatrix> Grid::getFockU<u_t>(); // Using U derivatives of density
