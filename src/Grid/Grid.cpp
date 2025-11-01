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
#include "Tensor.h"
#include "Grid.h"

void Grid::getAO(int derivative, int output){
	#pragma omp parallel for schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getAO(derivative, output);
		}
	}
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
			n += subgrid->getNumElectrons();
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
			e += subgrid->getEnergy();
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
	std::vector<double> e(3 * this->Mwfn->getNumCenters(), 0);
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
	std::vector<std::vector<double>> e(3 * this->Mwfn->getNumCenters(), std::vector<double>(3 * this->Mwfn->getNumCenters(), 0));
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
	EigenMatrix F = EigenZero(this->Mwfn->getNumBasis(), this->Mwfn->getNumBasis());
	#pragma omp declare reduction(VectorSumReduction: EigenMatrix: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(MatrixSumReduction: F) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getFock(F);
		}
	}
	return F;
}

std::vector<EigenMatrix> Grid::getFockSkeleton(){
	std::vector<EigenMatrix> Fs(this->Mwfn->getNumCenters() * 3, EigenZero(this->Mwfn->getNumBasis(), this->Mwfn->getNumBasis());
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
	std::vector<EigenMatrix> Fs( []() -> int {
			if constexpr (std::is_same_v<d_t, s_t>) return this->Mwfn->getNumCenters() * 3;
			else return this->SubGridBatches[0][0]->RhoU.dimension(1); // Crap. I have to go inside a SubGrid.
	}, EigenZero(this->Mwfn->getNumBasis(), this->Mwfn->getNumBasis()) );
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
