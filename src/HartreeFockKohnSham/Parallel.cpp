#include <Eigen/Core>
#include <vector>

#include "../Macro.h"

std::vector<std::vector<EigenArray>> Matrices2Arrays(std::vector<EigenMatrix>& matrices, int nmatrices_redun){
	const int nmatrices = matrices.size();
	const int nrows = matrices[0].rows();
	const int ncols = matrices[0].cols();
	std::vector<std::vector<EigenArray>> arrays(nrows);
	for ( int i = 0; i < nrows; i++ ){
		arrays[i].resize(ncols);
		for ( int j = 0; j < ncols; j++ ){
			arrays[i][j] = EigenZero(nmatrices_redun, 1).array();
			for ( int k = 0; k < nmatrices; k++ )
				arrays[i][j](k) = matrices[k](i, j);
		}
	}
	return arrays;
}

std::vector<std::vector<EigenArray>> Matrices2Arrays(std::vector<EigenMatrix>& matrices){
	const int nmatrices = matrices.size();
	return Matrices2Arrays(matrices, nmatrices);
}

std::vector<EigenMatrix> Arrays2Matrices(std::vector<std::vector<EigenArray>>& arrays, int nmatrices_noredun){
	const int nrows = arrays.size();
	const int ncols = arrays[0].size();
	std::vector<EigenMatrix> matrices(nmatrices_noredun, EigenZero(nrows, ncols));
	for ( int i = 0; i < nrows; i++ )
		for ( int j = 0; j < ncols; j++ )
			for ( int k = 0; k < nmatrices_noredun; k++ )
				matrices[k](i, j) = arrays[i][j](k);
	return matrices;
}

std::vector<EigenMatrix> Arrays2Matrices(std::vector<std::vector<EigenArray>>& arrays){
	const int nmatrices_noredun = arrays[0][0].size();
	return Arrays2Matrices(arrays, nmatrices_noredun);
}

void MultipleMatrixReduction(
		std::vector<std::vector<EigenArray>>& omp_out,
		std::vector<std::vector<EigenArray>>& omp_in){
	for ( int i = 0; i < (int)omp_out.size(); i++ )
		for ( int j = 0; j < (int)omp_out[i].size(); j++ )
			omp_out[i][j] += omp_in[i][j];
}

std::vector<std::vector<EigenArray>> MultipleMatrixInitialization(int nmatrices, int nrows, int ncols){
	std::vector<std::vector<EigenArray>> arrays(nrows);
	for ( int i = 0; i < nrows; i++ ){
		arrays[i].resize(ncols);
		for ( int j = 0; j < ncols; j++ )
			arrays[i][j] = EigenZero(nmatrices, 1).array();
	}
	return arrays;
}
