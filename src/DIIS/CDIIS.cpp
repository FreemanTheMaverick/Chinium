#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <string>
#include <cstdio>
#include <chrono>

#include "../Macro.h"
#include "CDIIS.h"


CDIIS::CDIIS(
		std::function<std::tuple<
			std::vector<double>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
			> (std::vector<EigenMatrix>&)
		>* update_func,
		int nmatrices, int max_size, double tolerance,
		int max_iter, bool verbose): DIIS(update_func, nmatrices, max_size, tolerance, max_iter, verbose){
	this->Name = "CDIIS";
}

EigenVector CDIIS::Extrapolate(int index){
	EigenVector weight = EigenZero(this->getCurrentSize(), 1);
	bool checked = 0;
	double ridge = 1;
	if (this->Verbose){
		std::printf("CDIIS extrapolation on Matrix %d\n", index);
		std::printf("| Size | Ridge factor | Determinant of LHS | Largest absolute weight |\n");
	}
	const int size = this->getCurrentSize();
	for ( int this_size = size; this_size > 1 && !checked; this_size--, ridge += 0.001 ){
		EigenMatrix H = EigenZero(this_size + 1, this_size + 1);
		EigenVector b = EigenZero(this_size + 1, 1); b(this_size) = 1;
		for ( int i = 0; i < this_size; i++ ){
			H(i, this_size) = H(this_size, i) = 1;
			for ( int j = i; j < this_size; j++ )
				H(i, j) = H(j, i) = Dot(this->Residualss[index][size - this_size + i], this->Residualss[index][size - this_size + j]);
		}
		H.diagonal() *= ridge;
		weight.setZero();
		weight.tail(this_size) = H.colPivHouseholderQr().solve(b).head(this_size);
		const double det = H.determinant();
		const double largest_abs_coefficient = weight.cwiseAbs().maxCoeff();
		checked = largest_abs_coefficient < 5 || det > 1e-12;
		if (this->Verbose) std::printf("| %d | %f | %E | %f |\n", this_size, ridge, det, largest_abs_coefficient);
	}
	return weight;
}
