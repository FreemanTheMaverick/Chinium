#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <tuple>
#include <string>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cassert>
#include <libmwfn.h>

#include "../../Macro.h"
#include "../../Integral.h"
#include "../../Grid.h"
#include "../../ExchangeCorrelation.h"

#include "../Restricted.h"

#define __Check_Vector_Array__(vec)\
	vec.size() && vec[0].size() && vec[0][0]

#define __VectorIncrement__(A, B, b)\
	assert(A.size() == B.size() && "The two std::vector<>-s have different sizes!");\
	for ( int i = 0; i < (int)B.size(); i++ ){\
		A[i] += (b) * B[i];\
	}

#define __VectorVectorIncrement__(A, B, b)\
	assert(A.size() == B.size() && "The two std::vector<std::vector<>>-s have different sizes!");\
	for ( int j = 0; j < (int)B.size(); j++ ){\
		__VectorIncrement__(A[j], B[j], b)\
	}

#define __GradAdd__(A)\
	for ( int iatom = 0, tot = 0; iatom < natoms; iatom++ ) for ( int xyz = 0; xyz < 3; xyz++, tot++){\
		Gradient[tot] += A(iatom, xyz);\
	}

#define __HessAdd__(A)\
	for ( int i = 0; i < 3 * natoms; i++ ) for ( int j = 0; j < 3 * natoms; j++){\
		Hessian[i][j] += ( A(i, j) + A(j, i) ) / 2;\
	}

#define __max_num_grids__ 100000
#define __Occupation_Cutoff__ 1.e-8

void R_SCF::Calculate1(){

	Eigen::setNbThreads(nthreads);

	const int natoms = mwfn.getNumCenters();
	const int nbasis = mwfn.getNumBasis();
	std::vector<double> Gradient(3 * natoms, 0);
	std::vector<int> bf2atom(nbasis);
	int2c1e.CalculateIntegrals(1, 1);

	for ( int iatom = 0, kbasis = 0; iatom < natoms; iatom++ )
		for ( int jbasis = 0; jbasis < mwfn.Centers[iatom].getNumBasis(); jbasis++, kbasis++ )
			bf2atom[kbasis] = iatom;

	const EigenMatrix D = mwfn.getDensity(1);
	const EigenMatrix W = mwfn.getEnergyDensity(1);

	auto [SWgrads, KDgrads, VDgrads] = int2c1e.ContractGrads(D, W, 1); // std::vector<double>
	const std::vector<double> DGDgrads = int4c2e.ContractGrads(D, D, 1);
	__VectorIncrement__(Gradient, KDgrads, 2)
	__VectorIncrement__(Gradient, VDgrads, 2)
	__VectorIncrement__(Gradient, SWgrads, -2)
	__VectorIncrement__(Gradient, DGDgrads, 1)

	if (xc){
		grid.getAO(1, 1);
		auto start = __now__;
		std::printf("Calculating skeleton gradient of density ...");
		grid.getDensitySkeleton({2 * D});
		std::printf(" Done in %f s\n", __duration__(start, __now__));

		start = __now__;
		std::printf("Calculating XC gradient ...");
		const std::vector<double> gxc = grid.getEnergyGrad();
		__VectorIncrement__(Gradient, gxc, 1)
		std::printf(" Done in %f s\n", __duration__(start, __now__));
	}

	for ( int iatom = 0, tot = 0; iatom < natoms; iatom++ ) for ( int xyz = 0; xyz < 3; xyz++, tot++ ){
		this->Gradient(iatom, xyz) += Gradient[tot];
	}
}


