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

#include "../CoupledPerturbed.h"
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

void R_SCF::Calculate2(){

	Eigen::setNbThreads(nthreads);

	const int natoms = mwfn.getNumCenters();
	const int nbasis = mwfn.getNumBasis();
	std::vector<std::vector<double>> Hessian(3 * natoms, std::vector<double>(3 * natoms, 0));
	std::vector<int> bf2atom(nbasis);

	for ( int iatom = 0, kbasis = 0; iatom < natoms; iatom++ )
		for ( int jbasis = 0; jbasis < mwfn.Centers[iatom].getNumBasis(); jbasis++, kbasis++ )
			bf2atom[kbasis] = iatom;

	const EigenMatrix D = mwfn.getDensity(1);
	const EigenMatrix W = mwfn.getEnergyDensity(1);
	std::vector<EigenMatrix> Fskeletons(3 * natoms, EigenMatrix(nbasis, nbasis));

	auto [SWhesss, KDhesss, VDhesss] = int2c1e.ContractHesss(D, W, 1); // std::vector<std::vector<double>>
	const std::vector<std::vector<double>> DGDhesss = int4c2e.ContractHesss(D, D, 1);
	__VectorVectorIncrement__(Hessian, SWhesss, -2)
	__VectorVectorIncrement__(Hessian, KDhesss, 2)
	__VectorVectorIncrement__(Hessian, VDhesss, 2)
	__VectorVectorIncrement__(Hessian, DGDhesss, 1)
	const std::vector<EigenMatrix> DGgrads = int4c2e.ContractGrads(D, 1);
	__VectorIncrement__(Fskeletons, int2c1e.KineticGrads, 1)
	__VectorIncrement__(Fskeletons, int2c1e.NuclearGrads, 1)
	__VectorIncrement__(Fskeletons, DGgrads, 1)

	if (xc){
		grid.getAO(2, 1);
		xc.Evaluate("f", grid);

		auto start = __now__;
		std::printf("Calculating skeleton hessian of density ...");
		grid.getDensitySkeleton2({2 * D});
		std::printf(" Done in %f s\n", __duration__(start, __now__));

		start = __now__;
		std::printf("Calculating XC skeleton hessian ...");
		const std::vector<std::vector<double>> hxc = grid.getEnergyHess();
		__VectorVectorIncrement__(Hessian, hxc, 1)
		std::printf(" Done in %f s\n", __duration__(start, __now__));

		start = __now__;
		std::printf("Calculating skeleton gradient of XC Fock matrices ...");
		std::vector<EigenMatrix> fxcskeletons = grid.getFockSkeleton()[0];
		__VectorIncrement__(Fskeletons, fxcskeletons, 1)
		fxcskeletons = grid.getFockU<s_t>()[0];
		__VectorIncrement__(Fskeletons, fxcskeletons, 1)
		std::printf(" Done in %f s\n", __duration__(start, __now__));
	}

	std::printf("Non-Idempotent coupled-perturbed self-consistent-field ...\n");
	auto start = __now__;
	auto [Us, dDs, dEs, dWs, dFs] = NonIdempotent( // std::vector<EigenMatrix>
			mwfn.getCoefficientMatrix(1),
			mwfn.getEnergy(1),
			mwfn.getOccupation() / 2.,
			int2c1e.OverlapGrads, Fskeletons,
			int4c2e, grid,
			1, nthreads
	);
	this->dEs = dEs;
	this->dFs = dFs;
	std::printf("Coupled-perturbed self-consistent-field done in %f s\n", __duration__(start, __now__));

	for ( int i = 0; i < 3 * natoms; i++ ) for ( int j = 0; j <= i; j++ ){
		Hessian[i][j] += 2 * dDs[i].cwiseProduct(Fskeletons[j]).sum();
		if ( i != j ) Hessian[j][i] += 2 * dDs[i].cwiseProduct(Fskeletons[j]).sum();
		Hessian[i][j] -= 2 * dWs[i].cwiseProduct(int2c1e.OverlapGrads[j]).sum();
		if ( i != j ) Hessian[j][i] -= 2 * dWs[i].cwiseProduct(int2c1e.OverlapGrads[j]).sum();
	}

	for ( int i = 0; i < 3 * natoms; i++ ) for ( int j = 0; j < 3 * natoms; j++ ){
		this->Hessian(i, j) += Hessian[i][j];
	}
}
