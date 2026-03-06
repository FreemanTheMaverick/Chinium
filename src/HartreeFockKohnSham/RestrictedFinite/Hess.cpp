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
#include "../RestrictedFinite.h"

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

void RGC_SCF::Calculate2(){
	R_SCF::Calculate2();

	Eigen::setNbThreads(nthreads);

	const int natoms = mwfn.getNumCenters();
	std::vector<std::vector<double>> Hessian(3 * natoms, std::vector<double>(3 * natoms, 0));

	std::vector<int> frac_indeces;
	for ( int i = 0; i < mwfn.getNumIndBasis(); i++ ){
		if ( mwfn.Orbitals[i].Occ > 2 * __Occupation_Cutoff__ && mwfn.Orbitals[i].Occ < 2. - 2 * __Occupation_Cutoff__)
			frac_indeces.push_back(i);
	}
	std::map<int, EigenMatrix> Dns;
	if ( !frac_indeces.empty() ){
		std::printf("Occupation-fluctuation coupled-perturbed self-consistent-field ...\n");
		auto start = __now__;
		Dns = OccupationFluctuation(
				mwfn.getCoefficientMatrix(1),
				mwfn.getEnergy(1),
				mwfn.getOccupation() / 2.,
				frac_indeces,
				int4c2e, grid,
				1, nthreads
		);
		std::printf("Occupation-fluctuation coupled-perturbed self-consistent-field done in %f s\n", __duration__(start, __now__));
	}

	std::printf("Occupation-gradient coupled-perturbed self-consistent-field ...\n");
	auto start = __now__;
	const EigenArray ns = mwfn.getOccupation().array() / 2;
	const EigenVector Nes = (ns * ( ns - 1. )) / Temperature;
	const std::vector<EigenVector> dNs = OccupationGradient(
			mwfn.getCoefficientMatrix(1),
			mwfn.getEnergy(1),
			Dns, Nes,
			int2c1e.OverlapGrads, dFs,
			int4c2e, grid,
			1, nthreads
	);
	std::printf("Occupation-gradient coupled-perturbed self-consistent-field done in %f s\n", __duration__(start, __now__));
	EigenMatrix hessog = EigenZero(3 * natoms, 3 * natoms);
	for ( int ipert = 0; ipert < 3 * natoms; ipert++ )
		for ( int jpert = 0; jpert < 3 * natoms; jpert++ )
			hessog(ipert, jpert) = dEs[jpert].dot(dNs[ipert]);
	const EigenMatrix hessog_ = hessog + hessog.transpose();
	__HessAdd__(hessog_)

	for ( int i = 0; i < 3 * natoms; i++ ) for ( int j = 0; j < 3 * natoms; j++ ){
		this->Hessian(i, j) += Hessian[i][j];
	}
}
