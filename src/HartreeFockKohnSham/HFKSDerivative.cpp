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
#include <omp.h>

#include "../Macro.h"
#include "../Multiwfn/Multiwfn.h"
#include "../Integral/Int2C1E.h"
#include "../Integral/Int4C2E.h"
#include "../Grid/Grid.h"
#include "../ExchangeCorrelation.h"

#include "CoupledPerturbed.h"

#include <iostream>


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

std::tuple<EigenMatrix, EigenMatrix> HFKSDerivative(Multiwfn& mwfn, Int2C1E& int2c1e, Int4C2E& int4c2e, ExchangeCorrelation& xc, Grid& grid, int derivative, int output, int nthreads){

	Eigen::setNbThreads(nthreads);

	const int natoms = mwfn.getNumCenters();
	const int nbasis = mwfn.getNumBasis();
	const long int ngrids = grid.NumGrids;
	std::vector<double> Gradient(3 * natoms, 0);
	std::vector<std::vector<double>> Hessian(3 * natoms, std::vector<double>(3 * natoms, 0));
	std::vector<int> bf2atom(nbasis);

	for ( int iatom = 0, kbasis = 0; iatom < natoms; iatom++ )
		for ( int jbasis = 0; jbasis < mwfn.Centers[iatom].getNumBasis(); jbasis++, kbasis++ )
			bf2atom[kbasis] = iatom;

	const EigenMatrix D = mwfn.getDensity() / 2;
	const EigenMatrix W = mwfn.getEnergyDensity() / 2;
	std::vector<EigenMatrix> Fskeletons(3 * natoms, EigenMatrix(nbasis, nbasis));

	// HF gradient
	auto [SWgrads, KDgrads, VDgrads] = int2c1e.ContractGrads(D, W); // std::vector<double>
	const std::vector<double> DGDgrads = int4c2e.ContractGrads(D, D);
	__VectorIncrement__(Gradient, KDgrads, 2)
	__VectorIncrement__(Gradient, VDgrads, 2)
	__VectorIncrement__(Gradient, SWgrads, -2)
	__VectorIncrement__(Gradient, DGDgrads, 1)

	// HF skeleton hessian and skeleton Fock
	if ( derivative > 1 ){
		auto [SWhesss, KDhesss, VDhesss] = int2c1e.ContractHesss(D, W); // std::vector<std::vector<double>>
		const std::vector<std::vector<double>> DGDhesss = int4c2e.ContractHesss(D, D);
		__VectorVectorIncrement__(Hessian, SWhesss, -2)
		__VectorVectorIncrement__(Hessian, KDhesss, 2)
		__VectorVectorIncrement__(Hessian, VDhesss, 2)
		__VectorVectorIncrement__(Hessian, DGDhesss, 1)
		const std::vector<EigenMatrix> DGgrads = int4c2e.ContractGrads(D);
		__VectorIncrement__(Fskeletons, int2c1e.KineticGrads, 1)
		__VectorIncrement__(Fskeletons, int2c1e.NuclearGrads, 1)
		__VectorIncrement__(Fskeletons, DGgrads, 1)
	}

	// XC
	if (xc){
		// Creating batches for grids
		const long int ngrids_per_batch = __max_num_grids__ > ngrids ? __max_num_grids__ : ngrids; // If only gradient is required, the grids will be done in batches. Otherwise, they will be done simultaneously, because the intermediate variable, nuclear gradient of density on grids, is required in CPSCF.
		std::vector<long int> batch_tails = {};
		long int tmp = 0;
		while ( tmp < ngrids ){
			if ( tmp + ngrids_per_batch < ngrids )
				batch_tails.push_back(tmp + ngrids_per_batch);
			else batch_tails.push_back(ngrids);
			tmp += ngrids_per_batch;
		}
		const int nbatches = batch_tails.size();
		if (output) std::printf("Calculating XC nuclear skeleton derivatives in %d batches ...\n", nbatches);

		const auto start_all = __now__;
		for ( int ibatch = 0; ibatch < nbatches; ibatch++ ){
			const auto start_batch = __now__;
			if (output) std::printf("| Batch %d\n", ibatch);
			const int batch_head = ibatch == 0 ? 0 : batch_tails[ibatch - 1];
			const int batch_tail = batch_tails[ibatch];
			if (output) std::printf("| | ");
			Grid batch_grid(grid, batch_head, batch_tail - batch_head, output);

			// XC gradient
			if (output) std::printf("| | ");
			batch_grid.getGridAO(derivative, output);

			auto start = __now__;
			if (output) std::printf("| | Calculating skeleton hessian of density on these grids ...");
			batch_grid.getGridDensitySkeleton(2 * D);
			if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));

			start = __now__;
			if (output) std::printf("| | Calculating XC gradient contributed by these grids ...");
			const std::vector<double> gxc = batch_grid.getEnergyGrad();
			__VectorIncrement__(Gradient, gxc, 1)
			if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));

			// XC skeleton hessian and skeleton Fock
			if ( derivative > 1 ){
				xc.Evaluate("f", batch_grid);

				start = __now__;
				if (output) std::printf("| | Calculating skeleton hessian of density on these grids ...");
				batch_grid.getGridDensitySkeleton2(2 * D);
				if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));

				start = __now__;
				if (output) std::printf("| | Calculating XC skeleton hessian contributed by these grids ...");
				const std::vector<std::vector<double>> hxc = batch_grid.getEnergyHess();
				__VectorVectorIncrement__(Hessian, hxc, 1)
				if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));

				start = __now__;
				if (output) std::printf("| | Calculating skeleton gradient of XC Fock matrices contributed by these grids ...");
				std::vector<EigenMatrix> fxcskeletons = batch_grid.getFockSkeleton();
				__VectorIncrement__(Fskeletons, fxcskeletons, 1)
				fxcskeletons = batch_grid.getFockDensity();
				__VectorIncrement__(Fskeletons, fxcskeletons, 1)
				if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));
			}
			if (output) std::printf("| | Done in %f s\n", __duration__(start_batch, __now__));
		}
		if (output) std::printf("| Done in %f s\n", __duration__(start_all, __now__));
	}

	if ( derivative > 1 ){
		if (output) std::printf("Non-Idempotent coupled-perturbed self-consistent-field ...\n");
		auto start = __now__;
		auto [Us, dDs, dEs, dWs, dFs] = NonIdempotent( // std::vector<EigenMatrix>
				mwfn.getCoefficientMatrix(),
				mwfn.getEnergy(),
				mwfn.getOccupation() / 2.,
				int2c1e.OverlapGrads, Fskeletons,
				int4c2e, grid,
				output - 1, nthreads
		);
		Fskeletons = dFs;
		if (output) std::printf("Coupled-perturbed self-consistent-field done in %f s\n", __duration__(start, __now__));

		for ( int i = 0; i < 3 * natoms; i++ ) for ( int j = 0; j <= i; j++ ){
			Hessian[i][j] += 2 * dDs[i].cwiseProduct(Fskeletons[j]).sum();
			if ( i != j ) Hessian[j][i] += 2 * dDs[i].cwiseProduct(Fskeletons[j]).sum();
			Hessian[i][j] += 2 * dWs[i].cwiseProduct(int2c1e.OverlapGrads[j]).sum();
			if ( i != j ) Hessian[j][i] += 2 * dWs[i].cwiseProduct(int2c1e.OverlapGrads[j]).sum();
		}

		if ( mwfn.Temperature > 0 ){
			std::vector<int> frac_indeces;
			for ( int i = 0; i < mwfn.getNumIndBasis(); i++ ){
				if ( mwfn.Orbitals[i].Occ > 2 * __Occupation_Cutoff__ && mwfn.Orbitals[i].Occ < 2. - 2 * __Occupation_Cutoff__)
					frac_indeces.push_back(i);
			}
			std::map<int, EigenMatrix> Dns;
			if ( !frac_indeces.empty() ){
				if (output) std::printf("Occupation-fluctuation coupled-perturbed self-consistent-field ...\n");
				start = __now__;
				Dns = OccupationFluctuation(
						mwfn.getCoefficientMatrix(),
						mwfn.getEnergy(),
						mwfn.getOccupation() / 2.,
						frac_indeces,
						int4c2e, grid,
						output - 1, nthreads
				);
				if (output) std::printf("Occupation-fluctuation coupled-perturbed self-consistent-field done in %f s\n", __duration__(start, __now__));
			}

			if (output) std::printf("Occupation-gradient coupled-perturbed self-consistent-field ...\n");
			start = __now__;
			const EigenArray ns = mwfn.getOccupation().array() / 2;
			const EigenVector Nes = (ns * ( ns - 1. )) / mwfn.Temperature;
			const std::vector<EigenVector> dNs = OccupationGradient(
					mwfn.getCoefficientMatrix(),
					mwfn.getEnergy(),
					Dns, Nes,
					int2c1e.OverlapGrads, dFs,
					int4c2e, grid,
					output - 1, nthreads
			);
			if (output) std::printf("Occupation-gradient coupled-perturbed self-consistent-field done in %f s\n", __duration__(start, __now__));
			EigenMatrix hessog = EigenZero(3 * natoms, 3 * natoms);
			for ( int ipert = 0; ipert < 3 * natoms; ipert++ )
				for ( int jpert = 0; jpert < 3 * natoms; jpert++ )
					hessog(ipert, jpert) = dEs[jpert].dot(dNs[ipert]);
			const EigenMatrix hessog_ = hessog + hessog.transpose();
			__HessAdd__(hessog_)
		}
	}

	EigenMatrix gradient = EigenZero(natoms, 3);
	for ( int iatom = 0, tot = 0; iatom < natoms; iatom++ ) for ( int xyz = 0; xyz < 3; xyz++, tot++ ){
		gradient(iatom, xyz) = Gradient[tot];
	}
	EigenMatrix hessian = EigenZero(3 * natoms, 3 * natoms);
	for ( int i = 0; i < 3 * natoms; i++ ) for ( int j = 0; j < 3 * natoms; j++ ){
		hessian(i, j) = Hessian[i][j];
	}
	return std::make_tuple(gradient, hessian);
}


