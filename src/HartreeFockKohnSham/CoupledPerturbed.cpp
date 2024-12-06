#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <tuple>
#include <string>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cassert>
#include <omp.h>

#include "../Macro.h"
#include "../Multiwfn.h"
#include "../Optimization/DIIS.h"
#include "../Grid/GridAO.h"
#include "../Grid/GridDensity.h"
#include "../ExchangeCorrelation/MwfnXC1.h"
#include "../Grid/GridPotential.h"
#include "FockFormation.h"

#include <iostream>

#define __Allocate_and_Copy__(dest, src)\
	if (!dest) dest = new double[ngrids];\
	std::memcpy(dest, src, ngrids * sizeof(double));
	

std::tuple<
	std::vector<EigenMatrix>,
	std::vector<EigenMatrix>,
	std::vector<EigenMatrix>,
	std::vector<EigenMatrix>
> NonIdempotent(
		EigenMatrix C, EigenVector es, EigenVector ns,
		std::vector<EigenMatrix>& Ss,
		std::vector<EigenMatrix>& Fskeletons,
		short int* is, short int* js, short int* ks, short int* ls,
		char* degs, double* ints, long int length,
		double kscale,
		std::vector<int> orders,
		double* ws,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2xxs, double* ao2yys, double* ao2zzs,
		double* ao2xys, double* ao2xzs, double* ao2yzs,
		double* d1xs, double* d1ys, double* d1zs,
		double* vrs, double* vss,
		double* vrrs, double* vrss, double* vsss,
		std::vector<std::vector<double*>>& gds_skeleton,
		std::vector<std::vector<double*>>& gd1xs_skeleton,
		std::vector<std::vector<double*>>& gd1ys_skeleton,
		std::vector<std::vector<double*>>& gd1zs_skeleton,
		long int ngrids,
		int output, int nthreads){

	// Step 1: Preparing data used by all perturbations
	
	// HF part
	const int nmatrices = Ss.size();
	const int nbasis = C.rows();
	const int nindbasis = C.cols();
	std::vector<EigenMatrix> CTSCs(nmatrices, EigenZero(nindbasis, nindbasis));
	std::vector<EigenMatrix> Bskeletons(nmatrices, EigenZero(nindbasis, nindbasis));
	std::vector<EigenMatrix> Bs(nmatrices, EigenZero(nindbasis, nindbasis));
	std::vector<EigenMatrix> Us(nmatrices, EigenZero(nindbasis, nindbasis));
	std::vector<EigenMatrix> Ds(nmatrices, EigenZero(nbasis, nbasis));
	std::vector<EigenMatrix> Ns(nmatrices, EigenZero(nbasis, nbasis));
	for ( int imatrix = 0; imatrix < nmatrices; imatrix++ ){
		const EigenMatrix CTSC = CTSCs[imatrix] = C.transpose() * Ss[imatrix] * C;
		Bs[imatrix] = Bskeletons[imatrix] = CTSC * es.asDiagonal() - C.transpose() * Fskeletons[imatrix] * C;
		Us[imatrix] = - 0.5 * CTSC;
		for ( int p = 0; p < nindbasis; p++ ){
			EigenMatrix Np = CTSC(p, p) * C.col(p) * C.col(p).transpose();
			for ( int q = 0; q < p; q++ ){
				const EigenMatrix CpCqT = C.col(p) * C.col(q).transpose();
				Np += CTSC(p, q) * ( CpCqT + CpCqT.transpose() );
			}
			Ns[imatrix] += ns[p] * Np;
		}
	}

	// KS part
	double* gds = nullptr;
	double* gd1xs = nullptr;
	double* gd1ys = nullptr;
	double* gd1zs = nullptr;

	std::vector<std::deque<EigenMatrix>> Bs_deque(nmatrices);
	std::vector<std::deque<EigenMatrix>> Rs_deque(nmatrices);
	std::vector<int> dones(nmatrices, 0);
	for ( int iiter = 0; iiter < 100; iiter++ ){
		int nundones = nmatrices;
		for ( int jdone : dones ) nundones -= jdone;
		if ( nundones == 0 ) break;
		if (output) std::printf("| Iteration %d with %d perturbations left ...", iiter, nundones); 
		auto start = __now__;

		std::vector<EigenMatrix> undoneDs(nundones, EigenZero(nbasis, nbasis));
		for ( int imatrix = 0, jundone = 0; imatrix < nmatrices; imatrix++ ) if ( !dones[imatrix] ){
			if ( Bs_deque[imatrix].size() > 1 )
				Bs[imatrix] = CDIIS(Rs_deque[imatrix], Bs_deque[imatrix]);
			Ds[imatrix] = - Ns[imatrix];
			for ( int p = 0; p < nindbasis; p++ ) for ( int q = 0; q < p; q++ ) if ( std::abs(es[p] - es[q]) > 1.e-6 ){
				const double Upq = Us[imatrix](p, q) = Bs[imatrix](p, q) / ( es(p) - es(q) );
				Us[imatrix](q, p) = - Upq - CTSCs[imatrix](p, q);
				const EigenMatrix CpCqT = C.col(p) * C.col(q).transpose();
				Ds[imatrix] += ( ns[q] - ns[p] ) * Upq * ( CpCqT + CpCqT.transpose() );
			}
			undoneDs[jundone++] = Ds[imatrix];
		}
		std::vector<EigenMatrix> undoneFUs = GhfMultiple(
				is, js, ks, ls,
				degs, ints, length,
				undoneDs, kscale, nthreads
		);
		for ( int imatrix = 0, jundone = 0; imatrix < nmatrices; imatrix++ ) if ( !dones[imatrix] ){
			if (std::find(orders.begin(), orders.end(), 0) != orders.end()){
				assert(aos && "AOs on grids do not exist!");
				assert(ao1xs && "First-order x-derivatives of AOs on grids do not exist!");
				assert(ao1ys && "First-order y-derivatives of AOs on grids do not exist!");
				assert(ao1zs && "First-order z-derivatives of AOs on grids do not exist!");
				__Allocate_and_Copy__(gds, gds_skeleton[imatrix/3][imatrix%3]);
			}
			if (std::find(orders.begin(), orders.end(), 1) != orders.end()){
				assert(aos && "AOs on grids do not exist!");
				assert(ao1xs && "First-order x-derivatives of AOs on grids do not exist!");
				assert(ao1ys && "First-order y-derivatives of AOs on grids do not exist!");
				assert(ao1zs && "First-order z-derivatives of AOs on grids do not exist!");
				assert(ao2xxs && "Second-order xx-derivatives of AOs on grids do not exist!");
				assert(ao2yys && "Second-order yy-derivatives of AOs on grids do not exist!");
				assert(ao2zzs && "Second-order zz-derivatives of AOs on grids do not exist!");
				assert(ao2xys && "Second-order xy-derivatives of AOs on grids do not exist!");
				assert(ao2xzs && "Second-order xz-derivatives of AOs on grids do not exist!");
				assert(ao2yzs && "Second-order yz-derivatives of AOs on grids do not exist!");
				__Allocate_and_Copy__(gd1xs, gd1xs_skeleton[imatrix/3][imatrix%3]);
				__Allocate_and_Copy__(gd1ys, gd1ys_skeleton[imatrix/3][imatrix%3]);
				__Allocate_and_Copy__(gd1zs, gd1zs_skeleton[imatrix/3][imatrix%3]);
			}
			GetDensity(
					orders,
					aos,
					ao1xs, ao1ys, ao1zs,
					nullptr,
					ngrids, 2*Ds[imatrix],
					gds, gd1xs, gd1ys, gd1zs,
					nullptr, nullptr
			);
			undoneFUs[jundone] += PotentialSkeleton(
					orders,
					ws, ngrids, nbasis,
					aos,
					ao1xs, ao1ys, ao1zs,
					ao2xxs, ao2yys, ao2zzs,
					ao2xys, ao2xzs, ao2yzs,
					d1xs, d1ys, d1zs,
					vrs, vss,
					vrrs, vrss, vsss,
					gds, gd1xs, gd1ys, gd1zs
			);
			const EigenMatrix Bnew = Bskeletons[imatrix] - C.transpose() * undoneFUs[jundone++] * C;
			const EigenMatrix Bdiff = Bnew - Bs[imatrix];
			Bs_deque[imatrix].push_back(Bnew);
			Rs_deque[imatrix].push_back(Bdiff);
			if ( Bdiff.cwiseAbs().maxCoeff() < 1.e-5 )
				dones[imatrix] = 1;
			Bs[imatrix] = Bnew;
		}
		if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));
	}

	std::vector<EigenMatrix> Es(nmatrices, EigenZero(nindbasis, 1));
	std::vector<EigenMatrix> Ws(nmatrices, EigenZero(nbasis, nbasis));
	for ( int imatrix = 0; imatrix < nmatrices; imatrix++ ){
		Es[imatrix] = - Bs[imatrix].diagonal();
		for ( int q = 0; q < nindbasis; q++ ){
			const EigenMatrix Wq1 = - C.col(q) * C.col(q).transpose() * Bs[imatrix](q, q);
			EigenMatrix Wq2 = EigenZero(nbasis, nbasis);
			for ( int p = 0; p < nindbasis; p++ ){
				const EigenMatrix CpCqT = C.col(p) * C.col(q).transpose();
				Wq2 += Us[imatrix](p, q) * ( CpCqT + CpCqT.transpose() );
			}
			Ws[imatrix] += ns(q) * ( Wq1 + es(q) * Wq2 );
		}
	}
	return std::make_tuple(Us, Ds, Es, Ws);
}



