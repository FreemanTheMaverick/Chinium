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

#define __Allocate_and_Zero__(array)\
	if (array) std::memset(array, 0, ngrids * sizeof(double));\
	else array = new double[ngrids]();

std::tuple<
	std::vector<EigenMatrix>,
	std::vector<EigenMatrix>,
	std::vector<EigenVector>,
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
	std::vector<EigenMatrix> Fs(nmatrices, EigenZero(nbasis, nbasis));
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
	double* gds = nullptr; // Nuclear U gradients of density on grids
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
			EigenMatrix Dtmp = EigenZero(nbasis, nbasis);
			#pragma omp declare reduction(EigenMatrixSum: EigenMatrix: omp_out += omp_in) initializer(omp_priv = omp_orig)
			#pragma omp parallel for reduction(EigenMatrixSum: Dtmp)
			for ( int p = 0; p < nindbasis; p++ ) for ( int q = 0; q < p; q++ ) if ( std::abs(es[p] - es[q]) > 1.e-6 ){
				const double Upq = Us[imatrix](p, q) = Bs[imatrix](p, q) / ( es(p) - es(q) );
				Us[imatrix](q, p) = - Upq - CTSCs[imatrix](p, q);
				const EigenMatrix CpCqT = C.col(p) * C.col(q).transpose();
				Dtmp += ( ns[q] - ns[p] ) * Upq * ( CpCqT + CpCqT.transpose() );
			}
			Ds[imatrix] += Dtmp;
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
				__Allocate_and_Zero__(gds);
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
				__Allocate_and_Zero__(gd1xs);
				__Allocate_and_Zero__(gd1ys);
				__Allocate_and_Zero__(gd1zs);
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
			Fs[imatrix] = Fskeletons[imatrix] + undoneFUs[jundone];
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

	std::vector<EigenVector> Es(nmatrices, EigenZero(nindbasis, 1));
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
	return std::make_tuple(Us, Ds, Es, Ws, Fs);
}

std::vector<EigenVector> DensityOccupationGradient(
		EigenMatrix C, EigenVector es, EigenVector Nes,
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
		long int ngrids,
		int output, int nthreads){

	// Step 1: Preparing data used by all perturbations

	// HF part
	const int nmatrices = Ss.size();
	const int nbasis = C.rows();
	const int nindbasis = C.cols();
	std::vector<EigenVector> CTSCEs(nmatrices, EigenZero(nindbasis, 1));
	std::vector<EigenMatrix> Fs(nmatrices, EigenZero(nbasis, nbasis));
	std::vector<EigenMatrix> Ds(nmatrices, EigenZero(nbasis, nbasis));
	std::vector<EigenVector> Exs(nmatrices, EigenZero(nindbasis, 1));
	for ( int imatrix = 0; imatrix < nmatrices; imatrix++ )
		CTSCEs[imatrix] = (C.transpose() * Ss[imatrix] * C).diagonal().cwiseProduct(es);
	std::vector<EigenMatrix> Des(nindbasis, EigenZero(nbasis,nbasis));
	for ( int p = 0; p < nindbasis ; p++ )
		Des[p] = C.col(p) * C.col(p).transpose() * Nes(p);

	// KS part
	double* gds = nullptr; // Nuclear U gradients of density on grids
	double* gd1xs = nullptr;
	double* gd1ys = nullptr;
	double* gd1zs = nullptr;

	std::vector<std::deque<EigenMatrix>> Fs_deque(nmatrices);
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
			if ( Fs_deque[imatrix].size() > 1 )
				Fs[imatrix] = CDIIS(Rs_deque[imatrix], Fs_deque[imatrix]);
			Exs[imatrix] = (C.transpose() * Fs[imatrix] * C).diagonal() - CTSCEs[imatrix];
			Ds[imatrix] = EigenZero(nbasis, nbasis);
			for ( int p = 0; p < nindbasis; p++ )
				Ds[imatrix] += Des[p] * Exs[imatrix](p);
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
				__Allocate_and_Zero__(gds);
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
				__Allocate_and_Zero__(gd1xs);
				__Allocate_and_Zero__(gd1ys);
				__Allocate_and_Zero__(gd1zs);
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
			const EigenMatrix Fnew = Fskeletons[imatrix] + undoneFUs[jundone++];
			const EigenMatrix Fdiff = Fnew - Fs[imatrix];
			Fs_deque[imatrix].push_back(Fnew);
			Rs_deque[imatrix].push_back(Fdiff);
			if ( Fdiff.cwiseAbs().maxCoeff() < 1.e-5 )
				dones[imatrix] = 1;
			Fs[imatrix] = Fnew;
		}
		if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));
	}

	std::vector<EigenVector> Nxs(nmatrices, EigenZero(nindbasis, 1));
	for ( int imatrix = 0; imatrix < nmatrices; imatrix++ )
		Nxs[imatrix] = Exs[imatrix].cwiseProduct(Nes);
	return Nxs;
}



