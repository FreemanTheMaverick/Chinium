#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <deque>
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
#include "../Integral/Int4C2E.h"
#include "../Integral/Parallel.h"
#include "../Grid/GridAO.h"
#include "../Grid/GridDensity.h"
#include "../ExchangeCorrelation/MwfnXC1.h"
#include "../Grid/GridPotential.h"
#include "../DIIS/CDIIS.h"
#include "FockFormation.h"

#define __Occupation_Cutoff__ 1.e-8
#define __Convervence_Threshold__ 1.e-8

#define __Allocate_and_Zero__(array)\
	if (array) std::memset(array, 0, ngrids * sizeof(double));\
	else array = new double[ngrids]();

/*
std::vector<EigenMatrix> GxcU(
		std::vector<EigenMatrix>& Ds,
		std::vector<int> orders,
		double* ws,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* d1xs, double* d1ys, double* d1zs,
		double* vrs, double* vss,
		double* vrrs, double* vrss, double* vsss,
		long int ngrids){

	const int nmatrices = (int)Ds.size();
	const int nbasis = Ds[0].rows();

	EigenMatrix RhoPhi = EigenZero(ngrids, nbasis);
	for ( int ibasis = 0; ibasis < nbasis; ibasis++ ){
		const double* ix = ao1xs + ibasis * ngrids;
		const double* iy = ao1ys + ibasis * ngrids;
		const double* iz = ao1zs + ibasis * ngrids;
		for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
			RhoPhi(kgrid, ibasis) = d1xs[kgrid] * ix[kgrid] + d1ys[kgrid] * iy[kgrid] + d1zs[kgrid] * iz[kgrid];
		}
	}

	EigenMatrix A = EigenZero(ngrids, nmatrices);
	EigenMatrix B = EigenZero(ngrids, nmatrices);
	EigenMatrix Cx = EigenZero(ngrids, nmatrices);
	EigenMatrix Cy = EigenZero(ngrids, nmatrices);
	EigenMatrix Cz = EigenZero(ngrids, nmatrices);
	EigenVector ij = EigenZero(ngrids, 1);
	EigenVector ixj = EigenZero(ngrids, 1);
	EigenVector iyj = EigenZero(ngrids, 1);
	EigenVector izj = EigenZero(ngrids, 1);
	EigenVector irhophij = EigenZero(ngrids, 1);

	for ( int i = 0; i < nbasis; i++ ){
		const double* iao = aos + i * ngrids;
		const double* ix = ao1xs + i * ngrids;
		const double* iy = ao1ys + i * ngrids;
		const double* iz = ao1zs + i * ngrids;
		const EigenVector irhophi = RhoPhi.col(i);
		for ( int j = i; j < nbasis; j++ ){
			const double* jao = aos + j * ngrids;
			const double* jx = ao1xs + j * ngrids;
			const double* jy = ao1ys + j * ngrids;
			const double* jz = ao1zs + j * ngrids;
			const EigenVector jrhophi = RhoPhi.col(j);
			for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
				ij[kgrid] = iao[kgrid] * jao[kgrid];
				ixj[kgrid] = ix[kgrid] * jao[kgrid] + iao[kgrid] * jx[kgrid];
				iyj[kgrid] = iy[kgrid] * jao[kgrid] + iao[kgrid] * jy[kgrid];
				izj[kgrid] = iz[kgrid] * jao[kgrid] + iao[kgrid] * jz[kgrid];
				irhophij[kgrid] = irhophi(kgrid) * jao[kgrid] + iao[kgrid] * jrhophi(kgrid);
			}
			for ( int mat = 0; mat < nmatrices; mat++ ){
				const double fourDij = ( i == j ? 4 : 8 ) * Ds[mat](i, j);
				const double eightDij = ( i == j ? 8 : 16 ) * Ds[mat](i, j);
				for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
					A(kgrid, mat) += fourDij * (
							vrrs[kgrid] * ij[kgrid]
							+ 2 * vrss[kgrid] * irhophij[kgrid]
					);
					B(kgrid, mat) += eightDij * (
							vrss[kgrid] * ij[kgrid]
							+ 2 * vsss[kgrid] * irhophij[kgrid]
					);
					Cx(kgrid, mat) += eightDij * vss[kgrid] * ixj[kgrid];
					Cy(kgrid, mat) += eightDij * vss[kgrid] * iyj[kgrid];
					Cz(kgrid, mat) += eightDij * vss[kgrid] * izj[kgrid];
				}
			}
		}
	}
	std::vector<EigenMatrix> Vs(nmatrices, EigenZero(nbasis, nbasis));
	for ( int i = 0; i < nbasis; i++ ){
		const double* iao = aos + i * ngrids;
		const double* ix = ao1xs + i * ngrids;
		const double* iy = ao1ys + i * ngrids;
		const double* iz = ao1zs + i * ngrids;
		const EigenVector irhophi = RhoPhi.col(i);
		for ( int j = i; j < nbasis; j++ ){
			const double* jao = aos + j * ngrids;
			const double* jx = ao1xs + j * ngrids;
			const double* jy = ao1ys + j * ngrids;
			const double* jz = ao1zs + j * ngrids;
			const EigenVector jrhophi = RhoPhi.col(j);
			for ( int mat = 0; mat < nmatrices; mat++ ){
				double Vij = 0;
				for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
					Vij += ws[kgrid] * (
							A(kgrid, mat) * iao[kgrid] * jao[kgrid]
							+ B(kgrid, mat) * (
								irhophi(kgrid) * jao[kgrid]
								+ iao[kgrid] * jrhophi(kgrid)
							)
							+ Cx(kgrid, mat) * (
								ix[kgrid] * jao[kgrid]
								+ iao[kgrid] * jx[kgrid]
							)
							+ Cy(kgrid, mat) * (
								iy[kgrid] * jao[kgrid]
								+ iao[kgrid] * jy[kgrid]
							)
							+ Cz(kgrid, mat) * (
								iz[kgrid] * jao[kgrid]
								+ iao[kgrid] * jz[kgrid]
							)
					);
				}
				Vs[mat](i, j) = Vs[mat](j, i) = Vij;
			}
		}
	}

	return Vs;
}
*/

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
		Int4C2E& int4c2e,
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
	Eigen::setNbThreads(nthreads);

	// Preparing data used by all perturbations
	
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
		for ( int p = 0; p < nindbasis; p++ ) if ( ns[p] > __Occupation_Cutoff__ ){
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


	std::function<std::tuple<
		std::vector<EigenMatrix>,
		std::vector<EigenMatrix>,
		std::vector<EigenMatrix>
		> (std::vector<EigenMatrix>&, std::vector<bool>&)
	> update_func = [&](std::vector<EigenMatrix>& Bs_, std::vector<bool>& Dones_){
		std::vector<EigenMatrix> newBs_ = Bs_;
		std::vector<EigenMatrix> Rs_(nmatrices, EigenZero(nbasis, nbasis));
		int num_undone = nmatrices;
		for ( bool done : Dones_ ) if (done) num_undone--;

		if (output) std::printf("Constructing gradients of density matrix ...");
		auto density_start = __now__;
		std::vector<EigenMatrix> undoneDs(num_undone, EigenZero(nbasis, nbasis));
		for ( int imatrix = 0; imatrix < nmatrices; imatrix++ ) if ( !Dones_[imatrix] ){
			Ds[imatrix] = - Ns[imatrix];
			for ( int p = 0; p < nindbasis; p++ ) for ( int q = 0; q < p; q++ ) if ( std::abs(es[p] - es[q]) > __Occupation_Cutoff__ ){
				const double Upq = Us[imatrix](p, q) = Bs[imatrix](p, q) / ( es(p) - es(q) );
				Us[imatrix](q, p) = - Upq - CTSCs[imatrix](p, q);
			}
		}

		Eigen::setNbThreads(1);
		std::vector<std::vector<EigenArray>> Uarrays = Matrices2Arrays(Us);
		EigenMatrix* D_raw_arrays = new EigenMatrix[nmatrices]; // Must be a raw pointer for OMP reduction.
		for ( int imatrix = 0; imatrix < nmatrices; imatrix++ ){
			D_raw_arrays[imatrix].resize(nbasis, nbasis);
			D_raw_arrays[imatrix] = EigenZero(nbasis, nbasis);
		}
		std::vector<EigenArray> Up(nindbasis, EigenZero(nindbasis, 1).array());
		EigenMatrix CpCqT = EigenZero(nbasis, nbasis);
		EigenMatrix n_CpCqT_sym = EigenZero(nbasis, nbasis);
		#pragma omp declare\
		reduction(EigenMatrixSum: EigenMatrix: omp_out += omp_in)\
		initializer(omp_priv = omp_orig)
		#pragma omp parallel for\
		reduction(EigenMatrixSum: D_raw_arrays[:nmatrices])\
		firstprivate(Up, CpCqT, n_CpCqT_sym, C, ns)\
		schedule(dynamic, 1) num_threads(nthreads)
		for ( int p = nindbasis - 1; p >= 0; p-- ){
			Up = Uarrays[p];
			for ( int q = 0; q < p; q++ ) if ( std::abs(ns[p] - ns[q]) > 1.e-6 ){
				CpCqT = C.col(p) * C.col(q).transpose();
				n_CpCqT_sym = ( ns[q] - ns[p] ) * (CpCqT + CpCqT.transpose() );
				for ( int imatrix = 0; imatrix < nmatrices; imatrix++ ) if ( !Dones_[imatrix] )
					D_raw_arrays[imatrix] += Up[q](imatrix) * n_CpCqT_sym;
			}
		}
		for ( int imatrix = 0; imatrix < nmatrices; imatrix++ )
			Ds[imatrix] += D_raw_arrays[imatrix];
		delete [] D_raw_arrays;
		Eigen::setNbThreads(nthreads);

		for ( int imatrix = 0, jundone = 0; imatrix < nmatrices; imatrix++ ) if ( !Dones_[imatrix] )
			undoneDs[jundone++] = Ds[imatrix];
		if (output) std::printf(" Done in %f s\n", __duration__(density_start, __now__));

		if (output) std::printf("Constructing gradients of Fock matrix ...");
		auto fock_start = __now__;
		std::vector<EigenMatrix> undoneFU1s = int4c2e.ContractInts(undoneDs, nthreads);
		/*
		std::vector<EigenMatrix> undoneFU2s = GxcU(
				undoneDs,
				orders,
				ws,
				aos,
				ao1xs, ao1ys, ao1zs,
				d1xs, d1ys, d1zs,
				vrs, vss,
				vrrs, vrss, vsss,
				ngrids
		);
		*/
		for ( int imatrix = 0, jundone = 0; imatrix < nmatrices; imatrix++ ) if ( !Dones_[imatrix] ){
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
					nullptr, nullptr,
					nthreads
			);
			const EigenMatrix FU = undoneFU1s[jundone++] + PotentialSkeleton(
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
			Fs[imatrix] = Fskeletons[imatrix] + FU;
			newBs_[imatrix] = Bskeletons[imatrix] - C.transpose() * FU * C;
			Rs_[imatrix] = newBs_[imatrix] - Bs_[imatrix];
		}
		if (output) std::printf(" Done in %f s\n", __duration__(fock_start, __now__));
		return std::make_tuple(newBs_, Rs_, Rs_); // The last Rs_ is dummy.
	};

	CDIIS cdiis(&update_func, nmatrices, 10, __Convervence_Threshold__, 100, output>0 ? 1 : 0);
	if ( !cdiis.Run(Bs) ) throw std::runtime_error("Convergence failed!");

	delete [] gds;
	delete [] gd1xs;
	delete [] gd1ys;
	delete [] gd1zs;

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

std::vector<EigenVector> OccupationGradient(
		EigenMatrix C, EigenVector es,
		std::map<int, EigenMatrix> Dns,
		EigenVector Nes,
		std::vector<EigenMatrix>& Ss,
		std::vector<EigenMatrix>& Fskeletons,
		Int4C2E& int4c2e,
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
	Eigen::setNbThreads(nthreads);

	// Preparing data used by all perturbations

	// HF part
	const int nmatrices = Ss.size();
	const int nbasis = C.rows();
	const int nindbasis = C.cols();
	std::vector<EigenVector> CTSCEs(nmatrices, EigenZero(nindbasis, 1));
	std::vector<EigenMatrix> Fs = Fskeletons;
	std::vector<EigenMatrix> Ds(nmatrices, EigenZero(nbasis, nbasis));
	std::vector<EigenVector> Exs(nmatrices, EigenZero(nindbasis, 1));
	for ( int imatrix = 0; imatrix < nmatrices; imatrix++ )
		CTSCEs[imatrix] = (C.transpose() * Ss[imatrix] * C).diagonal().cwiseProduct(es);
	std::vector<EigenMatrix> Des(nindbasis, EigenZero(nbasis,nbasis));
	for ( int p = 0; p < nindbasis ; p++ ){
		if ( Dns.contains(p) ) Des[p] = Dns[p] * Nes(p);
		else Des[p] = C.col(p) * C.col(p).transpose() * Nes(p);
	}

	// KS part
	double* gds = nullptr; // Nuclear U gradients of density on grids
	double* gd1xs = nullptr;
	double* gd1ys = nullptr;
	double* gd1zs = nullptr;

	std::function<std::tuple<
		std::vector<EigenMatrix>,
		std::vector<EigenMatrix>,
		std::vector<EigenMatrix>
		> (std::vector<EigenMatrix>&, std::vector<bool>&)
	> update_func = [&](std::vector<EigenMatrix>& Fs_, std::vector<bool>& Dones_){
		std::vector<EigenMatrix> newFs_ = Fs_;
		std::vector<EigenMatrix> Rs_(nmatrices, EigenZero(nbasis, nbasis));
		int num_undone = nmatrices;
		for ( bool done : Dones_ ) if (done) num_undone--;

		std::vector<EigenMatrix> undoneDs(num_undone, EigenZero(nbasis, nbasis));
		for ( int imatrix = 0, jundone = 0; imatrix < nmatrices; imatrix++ ) if ( !Dones_[imatrix] ){
			Exs[imatrix] = (C.transpose() * Fs_[imatrix] * C).diagonal() - CTSCEs[imatrix];
			Ds[imatrix] = EigenZero(nbasis, nbasis);
			for ( int p = 0; p < nindbasis; p++ )
				Ds[imatrix] += Des[p] * Exs[imatrix](p);
			undoneDs[jundone++] = Ds[imatrix];
		}
		std::vector<EigenMatrix> undoneFUs = int4c2e.ContractInts(undoneDs, nthreads);
		for ( int imatrix = 0, jundone = 0; imatrix < nmatrices; imatrix++ ) if ( !Dones_[imatrix] ){
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
					nullptr, nullptr,
					nthreads
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
			newFs_[imatrix] = Fskeletons[imatrix] + undoneFUs[jundone++];
			Rs_[imatrix] = newFs_[imatrix] - Fs_[imatrix];
		}
		return std::make_tuple(newFs_, Rs_, Rs_); // The last Rs_ is dummy.
	};

	CDIIS cdiis(&update_func, nmatrices, 10, __Convervence_Threshold__, 100, output>0 ? 1 : 0);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");

	delete [] gds;
	delete [] gd1xs;
	delete [] gd1ys;
	delete [] gd1zs;

	std::vector<EigenVector> Nxs(nmatrices, EigenZero(nindbasis, 1));
	for ( int imatrix = 0; imatrix < nmatrices; imatrix++ )
		Nxs[imatrix] = Exs[imatrix].cwiseProduct(Nes);
	return Nxs;
}

std::map<int, EigenMatrix> OccupationFluctuation(
		EigenMatrix C, EigenVector es, EigenVector ns,
		std::vector<int> frac_indeces,
		Int4C2E& int4c2e,
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
	Eigen::setNbThreads(nthreads);

	// Preparing data used by all perturbations

	// HF part
	const int nbasis = C.rows();
	const int nindbasis = C.cols();
	const int nmatrices = (int)frac_indeces.size();
	std::vector<EigenMatrix> Gs(nmatrices, EigenZero(nindbasis, nindbasis));
	std::vector<EigenMatrix> Vs(nmatrices, EigenZero(nindbasis, nindbasis));
	std::vector<EigenMatrix> Ds(nmatrices, EigenZero(nbasis, nbasis));
	for ( int i = 0; i < nmatrices; i++ )
		Ds[i] = C.col(frac_indeces[i]) * C.col(frac_indeces[i]).transpose();


	// KS part
	double* gds = nullptr; // Nuclear U gradients of density on grids
	double* gd1xs = nullptr;
	double* gd1ys = nullptr;
	double* gd1zs = nullptr;


	std::function<std::tuple<
		std::vector<EigenMatrix>,
		std::vector<EigenMatrix>,
		std::vector<EigenMatrix>
		> (std::vector<EigenMatrix>&, std::vector<bool>&)
	> update_func = [&](std::vector<EigenMatrix>& Ds_, std::vector<bool>& Dones_){
		std::vector<EigenMatrix> newDs_ = Ds_;
		std::vector<EigenMatrix> Rs_(nmatrices, EigenZero(nbasis, nbasis));
		int num_undone = nmatrices;
		std::vector<EigenMatrix> undoneDs;
		for ( int i = 0; i < nmatrices; i++ ){
			if ( Dones_[i] ) num_undone--;
			else undoneDs.push_back(Ds_[i]);
		}

		if (output) std::printf("Constructing gradients of Fock matrix ...");
		auto fock_start = __now__;
		std::vector<EigenMatrix> undoneGs = int4c2e.ContractInts(undoneDs, nthreads);
		for ( int imatrix = 0, jundone = 0; imatrix < nmatrices; imatrix++ ) if ( !Dones_[imatrix] ){
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
					ngrids, 2*Ds_[imatrix],
					gds, gd1xs, gd1ys, gd1zs,
					nullptr, nullptr,
					nthreads
			);
			Gs[imatrix] = undoneGs[jundone++] + PotentialSkeleton(
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
		}
		if (output) std::printf(" Done in %f s\n", __duration__(fock_start, __now__));

		if (output) std::printf("Constructing gradients of density matrix ...");
		auto density_start = __now__;
		for ( int imatrix = 0; imatrix < nmatrices; imatrix++ ) if ( !Dones_[imatrix] ){
			const EigenMatrix Gmo = C.transpose() * Gs[imatrix] * C;
			for ( int p = 0; p < nindbasis; p++ ) for ( int q = 0; q < p; q++ ) if ( std::abs(es[p] - es[q]) > __Occupation_Cutoff__ ){
				const double Vpq = Vs[imatrix](p, q) = - Gmo(p, q) / ( es(p) - es(q) );
				Vs[imatrix](q, p) = - Vpq;
			}
			newDs_[imatrix] = C.col(frac_indeces[imatrix]) * C.col(frac_indeces[imatrix]).transpose();;
			for ( int q = 0; q < nindbasis; q++ ) for ( int p = q + 1; p < nindbasis; p++ ){
				const double nq_minus_np = ns(q) - ns(p);
				if ( std::abs(nq_minus_np) > __Occupation_Cutoff__*0 ){
					const EigenMatrix CpCqT = C.col(p) * C.col(q).transpose();
					const EigenMatrix Mpq = CpCqT + CpCqT.transpose();
					newDs_[imatrix] += nq_minus_np * Vs[imatrix](p, q) * Mpq;
				}
			}
			Rs_[imatrix] = newDs_[imatrix] - Ds_[imatrix];
		}
		if (output) std::printf(" Done in %f s\n", __duration__(density_start, __now__));
		return std::make_tuple(newDs_, Rs_, Rs_); // The last Rs_ is dummy.
	};

	CDIIS cdiis(&update_func, nmatrices, 10, __Convervence_Threshold__, 100, output>0 ? 1 : 0);
	if ( !cdiis.Run(Ds) ) throw std::runtime_error("Convergence failed!");

	delete [] gds;
	delete [] gd1xs;
	delete [] gd1ys;
	delete [] gd1zs;
	
	std::map<int, EigenMatrix> Dns;
	for ( int i = 0; i < nmatrices; i++ ){
		const int key = frac_indeces[i];
		const EigenMatrix value = Ds[i];
		Dns[key] = value;
	}
	return Dns;
}
