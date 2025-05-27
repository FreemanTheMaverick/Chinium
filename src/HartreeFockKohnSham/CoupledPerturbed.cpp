#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
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
#include "../Grid/Grid.h"
#include "../DIIS/CDIIS.h"

#define __Occupation_Cutoff__ 1.e-8
#define __Convervence_Threshold__ 1.e-8

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
		Int4C2E& int4c2e, Grid& grid,
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
		for ( int imatrix = 0, jundone = 0; imatrix < nmatrices; imatrix++ ) if ( !Dones_[imatrix] ){
			grid.getGridDensity(2 * Ds[imatrix]);
			const EigenMatrix FU = undoneFU1s[jundone++] + grid.getFockDensitySelf();
			Fs[imatrix] = Fskeletons[imatrix] + FU;
			newBs_[imatrix] = Bskeletons[imatrix] - C.transpose() * FU * C;
			Rs_[imatrix] = newBs_[imatrix] - Bs_[imatrix];
		}
		if (output) std::printf(" Done in %f s\n", __duration__(fock_start, __now__));
		return std::make_tuple(newBs_, Rs_, Rs_); // The last Rs_ is dummy.
	};

	CDIIS cdiis(&update_func, nmatrices, 10, __Convervence_Threshold__, 100, output>0 ? 1 : 0);
	if ( !cdiis.Run(Bs) ) throw std::runtime_error("Convergence failed!");

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
		Int4C2E& int4c2e, Grid& grid,
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
			grid.getGridDensity(2 * Ds[imatrix]);
			undoneFUs[jundone] += grid.getFockDensitySelf();
			newFs_[imatrix] = Fskeletons[imatrix] + undoneFUs[jundone++];
			Rs_[imatrix] = newFs_[imatrix] - Fs_[imatrix];
		}
		return std::make_tuple(newFs_, Rs_, Rs_); // The last Rs_ is dummy.
	};

	CDIIS cdiis(&update_func, nmatrices, 10, __Convervence_Threshold__, 100, output>0 ? 1 : 0);
	if ( !cdiis.Run(Fs) ) throw std::runtime_error("Convergence failed!");

	std::vector<EigenVector> Nxs(nmatrices, EigenZero(nindbasis, 1));
	for ( int imatrix = 0; imatrix < nmatrices; imatrix++ )
		Nxs[imatrix] = Exs[imatrix].cwiseProduct(Nes);
	return Nxs;
}

std::map<int, EigenMatrix> OccupationFluctuation(
		EigenMatrix C, EigenVector es, EigenVector ns,
		std::vector<int> frac_indeces,
		Int4C2E& int4c2e, Grid& grid,
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
			grid.getGridDensity(2 * Ds_[imatrix]);
			Gs[imatrix] = undoneGs[jundone++] + grid.getFockDensitySelf();
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

	std::map<int, EigenMatrix> Dns;
	for ( int i = 0; i < nmatrices; i++ ){
		const int key = frac_indeces[i];
		const EigenMatrix value = Ds[i];
		Dns[key] = value;
	}
	return Dns;
}
