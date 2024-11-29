#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <string>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cassert>
#include <omp.h>

#include "../Macro.h"
#include "../Multiwfn.h"
#include "../Grid/GridAO.h"
#include "../Grid/GridDensity.h"
#include "../ExchangeCorrelation/MwfnXC1.h"
#include "../Grid/GridPotential.h"
#include "CoupledPerturbed.h"

#include <iostream>

EigenMatrix HFGradient(
		EigenMatrix D, std::vector<std::vector<EigenMatrix>>& Hgrads,
		EigenMatrix W, std::vector<std::vector<EigenMatrix>>& Sgrads,
		std::vector<std::vector<EigenMatrix>>& Ggrads){
	EigenMatrix g = EigenZero(Hgrads.size(), 3);
	for ( int iatom = 0; iatom < (int)Hgrads.size(); iatom++ )
		for ( int xyz = 0; xyz < 3; xyz++ )
			g(iatom, xyz) = ( D * Hgrads[iatom][xyz] - W * Sgrads[iatom][xyz] + 0.5 * D * Ggrads[iatom][xyz] ).trace();
	return g;
}

#define __Check_Vector_Array__(vec)\
	vec.size() > 0 && vec[0].size() == 3 && vec[0][0]

EigenMatrix XCGradient(
		int order,
		EigenMatrix D, std::vector<int>& bf2atom,
		double* ws, long int ngrids,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2xxs, double* ao2yys, double* ao2zzs,
		double* ao2xys, double* ao2xzs, double* ao2yzs,
		double* rho1xs, double* rho1ys, double* rho1zs,
		double* erhos, double* esigmas,
		std::vector<std::vector<double*>>& gds,
		std::vector<std::vector<double*>>& gd1xs,
		std::vector<std::vector<double*>>& gd1ys,
		std::vector<std::vector<double*>>& gd1zs){
	std::vector<int> orders = {};
	if ( order >= 0 ){
		orders.push_back(0);
		assert(aos && "AOs on grids do not exist!");
		assert(ao1xs && "First-order x-derivatives of AOs on grids do not exist!");
		assert(ao1ys && "First-order y-derivatives of AOs on grids do not exist!");
		assert(ao1zs && "First-order z-derivatives of AOs on grids do not exist!");
		assert(erhos && "First-order XC potential w.r.t. density on grids do not exist!");
	}
	if ( order >= 1 ){
		orders.push_back(1);
		assert(ao2xxs && "Second-order xx-derivatives of AOs on grids do not exist!");
		assert(ao2yys && "Second-order yy-derivatives of AOs on grids do not exist!");
		assert(ao2zzs && "Second-order zz-derivatives of AOs on grids do not exist!");
		assert(ao2xys && "Second-order xy-derivatives of AOs on grids do not exist!");
		assert(ao2xzs && "Second-order xz-derivatives of AOs on grids do not exist!");
		assert(ao2yzs && "Second-order yz-derivatives of AOs on grids do not exist!");
		assert(rho1xs && "First-order x-derivatives of density on grids do not exist!");
		assert(rho1ys && "First-order y-derivatives of density on grids do not exist!");
		assert(rho1zs && "First-order z-derivatives of density on grids do not exist!");
		assert(esigmas && "First-order XC potential w.r.t. sigma on grids do not exist!");
	}
	GetDensitySkeleton(
			orders,
			aos,
			ao1xs, ao1ys, ao1zs,
			ao2xxs, ao2yys, ao2zzs,
			ao2xys, ao2xzs, ao2yzs,
			ngrids, D,
			bf2atom,
			gds, gd1xs, gd1ys, gd1zs);
	const int natoms = gds.size();
	EigenMatrix xcg0 = EigenZero(natoms, 3);
	if ( std::find(orders.begin(), orders.end(), 0) != orders.end() ){
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			for ( int xyz = 0; xyz < 3; xyz++ ){
				double* this_g = &xcg0(iatom, xyz);
				double* this_gds = gds[iatom][xyz];
				for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
					*this_g += ws[kgrid] * erhos[kgrid] * this_gds[kgrid];
				}
			}
		}
	}
	EigenMatrix xcg1 = EigenZero(natoms, 3);
	if ( std::find(orders.begin(), orders.end(), 1) != orders.end() ){
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			for ( int xyz = 0; xyz < 3; xyz++ ){
				double* this_g = &xcg1(iatom, xyz);
				double* this_gd1xs = gd1xs[iatom][xyz];
				double* this_gd1ys = gd1ys[iatom][xyz];
				double* this_gd1zs = gd1zs[iatom][xyz];
				for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
					*this_g += ws[kgrid] * esigmas[kgrid] * (
							  rho1xs[kgrid] * this_gd1xs[kgrid]
							+ rho1ys[kgrid] * this_gd1ys[kgrid]
							+ rho1zs[kgrid] * this_gd1zs[kgrid]);
				}
			}
		}
	}
	return xcg0 + 2 * xcg1;
}




#define __Allocate_and_Zero__(array)\
	if (array) std::memset(array, 0, this->getNumBasis() * ngrids_this_batch * sizeof(double));\
	else array = new double[this->getNumBasis() * ngrids_this_batch]();

#define __Allocate_and_Zero_2__(vector)\
	for ( int icenter = 0; icenter < this->getNumCenters(); icenter++ )\
		for ( int xyz = 0; xyz < 3; xyz++ ){\
			if (vector[icenter][xyz]) std::memset(vector[icenter][xyz], 0, ngrids_this_batch * sizeof(double));\
			else vector[icenter][xyz] = new double[ngrids_this_batch]();\
		}

void Multiwfn::HFKSDerivative(int order, int output, int nthreads){
	const int natoms = this->getNumCenters();
	const int nbasis = this->getNumBasis();
	const long int ngrids = this->NumGrids;
	assert((int)this->OverlapGrads.size() == natoms && "Overlap matrix gradient does not exist!");
	assert((int)this->KineticGrads.size() == natoms && "Kinetic matrix gradient does not exist!");
	assert((int)this->NuclearGrads.size() == natoms && "Nuclear matrix gradient does not exist!");
	assert((int)this->GGrads.size() == natoms && "Two-electron matrix gradient does not exist!");
	std::vector<std::vector<EigenMatrix>> Hgrads = {};
	for ( int iatom = 0; iatom < natoms; iatom++ )
		Hgrads.push_back({
			this->KineticGrads[iatom][0] + this->NuclearGrads[iatom][0],
			this->KineticGrads[iatom][1] + this->NuclearGrads[iatom][1],
			this->KineticGrads[iatom][2] + this->NuclearGrads[iatom][2]
		});
	this->Gradient += HFGradient(this->getDensity(), Hgrads, this->getEnergyDensity(), this->OverlapGrads, this->GGrads);

	if (this->XC.XCcode){
		std::vector<int> bf2atom(nbasis);
		for ( int iatom = 0, kbasis = 0; iatom < natoms; iatom++ )
			for ( int jbasis = 0; jbasis < this->Centers[iatom].getNumBasis(); jbasis++, kbasis++ )
				bf2atom[kbasis] = iatom;

		EigenMatrix xcg = EigenZero(natoms, 3);
		// Distributing grids to batches
		const long int ngrids_per_batch = 3000;
		//const long int ngrids_per_batch = 65536;
		std::vector<long int> batch_tails = {};
		long int tmp = 0;
		while ( tmp < ngrids ){
			if ( tmp + ngrids_per_batch < ngrids )
				batch_tails.push_back(tmp + ngrids_per_batch);
			else batch_tails.push_back(ngrids);
			tmp += ngrids_per_batch;
		}
		const int nbatches = batch_tails.size();
		std::printf("Calculating XC nuclear gradient in %d batches ...\n", nbatches);
		const auto start_all = __now__;

		// AO values in batches
		double* batch_aos = nullptr;
		double* batch_ao1xs = nullptr;
		double* batch_ao1ys = nullptr;
		double* batch_ao1zs = nullptr;
		double* batch_ao2xxs = nullptr;
		double* batch_ao2yys = nullptr;
		double* batch_ao2zzs = nullptr;
		double* batch_ao2xys = nullptr;
		double* batch_ao2xzs = nullptr;
		double* batch_ao2yzs = nullptr;

		// Nuclear gradient of density in batches
		// They are also necessary for nuclear hessian of DFT energy, so we store them temporarily in std::vector.
		std::vector<std::vector<double*>> batch_gds(natoms);
		std::vector<std::vector<double*>> batch_gd1xs(natoms);
		std::vector<std::vector<double*>> batch_gd1ys(natoms);
		std::vector<std::vector<double*>> batch_gd1zs(natoms);
		for ( int iatom = 0; iatom < natoms; iatom++ )
			batch_gds[iatom] = batch_gd1xs[iatom] = batch_gd1ys[iatom] = batch_gd1zs[iatom] = {nullptr, nullptr, nullptr};
		for ( int ibatch = 0; ibatch < nbatches; ibatch++ ){
			std::printf("| Batch %d\n", ibatch);
			const long int grid_head = (ibatch == 0) ? 0 : batch_tails[ibatch - 1];
			const long int grid_tail = batch_tails[ibatch];
			const long int ngrids_this_batch = grid_tail - grid_head;
			if (output) std::printf("| | Evaluating AO of %ld grids ...", ngrids_this_batch);
			auto start = __now__;
			int order = 0;
			if ( this->XC.XCfamily.compare("LDA") == 0 ){
				order = 0;
				__Allocate_and_Zero__(batch_aos);
				__Allocate_and_Zero__(batch_ao1xs);
				__Allocate_and_Zero__(batch_ao1ys);
				__Allocate_and_Zero__(batch_ao1zs);
				__Allocate_and_Zero_2__(batch_gds);
			}else if ( this->XC.XCfamily.compare("GGA") == 0 ){
				order = 1;
				__Allocate_and_Zero__(batch_aos);
				__Allocate_and_Zero__(batch_ao1xs);
				__Allocate_and_Zero__(batch_ao1ys);
				__Allocate_and_Zero__(batch_ao1zs);
				__Allocate_and_Zero__(batch_ao2xxs);
				__Allocate_and_Zero__(batch_ao2yys);
				__Allocate_and_Zero__(batch_ao2zzs);
				__Allocate_and_Zero__(batch_ao2xys);
				__Allocate_and_Zero__(batch_ao2xzs);
				__Allocate_and_Zero__(batch_ao2yzs);
				__Allocate_and_Zero_2__(batch_gds);
				__Allocate_and_Zero_2__(batch_gd1xs);
				__Allocate_and_Zero_2__(batch_gd1ys);
				__Allocate_and_Zero_2__(batch_gd1zs);
			}
			GetAoValues(
					this->Centers,
					this->Xs + grid_head,
					this->Ys + grid_head,
					this->Zs + grid_head,
					ngrids_this_batch,
					batch_aos,
					batch_ao1xs, batch_ao1ys, batch_ao1zs,
					nullptr,
					batch_ao2xxs, batch_ao2yys, batch_ao2zzs,
					batch_ao2xys, batch_ao2xzs, batch_ao2yzs);
			if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
			if (output) std::printf("| | Calculating XC gradient contributed by these grids ...");
			start = __now__;
			xcg += XCGradient(
				order,
				this->getDensity(), bf2atom,
				this->Ws + grid_head, ngrids_this_batch,
				batch_aos,
				batch_ao1xs, batch_ao1ys, batch_ao1zs,
				batch_ao2xxs, batch_ao2yys, batch_ao2zzs,
				batch_ao2xys, batch_ao2xzs, batch_ao2yzs,
				this->Rho1Xs + grid_head,
				this->Rho1Ys + grid_head,
				this->Rho1Zs + grid_head,
				this->E1Rhos + grid_head,
				this->E1Sigmas + grid_head,
				batch_gds, batch_gd1xs, batch_gd1ys, batch_gd1zs);
			if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
		}
		if (output) std::printf("| Done in %f s\n", __duration__(start_all, __now__));
		this->Gradient += xcg;
	}

	if ( order >= 2){
		// CPSCF
		if (output) std::printf("Coupled-perturbed self-consistent-field ...\n");
		auto start = __now__;
		std::vector<EigenMatrix> Ss(3 * natoms, EigenZero(nbasis, nbasis));
		std::vector<EigenMatrix> Fskeletons(3 * natoms, EigenZero(nbasis, nbasis));
		for ( int icenter = 0; icenter < natoms; icenter++ ) for ( int xyz = 0; xyz < 3; xyz++ ){
			Ss[3 * icenter + xyz] = this->OverlapGrads[icenter][xyz];
			Fskeletons[3 * icenter + xyz] = Hgrads[icenter][xyz] + this->GGrads[icenter][xyz];
		}
		auto [Us, dDs, dEs, dWs] = NonIdempotent( // std::vector<EigenMatrix>
				this->getCoefficientMatrix(),
				this->getEnergy(),
				this->getOccupation() / 2.,
				Ss, Fskeletons,
				this->RepulsionIs, this->RepulsionJs,
				this->RepulsionKs, this->RepulsionLs,
				this->RepulsionDegs, this->Repulsions,
				this->RepulsionLength, this->XC.EXX, output - 1, nthreads);
		if (output) std::printf("| Done in %f s\n", __duration__(start, __now__));
		EigenMatrix hfh = EigenZero(3 * natoms, 3 * natoms);
		for ( int xpert = 0; xpert < 3 * natoms; xpert++ )
			for ( int ypert = 0; ypert <= xpert; ypert++ )
				hfh(xpert, ypert) = hfh(ypert, xpert) =
					( dDs[xpert] * Fskeletons[ypert] ).trace()
					- ( dWs[xpert] * Ss[ypert] ).trace();
		hfh += this->KineticHess + this->NuclearHess - this->OverlapHess + 0.5 * this->GHess;
		this->Hessian += 2 * hfh;
	}
}


