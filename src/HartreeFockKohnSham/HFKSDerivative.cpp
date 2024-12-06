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
	const int natoms = gds.size();
	GetDensitySkeleton(
			orders,
			aos,
			ao1xs, ao1ys, ao1zs,
			ao2xxs, ao2yys, ao2zzs,
			ao2xys, ao2xzs, ao2yzs,
			ngrids, D,
			bf2atom,
			gds, gd1xs, gd1ys, gd1zs
	);
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

#define iyx ixy
#define izx ixz
#define izy iyz

std::vector<std::vector<EigenMatrix>> GxcSkeleton(
		std::vector<int>& orders,
		double* ws, long int ngrids,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2xxs, double* ao2yys, double* ao2zzs,
		double* ao2xys, double* ao2xzs, double* ao2yzs,
		double* d1xs, double* d1ys, double* d1zs,
		double* vrs, double* vss,
		std::vector<int>& bf2atom){
	const int nbasis = bf2atom.size();
	const int natoms = *std::max_element(bf2atom.begin(), bf2atom.end()) + 1;
	std::vector<std::vector<EigenMatrix>> Gs(natoms, {EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis)});
	for ( int i = 0; i < nbasis; i++ ){
		const double* ix = ao1xs + i * ngrids;
		const double* iy = ao1ys + i * ngrids;
		const double* iz = ao1zs + i * ngrids;
		const double* ixx = ao2xxs + i * ngrids;
		const double* iyy = ao2yys + i * ngrids;
		const double* izz = ao2zzs + i * ngrids;
		const double* ixy = ao2xys + i * ngrids;
		const double* ixz = ao2xzs + i * ngrids;
		const double* iyz = ao2yzs + i * ngrids;
		const int iatom = bf2atom[i];
		for ( int j = 0; j < nbasis; j++ ){
			const double* jao = aos + j * ngrids;
			const double* jx = ao1xs + j * ngrids;
			const double* jy = ao1ys + j * ngrids;
			const double* jz = ao1zs + j * ngrids;
			double tmp1 = 0;
			double tmp2 = 0;
			if (std::find(orders.begin(), orders.end(), 0) != orders.end()) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
				tmp1 = 2 * ws[kgrid] * vrs[kgrid] * jao[kgrid];
				Gs[iatom][0](i, j) -= tmp1 * ix[kgrid];
				Gs[iatom][1](i, j) -= tmp1 * iy[kgrid];
				Gs[iatom][2](i, j) -= tmp1 * iz[kgrid];
			}
			if (std::find(orders.begin(), orders.end(), 1) != orders.end()) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
				tmp1 = 4 * ws[kgrid] * vss[kgrid];
				tmp2 = d1xs[kgrid] * jx[kgrid] + d1ys[kgrid] * jy[kgrid] + d1zs[kgrid] * jz[kgrid];
				Gs[iatom][0](i, j) -= tmp1 * (
					(
						d1xs[kgrid] * ixx[kgrid] +
						d1ys[kgrid] * ixy[kgrid] +
						d1zs[kgrid] * ixz[kgrid]
					) * jao[kgrid] +
					ix[kgrid] * tmp2
				);
				Gs[iatom][1](i, j) -= tmp1 * (
					(
						d1xs[kgrid] * iyx[kgrid] +
						d1ys[kgrid] * iyy[kgrid] +
						d1zs[kgrid] * iyz[kgrid]
					) * jao[kgrid] +
					iy[kgrid] * tmp2
				);
				Gs[iatom][2](i, j) -= tmp1 * (
					(
						d1xs[kgrid] * izx[kgrid] +
						d1ys[kgrid] * izy[kgrid] +
						d1zs[kgrid] * izz[kgrid]
					) * jao[kgrid] +
					iz[kgrid] * tmp2
				);
			}
		}
	}
	for ( int iatom = 0; iatom < natoms; iatom++ ) for ( int xyz = 0; xyz < 3; xyz++ ){
		const EigenMatrix G = Gs[iatom][xyz];
		Gs[iatom][xyz] = 0.5 * ( G + G.transpose() );
	}
	return Gs;
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

void Multiwfn::HFKSDerivative(int derivative, int output, int nthreads){
	const int natoms = this->getNumCenters();
	const int nbasis = this->getNumBasis();
	const long int ngrids = this->NumGrids;
	std::vector<int> bf2atom(nbasis);
	for ( int iatom = 0, kbasis = 0; iatom < natoms; iatom++ )
		for ( int jbasis = 0; jbasis < this->Centers[iatom].getNumBasis(); jbasis++, kbasis++ )
			bf2atom[kbasis] = iatom;

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

	// Nuclear gradient of density in batches
	// They are also necessary for nuclear hessian of DFT energy, so we store them temporarily in std::vector.
	std::vector<std::vector<double*>> batch_gds(natoms);
	std::vector<std::vector<double*>> batch_gd1xs(natoms);
	std::vector<std::vector<double*>> batch_gd1ys(natoms);
	std::vector<std::vector<double*>> batch_gd1zs(natoms);
	for ( int iatom = 0; iatom < natoms; iatom++ )
		batch_gds[iatom] = batch_gd1xs[iatom] = batch_gd1ys[iatom] = batch_gd1zs[iatom] = {nullptr, nullptr, nullptr};

	if (this->XC.XCcode){

		EigenMatrix xcg = EigenZero(natoms, 3);
		// Distributing grids to batches
		const long int ngrids_per_batch = derivative < 2 ? 3000 : ngrids; // If only gradient is required, the grids will be done in batches. Otherwise, they will be done simultaneously, because the intermediate variable, nuclear gradient of density on grids, is required in CPSCF.
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
				__Allocate_and_Zero_2__(batch_gds);
			}else if ( this->XC.XCfamily.compare("GGA") == 0 ){
				order = 1;
				__Allocate_and_Zero_2__(batch_gds);
				__Allocate_and_Zero_2__(batch_gd1xs);
				__Allocate_and_Zero_2__(batch_gd1ys);
				__Allocate_and_Zero_2__(batch_gd1zs);
			}
			if ( derivative < 2 ){
				if ( this->XC.XCfamily.compare("LDA") == 0 ){
					__Allocate_and_Zero__(batch_aos);
					__Allocate_and_Zero__(batch_ao1xs);
					__Allocate_and_Zero__(batch_ao1ys);
					__Allocate_and_Zero__(batch_ao1zs);
				}else if ( this->XC.XCfamily.compare("GGA") == 0 ){
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
						batch_ao2xys, batch_ao2xzs, batch_ao2yzs
				);
			}else{
				if ( this->XC.XCfamily.compare("LDA") == 0 ){
					assert(this->AOs && "AOs on grids do not exist!");
					assert(this->AO1Xs && "First-order x-derivatives of AOs on grids do not exist!");
					assert(this->AO1Ys && "First-order y-derivatives of AOs on grids do not exist!");
					assert(this->AO1Zs && "First-order z-derivatives of AOs on grids do not exist!");
				}else if ( this->XC.XCfamily.compare("GGA") == 0 ){
					assert(this->AOs && "AOs on grids do not exist!");
					assert(this->AO1Xs && "First-order x-derivatives of AOs on grids do not exist!");
					assert(this->AO1Ys && "First-order y-derivatives of AOs on grids do not exist!");
					assert(this->AO1Zs && "First-order z-derivatives of AOs on grids do not exist!");
					assert(this->AO2XXs && "Second-order xx-derivatives of AOs on grids do not exist!");
					assert(this->AO2YYs && "Second-order yy-derivatives of AOs on grids do not exist!");
					assert(this->AO2ZZs && "Second-order zz-derivatives of AOs on grids do not exist!");
					assert(this->AO2XYs && "Second-order xy-derivatives of AOs on grids do not exist!");
					assert(this->AO2XZs && "Second-order xz-derivatives of AOs on grids do not exist!");
					assert(this->AO2YZs && "Second-order yz-derivatives of AOs on grids do not exist!");
				}
				batch_aos = this->AOs;
				batch_ao1xs = this->AO1Xs;
				batch_ao1ys = this->AO1Ys;
				batch_ao1zs = this->AO1Zs;
				batch_ao2xxs = this->AO2XXs;
				batch_ao2yys = this->AO2YYs;
				batch_ao2zzs = this->AO2ZZs;
				batch_ao2xys = this->AO2XYs;
				batch_ao2xzs = this->AO2XZs;
				batch_ao2yzs = this->AO2YZs;
			}
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

	if ( derivative >= 2){
		if (output) std::printf("Coupled-perturbed self-consistent-field ...\n");
		auto start = __now__;
		std::vector<EigenMatrix> Ss(3 * natoms, EigenZero(nbasis, nbasis));
		std::vector<EigenMatrix> Fskeletons(3 * natoms, EigenZero(nbasis, nbasis));
		std::vector<std::vector<EigenMatrix>> fxcskeletons(natoms, {EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis)});
		std::vector<int> orders = {};
		if (this->XC.XCcode){
			if (this->XC.XCfamily.compare("LDA") == 0)
				orders = {0};
			else if (this->XC.XCfamily.compare("GGA") == 0)
				orders = {0, 1};
			this->XC.Evaluate(
					"f", ngrids,
					this->Rhos,
					this->Sigmas,
					nullptr, nullptr,
					nullptr,
					nullptr, nullptr, nullptr, nullptr,
					this->E2Rho2s, this->E2RhoSigmas, this->E2Sigma2s,
					nullptr, nullptr, nullptr, nullptr,
					nullptr, nullptr, nullptr, nullptr, nullptr
			);
			fxcskeletons = GxcSkeleton(
					orders,
					this->Ws, ngrids,
					this->AOs,
					this->AO1Xs, this->AO1Ys, this->AO1Zs,
					this->AO2XXs, this->AO2YYs, this->AO2ZZs,
					this->AO2XYs, this->AO2XZs, this->AO2YZs,
					this->Rho1Xs, this->Rho1Ys, this->Rho1Zs,
					this->E1Rhos, this->E1Sigmas,
					bf2atom
			);
		}

		for ( int icenter = 0; icenter < natoms; icenter++ ) for ( int xyz = 0; xyz < 3; xyz++ ){
			Ss[3 * icenter + xyz] = this->OverlapGrads[icenter][xyz];
			Fskeletons[3 * icenter + xyz] = Hgrads[icenter][xyz] + this->GGrads[icenter][xyz] + fxcskeletons[icenter][xyz];
		}
		
		auto [Us, dDs, dEs, dWs] = NonIdempotent( // std::vector<EigenMatrix>
				this->getCoefficientMatrix(),
				this->getEnergy(),
				this->getOccupation() / 2.,
				Ss, Fskeletons,
				this->RepulsionIs, this->RepulsionJs,
				this->RepulsionKs, this->RepulsionLs,
				this->RepulsionDegs, this->Repulsions,
				this->RepulsionLength, this->XC.EXX,
				orders,
				this->Ws,
				this->AOs,
				this->AO1Xs, this->AO1Ys, this->AO1Zs,
				this->AO2XXs, this->AO2YYs, this->AO2ZZs,
				this->AO2XYs, this->AO2XZs, this->AO2YZs,
				this->Rho1Xs, this->Rho1Ys, this->Rho1Zs,
				this->E1Rhos, this->E1Sigmas,
				this->E2Rho2s, this->E2RhoSigmas, this->E2Sigma2s,
				batch_gds, batch_gd1xs, batch_gd1ys, batch_gd1zs,
				this->NumGrids,
				output - 1, nthreads
		);
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


