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
	vec.size() && vec[0].size() && vec[0][0]

EigenMatrix XCGradient(
		int order,
		double* ws, long int ngrids,
		double* rho1xs, double* rho1ys, double* rho1zs,
		double* erhos, double* esigmas,
		std::vector<std::vector<double*>>& gds,
		std::vector<std::vector<double*>>& gd1xs,
		std::vector<std::vector<double*>>& gd1ys,
		std::vector<std::vector<double*>>& gd1zs){
	std::vector<int> orders = {};
	if ( order >= 0 ){
		orders.push_back(0);
		assert(erhos && "First-order XC potential w.r.t. density on grids do not exist!");
		assert(__Check_Vector_Array__(gds) && "Nuclear Gradient of density on grids array does not exist!");
	}
	if ( order >= 1 ){
		orders.push_back(1);
		assert(rho1xs && "First-order x-derivatives of density on grids do not exist!");
		assert(rho1ys && "First-order y-derivatives of density on grids do not exist!");
		assert(rho1zs && "First-order z-derivatives of density on grids do not exist!");
		assert(esigmas && "First-order XC potential w.r.t. sigma on grids do not exist!");
		assert(__Check_Vector_Array__(gd1xs) && "Nuclear Gradient of first-order x-derivatives of density on grids does not exist!");
		assert(__Check_Vector_Array__(gd1ys) && "Nuclear Gradient of first-order y-derivatives of density on grids does not exist!");
		assert(__Check_Vector_Array__(gd1zs) && "Nuclear Gradient of first-order z-derivatives of density on grids does not exist!");
	}
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

EigenMatrix HFHessian(
		EigenMatrix Eskeleton,
		std::vector<EigenMatrix>& dDs, std::vector<EigenMatrix>& Fskeletons,
		std::vector<EigenMatrix>& dWs, std::vector<EigenMatrix>& Sgrads){

	const int nmatrices = dDs.size();
	EigenMatrix hfh = EigenZero(nmatrices, nmatrices);
	for ( int xpert = 0; xpert < nmatrices; xpert++ )
		for ( int ypert = 0; ypert <= xpert; ypert++ )
			hfh(xpert, ypert) = hfh(ypert, xpert) =
				( dDs[xpert] * Fskeletons[ypert] ).trace()
				- ( dWs[xpert] * Sgrads[ypert] ).trace();
	hfh += Eskeleton;
	return 2 * hfh;
}

EigenMatrix HxcSkeleton(
		std::vector<int> orders,
		long int ngrids, double* ws,
		bool part1,
		double* d1xs, double* d1ys, double* d1zs,
		double* vrrs, double* vrss, double* vsss,
		std::vector<std::vector<double*>>& gds,
		std::vector<std::vector<double*>>& gd1xs,
		std::vector<std::vector<double*>>& gd1ys,
		std::vector<std::vector<double*>>& gd1zs,
		bool part2,
		double* vrs, double* vss,
		std::vector<std::vector<double*>>& hds,
		std::vector<std::vector<double*>>& hd1xs,
		std::vector<std::vector<double*>>& hd1ys,
		std::vector<std::vector<double*>>& hd1zs){
	bool zeroth = 0;
	bool first = 0;
	if (std::find(orders.begin(), orders.end(), 0) != orders.end()){
		zeroth = 1;
		if (part1){
			assert(vrrs && "Second-order XC potential w.r.t. density on grids do not exist!");
			assert(__Check_Vector_Array__(gds) && "Nuclear gradient of density on grids array does not exist!");
		}
		if (part2){
			assert(vrs && "First-order XC potential w.r.t. density on grids do not exist!");
			assert(__Check_Vector_Array__(hds) && "Nuclear hessian of density on grids array does not exist!");
		}
	}
	if (std::find(orders.begin(), orders.end(), 1) != orders.end()){
		first = 1;
		if (part1){
			assert(vss && "First-order XC potential w.r.t. sigma on grids do not exist!");
			assert(vrss && "Second-order XC potential w.r.t. density and sigma on grids do not exist!");
			assert(vsss && "Second-order XC potential w.r.t. sigma on grids do not exist!");
			assert(__Check_Vector_Array__(gd1xs) && "Nuclear gradient of first-order x-derivative of density on grids array does not exist!");
			assert(__Check_Vector_Array__(gd1ys) && "Nuclear gradient of first-order y-derivative of density on grids array does not exist!");
			assert(__Check_Vector_Array__(gd1zs) && "Nuclear gradient of first-order z-derivative of density on grids array does not exist!");
		}
		if (part2){
			assert(d1xs && "First-order x-derivatives of density on grids do not exist!");
			assert(d1ys && "First-order y-derivatives of density on grids do not exist!");
			assert(d1zs && "First-order z-derivatives of density on grids do not exist!");
			assert(vss && "First-order XC potential w.r.t. sigma on grids do not exist!");
			assert(__Check_Vector_Array__(hd1xs) && "Nuclear hessian of first-order x-derivative of density on grids array does not exist!");
			assert(__Check_Vector_Array__(hd1ys) && "Nuclear hessian of first-order y-derivative of density on grids array does not exist!");
			assert(__Check_Vector_Array__(hd1zs) && "Nuclear hessian of first-order z-derivative of density on grids array does not exist!");
		}
	}
	const int nmatrices = hds.size();

	std::vector<std::vector<double*>> gss(nmatrices / 3, {nullptr, nullptr, nullptr});
	if (first && part1) for ( int ipert = 0; ipert < nmatrices; ipert++ ){
		double* gs = gss[ipert / 3][ipert % 3] = new double[ngrids]();
		const double* gd1x = gd1xs[ipert / 3][ipert % 3];
		const double* gd1y = gd1ys[ipert / 3][ipert % 3];
		const double* gd1z = gd1zs[ipert / 3][ipert % 3];
		for ( long int kgrid = 0; kgrid < ngrids; kgrid++ )
			gs[kgrid] += 2 * ( d1xs[kgrid] * gd1x[kgrid] + d1ys[kgrid] * gd1y[kgrid] + d1zs[kgrid] * gd1z[kgrid] );
	}

	EigenMatrix h = EigenZero(nmatrices, nmatrices);
	for ( int ipert = 0; ipert < nmatrices; ipert++ ){
		for ( int jpert = ipert; jpert < nmatrices; jpert++ ){
			double hij = 0;
			if (part1){
				const double* gdi = gds[ipert / 3][ipert % 3];
				const double* gdj = gds[jpert / 3][jpert % 3];
				if (zeroth){
					for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
						hij += ws[kgrid] * vrrs[kgrid] * gdi[kgrid] * gdj[kgrid];
					}
				}
				const double* gsi = gss[ipert / 3][ipert % 3];
				const double* gsj = gss[jpert / 3][jpert % 3];
				const double* gd1xi = gd1xs[ipert / 3][ipert % 3];
				const double* gd1yi = gd1ys[ipert / 3][ipert % 3];
				const double* gd1zi = gd1zs[ipert / 3][ipert % 3];
				const double* gd1xj = gd1xs[jpert / 3][jpert % 3];
				const double* gd1yj = gd1ys[jpert / 3][jpert % 3];
				const double* gd1zj = gd1zs[jpert / 3][jpert % 3];
				if (first){
					for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
						hij += ws[kgrid] * (
								vrss[kgrid] * (
									gdi[kgrid] * gsj[kgrid] +
									gdj[kgrid] * gsi[kgrid]
								) +
								vsss[kgrid] * gsi[kgrid] * gsj[kgrid] +
								2 * vss[kgrid] * (
									gd1xi[kgrid] * gd1xj[kgrid] +
									gd1yi[kgrid] * gd1yj[kgrid] +
									gd1zi[kgrid] * gd1zj[kgrid]
								)
						);
					}
				}
			}
			if (part2){
				const double* hdij = hds[ipert][jpert];
				if (zeroth){
					for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
						hij += ws[kgrid] * vrs[kgrid] * hdij[kgrid];
					}
				}
				const double* hd1xi = hd1xs[ipert][jpert];
				const double* hd1yi = hd1ys[ipert][jpert];
				const double* hd1zi = hd1zs[ipert][jpert];
				if (first){
					for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
						hij += 2 * ws[kgrid] * vss[kgrid] * (
								hd1xi[kgrid] * d1xs[kgrid] +
								hd1yi[kgrid] * d1ys[kgrid] +
								hd1zi[kgrid] * d1zs[kgrid] 
						);
					}
				}
			}
			h(ipert, jpert) = h(jpert, ipert) = hij;
		}
	}
	if (first && part1) for ( int ipert = 0; ipert < nmatrices; ipert++ )
		delete [] gss[ipert / 3][ipert % 3];
	return h;
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

#define __Allocate_and_Zero_3__(vector)\
	for ( int ipert = 0; ipert < 3*this->getNumCenters(); ipert++ )\
		for ( int jpert = 0; jpert < 3*this->getNumCenters(); jpert++ ){\
			if (vector[ipert][jpert]) std::memset(vector[ipert][jpert], 0, ngrids_this_batch * sizeof(double));\
			else vector[ipert][jpert] = new double[ngrids_this_batch]();\
		}

#define __max_num_grids__ 100000


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
	std::vector<std::vector<double*>> batch_gds(natoms, {nullptr, nullptr, nullptr});
	std::vector<std::vector<double*>> batch_gd1xs(natoms, {nullptr, nullptr, nullptr});
	std::vector<std::vector<double*>> batch_gd1ys(natoms, {nullptr, nullptr, nullptr});
	std::vector<std::vector<double*>> batch_gd1zs(natoms, {nullptr, nullptr, nullptr});

	if (this->XC.XCcode){

		EigenMatrix xcg = EigenZero(natoms, 3);

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
		double* batch_ao3xxxs = nullptr;
		double* batch_ao3xxys = nullptr;
		double* batch_ao3xxzs = nullptr;
		double* batch_ao3xyys = nullptr;
		double* batch_ao3xyzs = nullptr;
		double* batch_ao3xzzs = nullptr;
		double* batch_ao3yyys = nullptr;
		double* batch_ao3yyzs = nullptr;
		double* batch_ao3yzzs = nullptr;
		double* batch_ao3zzzs = nullptr;

		// Distributing grids to batches
		const long int ngrids_per_batch = derivative < 2 ? __max_num_grids__ : ngrids; // If only gradient is required, the grids will be done in batches. Otherwise, they will be done simultaneously, because the intermediate variable, nuclear gradient of density on grids, is required in CPSCF.
		std::vector<long int> batch_tails = {};
		long int tmp = 0;
		while ( tmp < ngrids ){
			if ( tmp + ngrids_per_batch < ngrids )
				batch_tails.push_back(tmp + ngrids_per_batch);
			else batch_tails.push_back(ngrids);
			tmp += ngrids_per_batch;
		}
		const int nbatches = batch_tails.size();
		if (output) std::printf("Calculating XC nuclear gradient in %d batches ...\n", nbatches);

		const auto start_all = __now__;
		for ( int ibatch = 0; ibatch < nbatches; ibatch++ ){
			if (output) std::printf("| Batch %d\n", ibatch);
			const long int grid_head = (ibatch == 0) ? 0 : batch_tails[ibatch - 1];
			const long int grid_tail = batch_tails[ibatch];
			const long int ngrids_this_batch = grid_tail - grid_head;
			if (output) std::printf("| | Evaluating AOs on %ld grids ...", ngrids_this_batch);
			auto start = __now__;
			int order = 0;
			std::vector<int> orders = {};
			if ( this->XC.XCfamily.compare("LDA") == 0 ){
				order = 0;
				orders = {0};
				__Allocate_and_Zero_2__(batch_gds);
			}else if ( this->XC.XCfamily.compare("GGA") == 0 ){
				order = 1;
				orders = {0, 1};
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
						batch_ao2xys, batch_ao2xzs, batch_ao2yzs,
						batch_ao3xxxs, batch_ao3xxys, batch_ao3xxzs,
						batch_ao3xyys, batch_ao3xyzs, batch_ao3xzzs,
						batch_ao3yyys, batch_ao3yyzs, batch_ao3yzzs, batch_ao3zzzs
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
			if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));
			if (output) std::printf("| | Calculating skeleton gradient of density on these grids ...");
			start = __now__;
			GetDensitySkeleton(
					orders,
					batch_aos,
					batch_ao1xs, batch_ao1ys, batch_ao1zs,
					batch_ao2xxs, batch_ao2yys, batch_ao2zzs,
					batch_ao2xys, batch_ao2xzs, batch_ao2yzs,
					ngrids_this_batch, this->getDensity(),
					bf2atom,
					batch_gds, batch_gd1xs, batch_gd1ys, batch_gd1zs
			);
			if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));
			if (output) std::printf("| | Calculating XC gradient contributed by these grids ...");
			start = __now__;
			xcg += XCGradient(
					order,
					this->Ws + grid_head, ngrids_this_batch,
					this->Rho1Xs + grid_head,
					this->Rho1Ys + grid_head,
					this->Rho1Zs + grid_head,
					this->E1Rhos + grid_head,
					this->E1Sigmas + grid_head,
					batch_gds, batch_gd1xs, batch_gd1ys, batch_gd1zs
			);
			if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));
		}
		if (output) std::printf("| Done in %f s\n", __duration__(start_all, __now__));
		this->Gradient += xcg;
	}

	if ( derivative >= 2){
		if (output) std::printf("Calculating nuclear skeleton gradient of Fock matrix ...");
		auto start = __now__;
		std::vector<EigenMatrix> Ss(3 * natoms, EigenZero(nbasis, nbasis));
		std::vector<EigenMatrix> Fskeletons(3 * natoms, EigenZero(nbasis, nbasis));
		std::vector<std::vector<EigenMatrix>> fxcskeleton1s(natoms, {EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis)});
		std::vector<std::vector<EigenMatrix>> fxcskeleton2s(natoms, {EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis)});
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
			fxcskeleton1s = GxcSkeleton(
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
			for ( int iatom = 0; iatom < natoms; iatom++ ) for ( int xyz = 0; xyz < 3; xyz++ )
				fxcskeleton2s[iatom][xyz] = PotentialSkeleton(
						orders,
						this->Ws, ngrids, nbasis,
						this->AOs,
						this->AO1Xs, this->AO1Ys, this->AO1Zs,
						this->AO2XXs, this->AO2YYs, this->AO2ZZs,
						this->AO2XYs, this->AO2XZs, this->AO2YZs,
						this->Rho1Xs, this->Rho1Ys, this->Rho1Zs,
						this->E1Rhos, this->E1Sigmas,
						this->E2Rho2s, this->E2RhoSigmas, this->E2Sigma2s,
						batch_gds[iatom][xyz],
						batch_gd1xs[iatom][xyz],
						batch_gd1ys[iatom][xyz],
						batch_gd1zs[iatom][xyz]
				);
		}

		for ( int icenter = 0; icenter < natoms; icenter++ ) for ( int xyz = 0; xyz < 3; xyz++ ){
			Ss[3 * icenter + xyz] = this->OverlapGrads[icenter][xyz];
			Fskeletons[3 * icenter + xyz] = Hgrads[icenter][xyz] + this->GGrads[icenter][xyz] + fxcskeleton1s[icenter][xyz] + fxcskeleton2s[icenter][xyz];
		}
		if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));
		
		if (output) std::printf("Coupled-perturbed self-consistent-field ...\n");
		start = __now__;
		auto [Us, dDs, dEs, dWs, dFs] = NonIdempotent( // std::vector<EigenMatrix>
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
				this->NumGrids,
				output - 1, nthreads
		);
		if (output) std::printf("| Done in %f s\n", __duration__(start, __now__));
		this->Hessian += HFHessian(this->KineticHess + this->NuclearHess - this->OverlapHess + 0.5 * this->GHess, dDs, Fskeletons, dWs, Ss);

		if ( this->Temperature > 0 ){
			if (output) std::printf("Density-based occupation-gradient coupled-perturbed self-consistent-field ...\n");
			start = __now__;
			const EigenArray ns = this->getOccupation().array()/2;
			const EigenVector Nes = (ns * ( ns - 1. ))/ this->Temperature;
			const std::vector<EigenVector> dNs = DensityOccupationGradient(
					this->getCoefficientMatrix(),
					this->getEnergy(),
					Nes,
					Ss, dFs,
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
					this->NumGrids,
					output - 1, nthreads
			);
			if (output) std::printf("| Done in %f s\n", __duration__(start, __now__));
			EigenMatrix hessog = EigenZero(3*natoms, 3*natoms);
			for ( int ipert = 0; ipert < 3 * natoms; ipert++ )
				for ( int jpert = 0; jpert < 3 * natoms; jpert++ )
					hessog(ipert, jpert) = dEs[jpert].dot(dNs[ipert]);
			this->Hessian += hessog + hessog.transpose();
		}

		if (this->XC.XCcode){
			// Distributing grids to batches
			const long int ngrids_per_batch = __max_num_grids__;
			std::vector<long int> batch_tails = {};
			long int tmp = 0;
			while ( tmp < ngrids ){
				if ( tmp + ngrids_per_batch < ngrids )
					batch_tails.push_back(tmp + ngrids_per_batch);
				else batch_tails.push_back(ngrids);
				tmp += ngrids_per_batch;
			}
			const int nbatches = batch_tails.size();

			// Nuclear hessian of density in batches
			std::vector<std::vector<double*>> batch_hds(3 * natoms);
			std::vector<std::vector<double*>> batch_hd1xs(3 * natoms);
			std::vector<std::vector<double*>> batch_hd1ys(3 * natoms);
			std::vector<std::vector<double*>> batch_hd1zs(3 * natoms);
			for ( int ipert = 0; ipert < 3 * natoms; ipert++ ){
				batch_hds[ipert].resize(3 * natoms);
				batch_hd1xs[ipert].resize(3 * natoms);
				batch_hd1ys[ipert].resize(3 * natoms);
				batch_hd1zs[ipert].resize(3 * natoms);
			}

			if (output) std::printf("Calculating XC nuclear hessian Part 1 (involving skeleton gradient of density on grids) ...");
			auto start = __now__;
			this->Hessian += HxcSkeleton(
					orders,
					ngrids, this->Ws,
					1,
					this->Rho1Xs, this->Rho1Ys, this->Rho1Zs,
					this->E2Rho2s, this->E2RhoSigmas, this->E2Sigma2s,
					batch_gds,
					batch_gd1xs, batch_gd1ys, batch_gd1zs,
					0,
					this->E1Rhos, this->E1Sigmas,
					batch_hds, // Dummy std::vector
					batch_hd1xs, batch_hd1ys, batch_hd1zs
			);
			if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));

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
			double* batch_ao3xxxs = nullptr;
			double* batch_ao3xxys = nullptr;
			double* batch_ao3xxzs = nullptr;
			double* batch_ao3xyys = nullptr;
			double* batch_ao3xyzs = nullptr;
			double* batch_ao3xzzs = nullptr;
			double* batch_ao3yyys = nullptr;
			double* batch_ao3yyzs = nullptr;
			double* batch_ao3yzzs = nullptr;
			double* batch_ao3zzzs = nullptr;

			if (output) std::printf("Calculating XC nuclear hessian Part 2 (involving skeleton hessian of density on grids) in %d batches ...\n", nbatches);
			const auto start_all = __now__;
			for ( int ibatch = 0; ibatch < nbatches; ibatch++ ){
				if (output) std::printf("| Batch %d\n", ibatch);
				const long int grid_head = (ibatch == 0) ? 0 : batch_tails[ibatch - 1];
				const long int grid_tail = batch_tails[ibatch];
				const long int ngrids_this_batch = grid_tail - grid_head;
				if (output) std::printf("| | Evaluating AOs on %ld grids ...", ngrids_this_batch);
				start = __now__;
				std::vector<int> orders = {};
				if ( this->XC.XCfamily.compare("LDA") == 0 ){
					orders = {0};
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
					__Allocate_and_Zero_3__(batch_hds);
				}else if ( this->XC.XCfamily.compare("GGA") == 0 ){
					orders = {0, 1};
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
					__Allocate_and_Zero__(batch_ao3xxxs);
					__Allocate_and_Zero__(batch_ao3xxys);
					__Allocate_and_Zero__(batch_ao3xxzs);
					__Allocate_and_Zero__(batch_ao3xyys);
					__Allocate_and_Zero__(batch_ao3xyzs);
					__Allocate_and_Zero__(batch_ao3xzzs);
					__Allocate_and_Zero__(batch_ao3yyys);
					__Allocate_and_Zero__(batch_ao3yyzs);
					__Allocate_and_Zero__(batch_ao3yzzs);
					__Allocate_and_Zero__(batch_ao3zzzs);
					__Allocate_and_Zero_3__(batch_hds);
					__Allocate_and_Zero_3__(batch_hd1xs);
					__Allocate_and_Zero_3__(batch_hd1ys);
					__Allocate_and_Zero_3__(batch_hd1zs);
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
						batch_ao2xys, batch_ao2xzs, batch_ao2yzs,
						batch_ao3xxxs, batch_ao3xxys, batch_ao3xxzs,
						batch_ao3xyys, batch_ao3xyzs, batch_ao3xzzs,
						batch_ao3yyys, batch_ao3yyzs, batch_ao3yzzs, batch_ao3zzzs
				);
				if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));
				if (output) std::printf("| | Calculating skeleton hessian of density on these grids ...");
				start = __now__;
				GetDensitySkeleton2(
						orders,
						batch_aos,
						batch_ao1xs, batch_ao1ys, batch_ao1zs,
						batch_ao2xxs, batch_ao2yys, batch_ao2zzs,
						batch_ao2xys, batch_ao2xzs, batch_ao2yzs,
						batch_ao3xxxs, batch_ao3xxys, batch_ao3xxzs,
						batch_ao3xyys, batch_ao3xyzs, batch_ao3xzzs,
						batch_ao3yyys, batch_ao3yyzs, batch_ao3yzzs, batch_ao3zzzs,
						ngrids_this_batch, this->getDensity(),
						bf2atom,
						batch_hds,
						batch_hd1xs, batch_hd1ys, batch_hd1zs
				);

				if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));
				if (output) std::printf("| | Calculating XC skeleton hessian contributed by these grids ...");
				start = __now__;
				this->Hessian += HxcSkeleton(
						orders,
						ngrids_this_batch, this->Ws + grid_head,
						0,
						this->Rho1Xs + grid_head,
						this->Rho1Ys + grid_head,
						this->Rho1Zs + grid_head,
						this->E2Rho2s, this->E2RhoSigmas, this->E2Sigma2s,
						batch_gds, // Dummy
						batch_gd1xs, batch_gd1ys, batch_gd1zs,
						1,
						this->E1Rhos + grid_head,
						this->E1Sigmas + grid_head,
						batch_hds,
						batch_hd1xs, batch_hd1ys, batch_hd1zs
				);
				if (output) std::printf(" Done in %f s\n", __duration__(start, __now__));
			}
			if (output) std::printf("| Done in %f s\n", __duration__(start_all, __now__));
		}
	}
}


