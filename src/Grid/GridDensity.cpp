#include <Eigen/Core>
#include <cmath>
#include <vector>
#include <set>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cassert>

#include "../Macro.h"
#include "../Multiwfn.h"

#include <iostream>

void GetDensity(
		std::vector<int> orders,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		long int ngrids, EigenMatrix D,
		double* ds,
		double* d1xs, double* d1ys, double* d1zs,
		double* d2s, double* ts){

	bool zeroth = 0;
	bool first = 0;
	bool second = 0;
	if (std::find(orders.begin(), orders.end(), 0) != orders.end()){
		zeroth = 1;
		assert(aos && "AOs on grids do not exist!");
		assert(ds && "Density on grids array is not allocated!");
	}
	if (std::find(orders.begin(), orders.end(), 1) != orders.end()){
		first = 1;
		assert(aos && "AOs on grids do not exist!");
		assert(ao1xs && "First order x-derivatives of AOs on grids do not exist!");
		assert(ao1ys && "First order y-derivatives of AOs on grids do not exist!");
		assert(ao1zs && "First order z-derivatives of AOs on grids do not exist!");
		assert(d1xs && "First order x-derivatives of density on grids array is not allocated!");
		assert(d1ys && "First order y-derivatives of density on grids array is not allocated!");
		assert(d1zs && "First order z-derivatives of density on grids array is not allocated!");
	}
	if (std::find(orders.begin(), orders.end(), 2) != orders.end()){
		second = 1;
		assert(aos && "AOs on grids do not exist!");
		assert(ao1xs && "First order x-derivatives of AOs on grids do not exist!");
		assert(ao1ys && "First order y-derivatives of AOs on grids do not exist!");
		assert(ao1zs && "First order z-derivatives of AOs on grids do not exist!");
		assert(ao2ls && "Laplacians of AOs on grids do not exist!");
		assert(d2s && "Laplacians of density on grids array is not allocated!");
		assert(ts && "Taus of density on grids array is not allocated!");
	}

	double* iao = aos;
	double* jao = aos;
	double* ix = ao1xs;
	double* jx = ao1xs;
	double* iy = ao1ys;
	double* jy = ao1ys;
	double* iz = ao1zs;
	double* jz = ao1zs;
	double* iao2 = ao2ls;
	double* jao2 = ao2ls;
	double Dij;
	for ( int ibasis = 0; ibasis < D.cols(); ibasis++ ){
		Dij = D(ibasis, ibasis);
		if (zeroth) iao = aos + ibasis * ngrids; // ibasis*ngrids+jgrid
		if (first){
			ix = ao1xs + ibasis * ngrids;
			iy = ao1ys + ibasis * ngrids;
			iz = ao1zs + ibasis * ngrids;
		}
		if (second) iao2 = ao2ls + ibasis * ngrids;
		for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
			if (zeroth) ds[kgrid] += Dij * iao[kgrid] * iao[kgrid];
			if (first){
				d1xs[kgrid] += 2 * Dij * ix[kgrid] * iao[kgrid];
				d1ys[kgrid] += 2 * Dij * iy[kgrid] * iao[kgrid];
				d1zs[kgrid] += 2 * Dij * iz[kgrid] * iao[kgrid];
			}
			if (second){
				ts[kgrid] += 0.5 * Dij * ( ix[kgrid] * ix[kgrid] + iy[kgrid] * iy[kgrid] + iz[kgrid] * iz[kgrid] );
				d2s[kgrid] += 2 * Dij * iao[kgrid] * iao2[kgrid];
			}
		}
		for ( int jbasis = 0; jbasis < ibasis; jbasis++ ){
			Dij = D(ibasis, jbasis);
			if (zeroth) jao = aos + jbasis * ngrids;
			if (first){
				jx = ao1xs + jbasis * ngrids;
				jy = ao1ys + jbasis * ngrids;
				jz = ao1zs + jbasis * ngrids;
			}
			if (second) jao2 = ao2ls + jbasis * ngrids;
			for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
				if (zeroth) ds[kgrid] += 2 * Dij * iao[kgrid] * jao[kgrid];
				if (first){
					d1xs[kgrid] += 2 * Dij * ( ix[kgrid] * jao[kgrid] + iao[kgrid] * jx[kgrid] );
					d1ys[kgrid] += 2 * Dij * ( iy[kgrid] * jao[kgrid] + iao[kgrid] * jy[kgrid]);
					d1zs[kgrid] += 2 * Dij * ( iz[kgrid] * jao[kgrid] + iao[kgrid] * jz[kgrid]);
				}
				if (second){
					ts[kgrid] += Dij * ( ix[kgrid] * jx[kgrid] + iy[kgrid] * jy[kgrid] + iz[kgrid] * jz[kgrid] );
					d2s[kgrid] += 2 * Dij * ( iao[kgrid] * jao2[kgrid] + iao2[kgrid] * jao[kgrid] );
				}
			}
		}
	}
	if (second) for ( long int igrid = 0; igrid < ngrids; igrid++ )
		d2s[igrid] += 4 * ts[igrid];
}

#define __Check_Vector_Array__(vec)\
	vec.size() > 0 && vec[0].size() == 3 && vec[0][0]

void GetDensitySkeleton(
		std::vector<int> orders,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2xxs, double* ao2yys, double* ao2zzs,
		double* ao2xys, double* ao2xzs, double* ao2yzs,
		long int ngrids, EigenMatrix D,
		std::vector<int>& bf2atom,
		std::vector<std::vector<double*>>& ds,
		std::vector<std::vector<double*>>& d1xs,
		std::vector<std::vector<double*>>& d1ys,
		std::vector<std::vector<double*>>& d1zs){

	bool zeroth = 0;
	bool first = 0;
	if (std::find(orders.begin(), orders.end(), 0) != orders.end()){
		zeroth = 1;
		assert(aos && "AOs on grids do not exist!");
		assert(ao1xs && "First-order x-derivatives of AOs on grids do not exist!");
		assert(ao1ys && "First-order y-derivatives of AOs on grids do not exist!");
		assert(ao1zs && "First-order z-derivatives of AOs on grids do not exist!");
		assert(__Check_Vector_Array__(ds) && "Nuclear gradient of density on grids arrays are not allocated!");
	}
	if (std::find(orders.begin(), orders.end(), 1) != orders.end()){
		first = 1;
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
		assert(__Check_Vector_Array__(d1xs) && "Nuclear Gradient of first-order x-derivatives of density on grids array is not allocated!");
		assert(__Check_Vector_Array__(d1ys) && "Nuclear Gradient of first-order y-derivatives of density on grids array is not allocated!");
		assert(__Check_Vector_Array__(d1zs) && "Nuclear Gradient of first-order z-derivatives of density on grids array is not allocated!");
	}

	//double* iao = aos; // AOs
	double* jao = aos;
	double* ix = ao1xs; // AO first derivatives
	double* jx = ao1xs;
	double* iy = ao1ys;
	double* jy = ao1ys;
	double* iz = ao1zs;
	double* jz = ao1zs;
	double* ixx = ao2xxs; // AO second derivatives
	//double* jxx = ao2xxs;
	double* iyy = ao2yys;
	//double* jyy = ao2yys;
	double* izz = ao2zzs;
	//double* jzz = ao2zzs;
	double* ixy = ao2xys;
	//double* jxy = ao2xys;
	double* ixz = ao2xzs;
	//double* jxz = ao2xzs;
	double* iyz = ao2yzs;
	//double* jyz = ao2yzs;
	int iatom = -1;
	int jatom = -1;
	double twoDij = 0;
	for ( int ibasis = 0; ibasis < D.cols(); ibasis++ ){
		//if (aos) iao = aos + ibasis * ngrids; // ibasis*ngrids+jgrid
		if (ao1xs && ao1ys && ao1zs){
			ix = ao1xs + ibasis * ngrids;
			iy = ao1ys + ibasis * ngrids;
			iz = ao1zs + ibasis * ngrids;
		}
		if (ao2xxs && ao2yys && ao2zzs && ao2xys && ao2xzs && ao2yzs){
			ixx = ao2xxs + ibasis * ngrids;
			iyy = ao2yys + ibasis * ngrids;
			izz = ao2zzs + ibasis * ngrids;
			ixy = ao2xys + ibasis * ngrids;
			ixz = ao2xzs + ibasis * ngrids;
			iyz = ao2yzs + ibasis * ngrids;
		}
		iatom = bf2atom[ibasis];
		for ( int jbasis = 0; jbasis < D.cols(); jbasis++ ){
			twoDij = 2 * D(ibasis, jbasis);
			if (aos) jao = aos + jbasis * ngrids;
			if (ao1xs && ao1ys && ao1zs){
				jx = ao1xs + jbasis * ngrids;
				jy = ao1ys + jbasis * ngrids;
				jz = ao1zs + jbasis * ngrids;
			}
			/*if (ao2xxs){
				jxx=ao2xxs+ibasis*ngrids;
				jyy=ao2yys+ibasis*ngrids;
				jzz=ao2zzs+ibasis*ngrids;
				jxy=ao2xys+ibasis*ngrids;
				jxz=ao2xzs+ibasis*ngrids;
				jyz=ao2yzs+ibasis*ngrids;
			}*/
			jatom = bf2atom[jbasis];
			for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
				if (zeroth){
					ds[iatom][0][kgrid] -= twoDij * ix[kgrid] * jao[kgrid];
					ds[iatom][1][kgrid] -= twoDij * iy[kgrid] * jao[kgrid];
					ds[iatom][2][kgrid] -= twoDij * iz[kgrid] * jao[kgrid];
				}
				if (first){
					d1xs[iatom][0][kgrid] -= twoDij * ixx[kgrid] * jao[kgrid];
					d1xs[jatom][0][kgrid] -= twoDij * ix[kgrid] * jx[kgrid];

					d1xs[iatom][1][kgrid] -= twoDij * ixy[kgrid] * jao[kgrid];
					d1xs[jatom][1][kgrid] -= twoDij * ix[kgrid] * jy[kgrid];

					d1xs[iatom][2][kgrid] -= twoDij * ixz[kgrid] * jao[kgrid];
					d1xs[jatom][2][kgrid] -= twoDij * ix[kgrid] * jz[kgrid];

					d1ys[iatom][0][kgrid] -= twoDij * ixy[kgrid] * jao[kgrid];
					d1ys[jatom][0][kgrid] -= twoDij * iy[kgrid] * jx[kgrid];

					d1ys[iatom][1][kgrid] -= twoDij * iyy[kgrid] * jao[kgrid];
					d1ys[jatom][1][kgrid] -= twoDij * iy[kgrid] * jy[kgrid];

					d1ys[iatom][2][kgrid] -= twoDij * iyz[kgrid] * jao[kgrid];
					d1ys[jatom][2][kgrid] -= twoDij * iy[kgrid] * jz[kgrid];

					d1zs[iatom][0][kgrid] -= twoDij * ixz[kgrid] * jao[kgrid];
					d1zs[jatom][0][kgrid] -= twoDij * iz[kgrid] * jx[kgrid];

					d1zs[iatom][1][kgrid] -= twoDij * iyz[kgrid] * jao[kgrid];
					d1zs[jatom][1][kgrid] -= twoDij * iz[kgrid] * jy[kgrid];

					d1zs[iatom][2][kgrid] -= twoDij * izz[kgrid] * jao[kgrid];
					d1zs[jatom][2][kgrid] -= twoDij * iz[kgrid] * jz[kgrid];
				}
			}
		}
	}
}


