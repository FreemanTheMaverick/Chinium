#include <Eigen/Core>
#include <cmath>
#include <vector>
#include <set>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cassert>
#include <omp.h>

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
		double* d2s, double* ts,
		int nthreads){

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

	long int ngrids0 = ngrids; // Dummy arrays to prevent the main body from being damaged by parallism.
	long int ngrids1 = ngrids;
	long int ngrids2 = ngrids;
	if (!zeroth){
		ngrids0 = 1;
		ds = new double();
	}
	if (!first){
		ngrids1 = 1;
		d1xs = new double();
		d1ys = new double();
		d1zs = new double();
	}
	if (!second){
		ngrids2 = 1;
		ts = new double();
		d2s = new double();
	}

	std::vector<int> basis_sequence(D.cols());
	for ( int i = 0; i <= ( D.cols() - 1 ) / 2 ; i++ ){
		basis_sequence[2 * i] = D.cols() - i - 1;
		if ( 2 * i + 1 < D.cols() ) basis_sequence[2 * i + 1] = i;
	}

	std::vector<double> Dij_iao;
	std::vector<double> Dij_jao;
	if (zeroth || first || second){
		Dij_iao.resize(ngrids);
		Dij_jao.resize(ngrids);
	}

	#pragma omp parallel for schedule(dynamic,2)\
	reduction(+:\
			ds[:ngrids0],\
			d1xs[:ngrids1], d1ys[:ngrids1], d1zs[:ngrids1],\
			ts[:ngrids2], d2s[:ngrids2]\
	)\
	num_threads(nthreads)\
	firstprivate(Dij_iao, Dij_jao)
	for ( int i = 0; i < D.cols(); i++ ){
		int ibasis = basis_sequence[i];
		double Dij = D(ibasis, ibasis);
		double *iao{}, *ix{}, *iy{}, *iz{}, *iao2{};
		if (zeroth) iao = aos + ibasis * ngrids; // ibasis*ngrids+jgrid
		if (first){
			ix = ao1xs + ibasis * ngrids;
			iy = ao1ys + ibasis * ngrids;
			iz = ao1zs + ibasis * ngrids;
		}
		if (second) iao2 = ao2ls + ibasis * ngrids;
		if (zeroth || first || second) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ )
			Dij_iao[kgrid] = Dij * iao[kgrid];
		if (zeroth) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ )
			ds[kgrid] += Dij_iao[kgrid] * iao[kgrid];
		if (first) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
			d1xs[kgrid] += 2 * ix[kgrid] * Dij_iao[kgrid];
			d1ys[kgrid] += 2 * iy[kgrid] * Dij_iao[kgrid];
			d1zs[kgrid] += 2 * iz[kgrid] * Dij_iao[kgrid];
		}
		if (second) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
			ts[kgrid] += 0.5 * Dij * ( ix[kgrid] * ix[kgrid] + iy[kgrid] * iy[kgrid] + iz[kgrid] * iz[kgrid] );
			d2s[kgrid] += 2 * Dij_iao[kgrid] * iao2[kgrid];
		}
		for ( int jbasis = ibasis - 1; jbasis >=0; jbasis-- ){
			Dij = D(ibasis, jbasis);
			double *jao{}, *jx{}, *jy{}, *jz{}, *jao2{};
			if (zeroth) jao = aos + jbasis * ngrids;
			if (first){
				jx = ao1xs + jbasis * ngrids;
				jy = ao1ys + jbasis * ngrids;
				jz = ao1zs + jbasis * ngrids;
			}
			if (second) jao2 = ao2ls + jbasis * ngrids;
			if (zeroth || first || second) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
				Dij_iao[kgrid] = 2 * Dij * iao[kgrid];
				Dij_jao[kgrid] = 2 * Dij * jao[kgrid];
			}
			if (zeroth) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ )
				ds[kgrid] += Dij_iao[kgrid] * jao[kgrid];
			if (first) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
				d1xs[kgrid] += ix[kgrid] * Dij_jao[kgrid] + Dij_iao[kgrid] * jx[kgrid];
				d1ys[kgrid] += iy[kgrid] * Dij_jao[kgrid] + Dij_iao[kgrid] * jy[kgrid];
				d1zs[kgrid] += iz[kgrid] * Dij_jao[kgrid] + Dij_iao[kgrid] * jz[kgrid];
			}
			if (second) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
				ts[kgrid] += Dij * ( ix[kgrid] * jx[kgrid] + iy[kgrid] * jy[kgrid] + iz[kgrid] * jz[kgrid] );
				d2s[kgrid] += Dij_iao[kgrid] * jao2[kgrid] + iao2[kgrid] * Dij_jao[kgrid];
			}
		}
	}
	if (second) for ( long int igrid = 0; igrid < ngrids; igrid++ )
		d2s[igrid] += 4 * ts[igrid];

	if (!zeroth) delete ds; // Cleaning the dummy arrays.
	if (!first){
		delete d1xs;
		delete d1ys;
		delete d1zs;
	}
	if (!second){
		delete ts;
		delete d2s;
	}
}

/*
void GetDensityMultiple(
		std::vector<int> orders,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		long int ngrids, std::vector<EigenMatrix>& Ds,
		EigenArray* ds,
		EigenArray* d1xs,
		EigenArray* d1ys,
		EigenArray* d1zs,
		EigenArray* d2s,
		EigenArray* ts){

	const int nbasis = Ds[0].rows();
	const int nmatrices = Ds.size();
	const int nmatrices_redun = std::ceil((double)nmatrices / 8.) * 8;
	std::vector<std::vector<EigenArray>> vvvD(nbasis);
	for ( int i = 0; i < nbasis; i++ ){
		vvvD[i].resize(nbasis);
		for ( int j = 0; j < nbasis; j++ ){
			vvvD[i][j] = EigenZero(nmatrices_redun, 1).array();
			for ( int k = 0; k < nmatrices; k++ )
				vvvD[i][j](k) = Ds[k](i, j);
		}
	}

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
	EigenArray Dij = EigenZero(nmatrices_redun, 1).array();
	for ( int ibasis = 0; ibasis < D.cols(); ibasis++ ){
		EigenArray Dij = vvvD[ibasis][ibasis];
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
		}
		for ( int jbasis = 0; jbasis < ibasis; jbasis++ ){
			Dij = vvvD[ibasis][jbasis];
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
}*/

#define __Check_Vector_Array__(vec)\
	vec.size() > 0 && vec[0].size() == 3 && vec[0][0]

#define iyx ixy
#define izx ixz
#define izy iyz

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

					d1ys[iatom][0][kgrid] -= twoDij * iyx[kgrid] * jao[kgrid];
					d1ys[jatom][0][kgrid] -= twoDij * iy[kgrid] * jx[kgrid];

					d1ys[iatom][1][kgrid] -= twoDij * iyy[kgrid] * jao[kgrid];
					d1ys[jatom][1][kgrid] -= twoDij * iy[kgrid] * jy[kgrid];

					d1ys[iatom][2][kgrid] -= twoDij * iyz[kgrid] * jao[kgrid];
					d1ys[jatom][2][kgrid] -= twoDij * iy[kgrid] * jz[kgrid];

					d1zs[iatom][0][kgrid] -= twoDij * izx[kgrid] * jao[kgrid];
					d1zs[jatom][0][kgrid] -= twoDij * iz[kgrid] * jx[kgrid];

					d1zs[iatom][1][kgrid] -= twoDij * izy[kgrid] * jao[kgrid];
					d1zs[jatom][1][kgrid] -= twoDij * iz[kgrid] * jy[kgrid];

					d1zs[iatom][2][kgrid] -= twoDij * izz[kgrid] * jao[kgrid];
					d1zs[jatom][2][kgrid] -= twoDij * iz[kgrid] * jz[kgrid];
				}
			}
		}
	}
}

#define __Check_Vector_Array_2__(vec)\
	vec.size() && vec[0].size() && vec[0][0]

#define ixyx ixxy
#define iyxx ixxy

#define ixzx ixxz
#define izxx ixxz

#define iyxy ixyy
#define iyyx ixyy

#define izxy ixyz
#define iyzx ixyz
#define iyxz ixyz
#define izyx ixyz
#define ixzy ixyz

#define izxz ixzz
#define izzx ixzz

#define iyzy iyyz
#define izyy iyyz

#define izyz iyzz
#define izzy iyzz

#define iyx ixy
#define izx ixz
#define izy iyz

#define jyx jxy
#define jzx jxz
#define jzy jyz

void GetDensitySkeleton2(
		std::vector<int> orders,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2xxs, double* ao2yys, double* ao2zzs,
		double* ao2xys, double* ao2xzs, double* ao2yzs,
		double* ao3xxxs, double* ao3xxys, double* ao3xxzs,
		double* ao3xyys, double* ao3xyzs, double* ao3xzzs,
		double* ao3yyys, double* ao3yyzs, double* ao3yzzs, double* ao3zzzs,
		long int ngrids, EigenMatrix D,
		std::vector<int>& bf2atom,
		std::vector<std::vector<double*>>& hds,
		std::vector<std::vector<double*>>& hd1xs,
		std::vector<std::vector<double*>>& hd1ys,
		std::vector<std::vector<double*>>& hd1zs){

	bool zeroth = 0;
	bool first = 0;
	if (std::find(orders.begin(), orders.end(), 0) != orders.end()){
		zeroth = 1;
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
		assert(__Check_Vector_Array_2__(hds) && "Nuclear hessian of density on grids arrays are not allocated!");
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
		assert(ao3xxxs && "Third-order xxx-derivatives of AOs on grids do not exist!");
		assert(ao3xxys && "Third-order xxy-derivatives of AOs on grids do not exist!");
		assert(ao3xxzs && "Third-order xxz-derivatives of AOs on grids do not exist!");
		assert(ao3xyys && "Third-order xyy-derivatives of AOs on grids do not exist!");
		assert(ao3xyzs && "Third-order xyz-derivatives of AOs on grids do not exist!");
		assert(ao3xzzs && "Third-order xzz-derivatives of AOs on grids do not exist!");
		assert(ao3yyys && "Third-order yyy-derivatives of AOs on grids do not exist!");
		assert(ao3yyzs && "Third-order yyz-derivatives of AOs on grids do not exist!");
		assert(ao3yzzs && "Third-order yzz-derivatives of AOs on grids do not exist!");
		assert(ao3zzzs && "Third-order zzz-derivatives of AOs on grids do not exist!");
		assert(__Check_Vector_Array_2__(hd1xs) && "Nuclear hessian of first-order x-derivatives of density on grids arrays are not allocated!");
		assert(__Check_Vector_Array_2__(hd1ys) && "Nuclear hessian of first-order y-derivatives of density on grids arrays are not allocated!");
		assert(__Check_Vector_Array_2__(hd1zs) && "Nuclear hessian of first-order z-derivatives of density on grids arrays are not allocated!");
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
	double* jxx = ao2xxs;
	double* iyy = ao2yys;
	double* jyy = ao2yys;
	double* izz = ao2zzs;
	double* jzz = ao2zzs;
	double* ixy = ao2xys;
	double* jxy = ao2xys;
	double* ixz = ao2xzs;
	double* jxz = ao2xzs;
	double* iyz = ao2yzs;
	double* jyz = ao2yzs;
	double* ixxx = ao3xxxs; // AO third derivatives
	double* ixxy = ao3xxys;
	double* ixxz = ao3xxzs;
	double* ixyy = ao3xyys;
	double* ixyz = ao3xyzs;
	double* ixzz = ao3xzzs;
	double* iyyy = ao3yyys;
	double* iyyz = ao3yyzs;
	double* iyzz = ao3yzzs;
	double* izzz = ao3zzzs;
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
		if (ao3xxxs && ao3xxys && ao3xxzs && ao3xyys && ao3xyzs && ao3xzzs && ao3yyys && ao3yyzs && ao3yzzs && ao3zzzs){
			ixxx = ao3xxxs + ibasis * ngrids;
			ixxy = ao3xxys + ibasis * ngrids;
			ixxz = ao3xxzs + ibasis * ngrids;
			ixyy = ao3xyys + ibasis * ngrids;
			ixyz = ao3xyzs + ibasis * ngrids;
			ixzz = ao3xzzs + ibasis * ngrids;
			iyyy = ao3yyys + ibasis * ngrids;
			iyyz = ao3yyzs + ibasis * ngrids;
			iyzz = ao3yzzs + ibasis * ngrids;
			izzz = ao3zzzs + ibasis * ngrids;
		}
		const int iatom = bf2atom[ibasis];
		for ( int jbasis = 0; jbasis < D.cols(); jbasis++ ){
			if (aos) jao = aos + jbasis * ngrids;
			if (ao1xs && ao1ys && ao1zs){
				jx = ao1xs + jbasis * ngrids;
				jy = ao1ys + jbasis * ngrids;
				jz = ao1zs + jbasis * ngrids;
			}
			if (ao2xxs && ao2yys && ao2zzs && ao2xys && ao2xzs && ao2yzs){
				jxx = ao2xxs + jbasis * ngrids;
				jyy = ao2yys + jbasis * ngrids;
				jzz = ao2zzs + jbasis * ngrids;
				jxy = ao2xys + jbasis * ngrids;
				jxz = ao2xzs + jbasis * ngrids;
				jyz = ao2yzs + jbasis * ngrids;
			}
			const double twoDij = 2 * D(ibasis, jbasis);
			double tmp, tmpx, tmpy, tmpz;
			if (zeroth) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
				tmp = twoDij * jao[kgrid];
				hds[3 * iatom + 0][3 * iatom + 0][kgrid] += tmp * ixx[kgrid];
				hds[3 * iatom + 0][3 * iatom + 1][kgrid] += tmp * ixy[kgrid];
				hds[3 * iatom + 0][3 * iatom + 2][kgrid] += tmp * ixz[kgrid];
				hds[3 * iatom + 1][3 * iatom + 1][kgrid] += tmp * iyy[kgrid];
				hds[3 * iatom + 1][3 * iatom + 2][kgrid] += tmp * iyz[kgrid];
				hds[3 * iatom + 2][3 * iatom + 2][kgrid] += tmp * izz[kgrid];
			}
			if (first) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
				tmp = twoDij * jao[kgrid];
				tmpx = twoDij * jx[kgrid];
				hd1xs[3 * iatom + 0][3 * iatom + 0][kgrid] += tmp * ixxx[kgrid] + ixx[kgrid] * tmpx;
				hd1xs[3 * iatom + 0][3 * iatom + 1][kgrid] += tmp * ixyx[kgrid] + ixy[kgrid] * tmpx;
				hd1xs[3 * iatom + 0][3 * iatom + 2][kgrid] += tmp * ixzx[kgrid] + ixz[kgrid] * tmpx;
				hd1xs[3 * iatom + 1][3 * iatom + 1][kgrid] += tmp * iyyx[kgrid] + iyy[kgrid] * tmpx;
				hd1xs[3 * iatom + 1][3 * iatom + 2][kgrid] += tmp * iyzx[kgrid] + iyz[kgrid] * tmpx;
				hd1xs[3 * iatom + 2][3 * iatom + 2][kgrid] += tmp * izzx[kgrid] + izz[kgrid] * tmpx;
				tmpy = twoDij * jy[kgrid];
				hd1ys[3 * iatom + 0][3 * iatom + 0][kgrid] += tmp * ixxy[kgrid] + ixx[kgrid] * tmpy;
				hd1ys[3 * iatom + 0][3 * iatom + 1][kgrid] += tmp * ixyy[kgrid] + ixy[kgrid] * tmpy;
				hd1ys[3 * iatom + 0][3 * iatom + 2][kgrid] += tmp * ixzy[kgrid] + ixz[kgrid] * tmpy;
				hd1ys[3 * iatom + 1][3 * iatom + 1][kgrid] += tmp * iyyy[kgrid] + iyy[kgrid] * tmpy;
				hd1ys[3 * iatom + 1][3 * iatom + 2][kgrid] += tmp * iyzy[kgrid] + iyz[kgrid] * tmpy;
				hd1ys[3 * iatom + 2][3 * iatom + 2][kgrid] += tmp * izzy[kgrid] + izz[kgrid] * tmpy;
				tmpz = twoDij * jz[kgrid];
				hd1zs[3 * iatom + 0][3 * iatom + 0][kgrid] += tmp * ixxz[kgrid] + ixx[kgrid] * tmpz;
				hd1zs[3 * iatom + 0][3 * iatom + 1][kgrid] += tmp * ixyz[kgrid] + ixy[kgrid] * tmpz;
				hd1zs[3 * iatom + 0][3 * iatom + 2][kgrid] += tmp * ixzz[kgrid] + ixz[kgrid] * tmpz;
				hd1zs[3 * iatom + 1][3 * iatom + 1][kgrid] += tmp * iyyz[kgrid] + iyy[kgrid] * tmpz;
				hd1zs[3 * iatom + 1][3 * iatom + 2][kgrid] += tmp * iyzz[kgrid] + iyz[kgrid] * tmpz;
				hd1zs[3 * iatom + 2][3 * iatom + 2][kgrid] += tmp * izzz[kgrid] + izz[kgrid] * tmpz;
			}
			const int jatom = bf2atom[jbasis];
			if ( iatom > jatom ) continue;
			else if ( iatom == jatom ){
				if (zeroth) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
					hds[3 * iatom + 0][3 * jatom + 0][kgrid] += twoDij * ix[kgrid] * jx[kgrid]; // x x
					hds[3 * iatom + 0][3 * jatom + 1][kgrid] += twoDij * ix[kgrid] * jy[kgrid]; // x y
					hds[3 * iatom + 0][3 * jatom + 2][kgrid] += twoDij * ix[kgrid] * jz[kgrid]; // x z
					hds[3 * iatom + 1][3 * jatom + 1][kgrid] += twoDij * iy[kgrid] * jy[kgrid]; // y y
					hds[3 * iatom + 1][3 * jatom + 2][kgrid] += twoDij * iy[kgrid] * jz[kgrid]; // y z
					hds[3 * iatom + 2][3 * jatom + 2][kgrid] += twoDij * iz[kgrid] * jz[kgrid]; // z z
				}
				if (first) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
					hd1xs[3 * iatom + 0][3 * jatom + 0][kgrid] += twoDij * ( ixx[kgrid] * jx[kgrid] + ix[kgrid] * jxx[kgrid] ); // t s r = x x x
					hd1xs[3 * iatom + 0][3 * jatom + 1][kgrid] += twoDij * ( ixx[kgrid] * jy[kgrid] + ix[kgrid] * jyx[kgrid] ); // t s r = x y x
					hd1xs[3 * iatom + 0][3 * jatom + 2][kgrid] += twoDij * ( ixx[kgrid] * jz[kgrid] + ix[kgrid] * jzx[kgrid] ); // t s r = x z x
					hd1xs[3 * iatom + 1][3 * jatom + 1][kgrid] += twoDij * ( iyx[kgrid] * jy[kgrid] + iy[kgrid] * jyx[kgrid] ); // t s r = y y x
					hd1xs[3 * iatom + 1][3 * jatom + 2][kgrid] += twoDij * ( iyx[kgrid] * jz[kgrid] + iy[kgrid] * jzx[kgrid] ); // t s r = y z x
					hd1xs[3 * iatom + 2][3 * jatom + 2][kgrid] += twoDij * ( izx[kgrid] * jz[kgrid] + iz[kgrid] * jzx[kgrid] ); // t s r = z z x
					hd1ys[3 * iatom + 0][3 * jatom + 0][kgrid] += twoDij * ( ixy[kgrid] * jx[kgrid] + ix[kgrid] * jxy[kgrid] ); // t s r = x x y
					hd1ys[3 * iatom + 0][3 * jatom + 1][kgrid] += twoDij * ( ixy[kgrid] * jy[kgrid] + ix[kgrid] * jyy[kgrid] ); // t s r = x y y
					hd1ys[3 * iatom + 0][3 * jatom + 2][kgrid] += twoDij * ( ixy[kgrid] * jz[kgrid] + ix[kgrid] * jzy[kgrid] ); // t s r = x z y
					hd1ys[3 * iatom + 1][3 * jatom + 1][kgrid] += twoDij * ( iyy[kgrid] * jy[kgrid] + iy[kgrid] * jyy[kgrid] ); // t s r = y y y
					hd1ys[3 * iatom + 1][3 * jatom + 2][kgrid] += twoDij * ( iyy[kgrid] * jz[kgrid] + iy[kgrid] * jzy[kgrid] ); // t s r = y z y
					hd1ys[3 * iatom + 2][3 * jatom + 2][kgrid] += twoDij * ( izy[kgrid] * jz[kgrid] + iz[kgrid] * jzy[kgrid] ); // t s r = z z y
					hd1zs[3 * iatom + 0][3 * jatom + 0][kgrid] += twoDij * ( ixz[kgrid] * jx[kgrid] + ix[kgrid] * jxz[kgrid] ); // t s r = x x z
					hd1zs[3 * iatom + 0][3 * jatom + 1][kgrid] += twoDij * ( ixz[kgrid] * jy[kgrid] + ix[kgrid] * jyz[kgrid] ); // t s r = x y z
					hd1zs[3 * iatom + 0][3 * jatom + 2][kgrid] += twoDij * ( ixz[kgrid] * jz[kgrid] + ix[kgrid] * jzz[kgrid] ); // t s r = x z z
					hd1zs[3 * iatom + 1][3 * jatom + 1][kgrid] += twoDij * ( iyz[kgrid] * jy[kgrid] + iy[kgrid] * jyz[kgrid] ); // t s r = y y z
					hd1zs[3 * iatom + 1][3 * jatom + 2][kgrid] += twoDij * ( iyz[kgrid] * jz[kgrid] + iy[kgrid] * jzz[kgrid] ); // t s r = y z z
					hd1zs[3 * iatom + 2][3 * jatom + 2][kgrid] += twoDij * ( izz[kgrid] * jz[kgrid] + iz[kgrid] * jzz[kgrid] ); // t s r = z z z
				}
			}else{
				if (zeroth) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
					hds[3 * iatom + 0][3 * jatom + 0][kgrid] += twoDij * ix[kgrid] * jx[kgrid]; // x x
					hds[3 * iatom + 0][3 * jatom + 1][kgrid] += twoDij * ix[kgrid] * jy[kgrid]; // x y
					hds[3 * iatom + 0][3 * jatom + 2][kgrid] += twoDij * ix[kgrid] * jz[kgrid]; // x z
					hds[3 * iatom + 1][3 * jatom + 0][kgrid] += twoDij * iy[kgrid] * jx[kgrid]; // y x
					hds[3 * iatom + 1][3 * jatom + 1][kgrid] += twoDij * iy[kgrid] * jy[kgrid]; // y y
					hds[3 * iatom + 1][3 * jatom + 2][kgrid] += twoDij * iy[kgrid] * jz[kgrid]; // y z
					hds[3 * iatom + 2][3 * jatom + 0][kgrid] += twoDij * iz[kgrid] * jx[kgrid]; // z x
					hds[3 * iatom + 2][3 * jatom + 1][kgrid] += twoDij * iz[kgrid] * jy[kgrid]; // z x
					hds[3 * iatom + 2][3 * jatom + 2][kgrid] += twoDij * iz[kgrid] * jz[kgrid]; // z z
				}
				if (first) for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
					hd1xs[3 * iatom + 0][3 * jatom + 0][kgrid] += twoDij * ( ixx[kgrid] * jx[kgrid] + ix[kgrid] * jxx[kgrid] ); // t s r = x x x
					hd1xs[3 * iatom + 0][3 * jatom + 1][kgrid] += twoDij * ( ixx[kgrid] * jy[kgrid] + ix[kgrid] * jyx[kgrid] ); // t s r = x y x
					hd1xs[3 * iatom + 0][3 * jatom + 2][kgrid] += twoDij * ( ixx[kgrid] * jz[kgrid] + ix[kgrid] * jzx[kgrid] ); // t s r = x z x
					hd1xs[3 * iatom + 1][3 * jatom + 0][kgrid] += twoDij * ( iyx[kgrid] * jx[kgrid] + iy[kgrid] * jxx[kgrid] ); // t s r = y x x
					hd1xs[3 * iatom + 1][3 * jatom + 1][kgrid] += twoDij * ( iyx[kgrid] * jy[kgrid] + iy[kgrid] * jyx[kgrid] ); // t s r = y y x
					hd1xs[3 * iatom + 1][3 * jatom + 2][kgrid] += twoDij * ( iyx[kgrid] * jz[kgrid] + iy[kgrid] * jzx[kgrid] ); // t s r = y z x
					hd1xs[3 * iatom + 2][3 * jatom + 0][kgrid] += twoDij * ( izx[kgrid] * jx[kgrid] + iz[kgrid] * jxx[kgrid] ); // t s r = z x x
					hd1xs[3 * iatom + 2][3 * jatom + 1][kgrid] += twoDij * ( izx[kgrid] * jx[kgrid] + iz[kgrid] * jxx[kgrid] ); // t s r = z y x
					hd1xs[3 * iatom + 2][3 * jatom + 2][kgrid] += twoDij * ( izx[kgrid] * jz[kgrid] + iz[kgrid] * jzx[kgrid] ); // t s r = z z x

					hd1ys[3 * iatom + 0][3 * jatom + 0][kgrid] += twoDij * ( ixy[kgrid] * jx[kgrid] + ix[kgrid] * jxy[kgrid] ); // t s r = x x y
					hd1ys[3 * iatom + 0][3 * jatom + 1][kgrid] += twoDij * ( ixy[kgrid] * jy[kgrid] + ix[kgrid] * jyy[kgrid] ); // t s r = x y y
					hd1ys[3 * iatom + 0][3 * jatom + 2][kgrid] += twoDij * ( ixy[kgrid] * jz[kgrid] + ix[kgrid] * jzy[kgrid] ); // t s r = x z y
					hd1ys[3 * iatom + 1][3 * jatom + 0][kgrid] += twoDij * ( iyy[kgrid] * jx[kgrid] + iy[kgrid] * jxy[kgrid] ); // t s r = y x y
					hd1ys[3 * iatom + 1][3 * jatom + 1][kgrid] += twoDij * ( iyy[kgrid] * jy[kgrid] + iy[kgrid] * jyy[kgrid] ); // t s r = y y y
					hd1ys[3 * iatom + 1][3 * jatom + 2][kgrid] += twoDij * ( iyy[kgrid] * jz[kgrid] + iy[kgrid] * jzy[kgrid] ); // t s r = y z y
					hd1ys[3 * iatom + 2][3 * jatom + 0][kgrid] += twoDij * ( izy[kgrid] * jx[kgrid] + iz[kgrid] * jxy[kgrid] ); // t s r = z x y
					hd1ys[3 * iatom + 2][3 * jatom + 1][kgrid] += twoDij * ( izy[kgrid] * jx[kgrid] + iz[kgrid] * jxy[kgrid] ); // t s r = z y y
					hd1ys[3 * iatom + 2][3 * jatom + 2][kgrid] += twoDij * ( izy[kgrid] * jz[kgrid] + iz[kgrid] * jzy[kgrid] ); // t s r = z z y

					hd1zs[3 * iatom + 0][3 * jatom + 0][kgrid] += twoDij * ( ixz[kgrid] * jx[kgrid] + ix[kgrid] * jxz[kgrid] ); // t s r = x x z
					hd1zs[3 * iatom + 0][3 * jatom + 1][kgrid] += twoDij * ( ixz[kgrid] * jy[kgrid] + ix[kgrid] * jyz[kgrid] ); // t s r = x y z
					hd1zs[3 * iatom + 0][3 * jatom + 2][kgrid] += twoDij * ( ixz[kgrid] * jz[kgrid] + ix[kgrid] * jzz[kgrid] ); // t s r = x z z
					hd1zs[3 * iatom + 1][3 * jatom + 0][kgrid] += twoDij * ( iyz[kgrid] * jx[kgrid] + iy[kgrid] * jxz[kgrid] ); // t s r = y x z
					hd1zs[3 * iatom + 1][3 * jatom + 1][kgrid] += twoDij * ( iyz[kgrid] * jy[kgrid] + iy[kgrid] * jyz[kgrid] ); // t s r = y y z
					hd1zs[3 * iatom + 1][3 * jatom + 2][kgrid] += twoDij * ( iyz[kgrid] * jz[kgrid] + iy[kgrid] * jzz[kgrid] ); // t s r = y z z
					hd1zs[3 * iatom + 2][3 * jatom + 0][kgrid] += twoDij * ( izz[kgrid] * jx[kgrid] + iz[kgrid] * jxz[kgrid] ); // t s r = z x z
					hd1zs[3 * iatom + 2][3 * jatom + 1][kgrid] += twoDij * ( izz[kgrid] * jx[kgrid] + iz[kgrid] * jxz[kgrid] ); // t s r = z y z
					hd1zs[3 * iatom + 2][3 * jatom + 2][kgrid] += twoDij * ( izz[kgrid] * jz[kgrid] + iz[kgrid] * jzz[kgrid] ); // t s r = z z z
				}
			}
		}
	}
}
