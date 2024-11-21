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

void GetDensity(
		std::vector<int> orders,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		long int ngrids, EigenMatrix D,
		double* ds,
		double* d1xs, double* d1ys, double* d1zs,
		double* d2s, double* ts){


void GetDensitySkeleton(
		std::vector<int> orders,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2xxs,double * ao2yys,double * ao2zzs,
		double * ao2xys,double * ao2xzs,double * ao2yzs,
		long int ngrids, EigenMatrix D,
		std::vector<int> bf2atom,
		double * dnxs,double * dnys,double * dnzs,
		double * dxnxs,double * dxnys,double * dxnzs, // For example, dxnys is the x component of grid density gradient derivative with respect to a nuclear y coordinate perturbation.
		double * dynxs,double * dynys,double * dynzs,
		double * dznxs,double * dznys,double * dznzs){
	double * iao=aos; // AOs
	double * jao=aos;
	double * ix=ao1xs; // AO first derivatives
	double * jx=ao1xs;
	double * iy=ao1ys;
	double * jy=ao1ys;
	double * iz=ao1zs;
	double * jz=ao1zs;
	double * ixx=ao2xxs; // AO second derivatives
	//double * jxx=ao2xxs;
	double * iyy=ao2yys;
	//double * jyy=ao2yys;
	double * izz=ao2zzs;
	//double * jzz=ao2zzs;
	double * ixy=ao2xys;
	//double * jxy=ao2xys;
	double * ixz=ao2xzs;
	//double * jxz=ao2xzs;
	double * iyz=ao2yzs;
	//double * jyz=ao2yzs;
	double Dij;
	for (int ibasis=0;ibasis<D.cols();ibasis++){
		if (aos) iao=aos+ibasis*ngrids; // ibasis*ngrids+jgrid
		if (ao1xs){
			ix=ao1xs+ibasis*ngrids;
			iy=ao1ys+ibasis*ngrids;
			iz=ao1zs+ibasis*ngrids;
		}
		if (ao2xxs){
			ixx=ao2xxs+ibasis*ngrids;
			iyy=ao2yys+ibasis*ngrids;
			izz=ao2zzs+ibasis*ngrids;
			ixy=ao2xys+ibasis*ngrids;
			ixz=ao2xzs+ibasis*ngrids;
			iyz=ao2yzs+ibasis*ngrids;
		}
		for (int jbasis=0;jbasis<D.cols();jbasis++){
			Dij=D(ibasis,jbasis);
			if (aos) jao=aos+jbasis*ngrids;
			if (ao1xs){
				jx=ao1xs+jbasis*ngrids;
				jy=ao1ys+jbasis*ngrids;
				jz=ao1zs+jbasis*ngrids;
			}
			/*if (ao2xxs){
				jxx=ao2xxs+ibasis*ngrids;
				jyy=ao2yys+ibasis*ngrids;
				jzz=ao2zzs+ibasis*ngrids;
				jxy=ao2xys+ibasis*ngrids;
				jxz=ao2xzs+ibasis*ngrids;
				jyz=ao2yzs+ibasis*ngrids;
			}*/
			for (long int kgrid=0;kgrid<ngrids;kgrid++){
				if (aos && ao1xs && dnxs){
					dnxs[kgrid]-=Dij*((bf2atom[ibasis]==atom)*ix[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iao[kgrid]*jx[kgrid]);
					dnys[kgrid]-=Dij*((bf2atom[ibasis]==atom)*iy[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iao[kgrid]*jy[kgrid]);
					dnzs[kgrid]-=Dij*((bf2atom[ibasis]==atom)*iz[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iao[kgrid]*jz[kgrid]);
				}
				if (aos && ao1xs && ao2xxs && dxnxs){
					dxnxs[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*ixx[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*ix[kgrid]*jx[kgrid]);
					dxnys[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*ixy[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*ix[kgrid]*jy[kgrid]);
					dxnzs[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*ixz[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*ix[kgrid]*jz[kgrid]);
					dynxs[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*ixy[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iy[kgrid]*jx[kgrid]);
					dynys[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*iyy[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iy[kgrid]*jy[kgrid]);
					dynzs[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*iyz[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iy[kgrid]*jz[kgrid]);
					dznxs[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*ixz[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iz[kgrid]*jx[kgrid]);
					dznys[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*iyz[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iz[kgrid]*jy[kgrid]);
					dznzs[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*izz[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iz[kgrid]*jz[kgrid]);
				}
			}
		}
	}
}


