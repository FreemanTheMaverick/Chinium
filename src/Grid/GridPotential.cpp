#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>

#include "../Macro.h"


EigenMatrix VMatrix(
		std::vector<int> orders,
		double* ws, long int ngrids, int nbasis,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		double* d1xs, double* d1ys, double* d1zs,
		double* vrs, double* vss,
		double* vls, double* vts){
	bool zeroth = 0;
	bool first = 0;
	bool second = 0;
	if (std::find(orders.begin(), orders.end(), 0) != orders.end()){
		zeroth = 1;
		assert(aos && "AOs on grids do not exist!");
		assert(vrs && "\\frac{\\partial Exc}{\\partial \\rho} on grids array is not allocated!");
	}
	if (std::find(orders.begin(), orders.end(), 1) != orders.end()){
		first = 1;
		assert(aos && "AOs on grids do not exist!");
		assert(ao1xs && "First order x-derivatives of AOs on grids do not exist!");
		assert(ao1ys && "First order y-derivatives of AOs on grids do not exist!");
		assert(ao1zs && "First order z-derivatives of AOs on grids do not exist!");
		assert(d1xs && "First order x-derivatives of density on grids do not exist!");
		assert(d1ys && "First order y-derivatives of density on grids do not exist!");
		assert(d1zs && "First order z-derivatives of density on grids do not exist!");
		assert(vrs && "\\frac{\\partial Exc}{\\partial \\rho} on grids do not exist!");
		assert(vss && "\\frac{\\partial Exc}{\\partial \\sigma} on grids do not exist!");
	}
	if (std::find(orders.begin(), orders.end(), 2) != orders.end()){
		second = 1;
		assert(aos && "AOs on grids do not exist!");
		assert(ao1xs && "First order x-derivatives of AOs on grids do not exist!");
		assert(ao1ys && "First order y-derivatives of AOs on grids do not exist!");
		assert(ao1zs && "First order z-derivatives of AOs on grids do not exist!");
		assert(ao2ls && "Laplacians of AOs on grids do not exist!");
		assert(d1xs && "First order x-derivatives of density on grids do not exist!");
		assert(d1ys && "First order y-derivatives of density on grids do not exist!");
		assert(d1zs && "First order z-derivatives of density on grids do not exist!");
		assert(vrs && "\\frac{\\partial Exc}{\\partial \\rho} on grids do not exist!");
		assert(vss && "\\frac{\\partial Exc}{\\partial \\sigma} on grids do not exist!");
		assert(vls && "\\frac{\\partial Exc}{\\partial \\nabla^2 \\rho} on grids do not exist!");
		assert(vts && "\\frac{\\partial Exc}{\\partial \\tau} on grids do not exist!");
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
	EigenMatrix Fxc = EigenZero(nbasis, nbasis);
	for ( int irow=0; irow < nbasis; irow++ ){
		if (zeroth || first || second) iao = aos + irow * ngrids;
		if (first || second){
			ix = ao1xs + irow * ngrids;
			iy = ao1ys + irow * ngrids;
			iz = ao1zs + irow * ngrids;
		}
		if (second) iao2 = ao2ls + irow * ngrids;
		for ( int jcol = 0; jcol <= irow; jcol++ ){
			if (zeroth || first || second) jao = aos + jcol * ngrids;
			if (first || second){
				jx = ao1xs + jcol * ngrids;
				jy = ao1ys + jcol * ngrids;
				jz = ao1zs + jcol * ngrids;
			}
			if (second) jao2 = ao2ls + jcol * ngrids;
			double fij = 0;
			for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
				if (zeroth && !first && !second)
					fij += ws[kgrid] * vrs[kgrid] * iao[kgrid] * jao[kgrid];
				else if (first && !second)
					fij += ws[kgrid] * ( vrs[kgrid] * iao[kgrid] * jao[kgrid]
					    + 2 * vss[kgrid] * (
							  d1xs[kgrid] * ( ix[kgrid] * jao[kgrid] + iao[kgrid] * jx[kgrid] )
							+ d1ys[kgrid] * ( iy[kgrid] * jao[kgrid] + iao[kgrid] * jy[kgrid] )
							+ d1zs[kgrid] * ( iz[kgrid] * jao[kgrid] + iao[kgrid] * jz[kgrid] )
						)
					);
				else if (second)
					fij += ws[kgrid] * (
						vrs[kgrid] * iao[kgrid] * jao[kgrid]
					    + 2 * vss[kgrid] * (
							  d1xs[kgrid] * ( ix[kgrid] * jao[kgrid] + iao[kgrid] * jx[kgrid] )
							+ d1ys[kgrid] * ( iy[kgrid] * jao[kgrid] + iao[kgrid] * jy[kgrid] )
							+ d1zs[kgrid] * ( iz[kgrid] * jao[kgrid] + iao[kgrid] * jz[kgrid] )
						)
						+ ( 0.5 * vts[kgrid] + 2 * vls[kgrid] ) * ( ix[kgrid] * jx[kgrid] + iy[kgrid] * jy[kgrid] + iz[kgrid] * jz[kgrid] )
					    + vls[kgrid] * ( iao[kgrid] * jao2[kgrid] + iao2[kgrid] * jao[kgrid] )
					);
			}
			Fxc(irow, jcol) = Fxc(jcol, irow) = fij;
		}
	}
	return Fxc;
}

#define __Check_Vector_Array__(vec)\
	vec.size() > 0 && vec[0].size() == 3 && vec[0][0]

#define iyx ixy
#define izx ixz
#define izy iyz
#define jyx jxy
#define jzx jxz
#define jzy jyz

EigenMatrix PotentialSkeleton(
		std::vector<int> orders,
		double* ws, long int ngrids, int nbasis,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2xxs, double* ao2yys, double* ao2zzs,
		double* ao2xys, double* ao2xzs, double* ao2yzs,
		double* d1xs, double* d1ys, double* d1zs,
		double* vrs, double* vss,
		double* vrrs, double* vrss, double* vsss,
		double* gds,
		double* gd1xs,
		double* gd1ys,
		double* gd1zs){
	bool zeroth = 0;
	bool first = 0;
	bool second = 0;
	if (std::find(orders.begin(), orders.end(), 0) != orders.end()){
		zeroth = 1;
		assert(aos && "AOs on grids do not exist!");
		assert(ao1xs && "First order x-derivatives of AOs on grids do not exist!");
		assert(ao1ys && "First order y-derivatives of AOs on grids do not exist!");
		assert(ao1zs && "First order z-derivatives of AOs on grids do not exist!");
		assert(vrs && "\\frac{\\partial Exc}{\\partial \\rho} on grids do not exist!");
		assert(vrrs && "\\frac{\\partial^2 Exc}{\\partial \\rho^2} on grids do not exist!");
		assert(gds && "Nuclear gradient of density on grids arrays are not allocated!");
	}
	if (std::find(orders.begin(), orders.end(), 1) != orders.end()){
		first = 1;
		assert(aos && "AOs on grids do not exist!");
		assert(ao1xs && "First order x-derivatives of AOs on grids do not exist!");
		assert(ao1ys && "First order y-derivatives of AOs on grids do not exist!");
		assert(ao1zs && "First order z-derivatives of AOs on grids do not exist!");
		assert(ao2xxs && "Second order xx-derivatives of AOs on grids do not exist!");
		assert(ao2yys && "Second order yy-derivatives of AOs on grids do not exist!");
		assert(ao2zzs && "Second order zz-derivatives of AOs on grids do not exist!");
		assert(ao2xys && "Second order xy-derivatives of AOs on grids do not exist!");
		assert(ao2xzs && "Second order xz-derivatives of AOs on grids do not exist!");
		assert(ao2yzs && "Second order yz-derivatives of AOs on grids do not exist!");
		assert(d1xs && "First order x-derivatives of density on grids do not exist!");
		assert(d1ys && "First order y-derivatives of density on grids do not exist!");
		assert(d1zs && "First order z-derivatives of density on grids do not exist!");
		assert(gds && "Nuclear gradient of density on grids arrays are not allocated!");
		assert(gd1xs && "Nuclear gradient of density on grids arrays are not allocated!");
		assert(gd1ys && "Nuclear gradient of density on grids arrays are not allocated!");
		assert(gd1zs && "Nuclear gradient of density on grids arrays are not allocated!");
	}

	double* iao = aos;
	double* jao = aos;
	double* ix = ao1xs;
	double* jx = ao1xs;
	double* iy = ao1ys;
	double* jy = ao1ys;
	double* iz = ao1zs;
	double* jz = ao1zs;
	EigenMatrix V = EigenZero(nbasis, nbasis);
	if ( !orders.size() ) return V;

	// Part 1
	for ( int irow = 0; irow < nbasis; irow++ ){
		if (zeroth || first || second) iao = aos + irow * ngrids;
		if (first || second){
			ix = ao1xs + irow * ngrids;
			iy = ao1ys + irow * ngrids;
			iz = ao1zs + irow * ngrids;
		}
		for ( int jcol = 0; jcol <= irow; jcol++ ){
			if (zeroth || first || second) jao = aos + jcol * ngrids;
			if (first || second){
				jx = ao1xs + jcol * ngrids;
				jy = ao1ys + jcol * ngrids;
				jz = ao1zs + jcol * ngrids;
			}
			double Vij = 0;
			if (zeroth && !first && !second)
				for ( long int kgrid = 0; kgrid < ngrids; kgrid++ )
					Vij += ws[kgrid] * vrrs[kgrid] * gds[kgrid] * iao[kgrid] * jao[kgrid];
			else if (zeroth && first && !second){
				for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
					const double tmp = d1xs[kgrid] * gd1xs[kgrid] + d1ys[kgrid] * gd1ys[kgrid] + d1zs[kgrid] * gd1zs[kgrid];
					const double tmpx = ix[kgrid] * jao[kgrid] + iao[kgrid] * jx[kgrid];
					const double tmpy = iy[kgrid] * jao[kgrid] + iao[kgrid] * jy[kgrid];
					const double tmpz = iz[kgrid] * jao[kgrid] + iao[kgrid] * jz[kgrid];
					Vij += ws[kgrid] * (
						(
							vrrs[kgrid] * gds[kgrid] +
							2 * vrss[kgrid] * tmp
						) *
						iao[kgrid] * jao[kgrid] +
						(
							2 * vrss[kgrid] * gds[kgrid] +
							4 * vsss[kgrid] * tmp
						) *
						(
							d1xs[kgrid] * tmpx +
							d1ys[kgrid] * tmpy +
							d1zs[kgrid] * tmpz
						) +
						2 * vss[kgrid] * (
							gd1xs[kgrid] * tmpx +
							gd1ys[kgrid] * tmpy +
							gd1zs[kgrid] * tmpz
						)
					);
				}
			}
			V(irow, jcol) = V(jcol, irow) = Vij;
		}
	}
/*
	// Part 2
	for ( int irow = 0; irow < nbasis; irow++ ){
		if (zeroth || first || second){
			iao = aos + irow * ngrids;
			ix = ao1xs + irow * ngrids;
			iy = ao1ys + irow * ngrids;
			iz = ao1zs + irow * ngrids;
		}
		if (first || second){
			ixx = ao2xxs + irow * ngrids;
			iyy = ao2yys + irow * ngrids;
			izz = ao2zzs + irow * ngrids;
			ixy = ao2xys + irow * ngrids;
			ixz = ao2xzs + irow * ngrids;
			iyz = ao2yzs + irow * ngrids;
		}
		const int iatom = bf2atom[irow];
		for ( int jcol = 0; jcol <= irow; jcol++ ){
			if (zeroth || first || second){
				jao = aos + jcol * ngrids;
				jx = ao1xs + jcol * ngrids;
				jy = ao1ys + jcol * ngrids;
				jz = ao1zs + jcol * ngrids;
			}
			if (first || second){
				jxx = ao2xxs + jcol * ngrids;
				jyy = ao2yys + jcol * ngrids;
				jzz = ao2zzs + jcol * ngrids;
				jxy = ao2xys + jcol * ngrids;
				jxz = ao2xzs + jcol * ngrids;
				jyz = ao2yzs + jcol * ngrids;
			}
			const int jatom = bf2atom[jcol];
			for ( long int kgrid = 0; kgrid < ngrids; kgrid++ ){
				if (zeroth && !first && !second){
					const double tmp = ws[kgrid] * vrs[kgrid];
					const double tmpj = tmp * iao[kgrid];
					const double tmpi = tmp * jao[kgrid]; // I hope branch prediction will work.
					if ( !dones[3*iatom+0] ) Vs[iatom][0](irow, jcol) -= tmpj * ix[kgrid];
					if ( !dones[3*iatom+1] ) Vs[iatom][1](irow, jcol) -= tmpj * iy[kgrid];
					if ( !dones[3*iatom+2] ) Vs[iatom][2](irow, jcol) -= tmpj * iz[kgrid];
					if ( !dones[3*jatom+0] ) Vs[jatom][0](irow, jcol) -= tmpi * jx[kgrid];
					if ( !dones[3*jatom+1] ) Vs[jatom][1](irow, jcol) -= tmpi * jy[kgrid];
					if ( !dones[3*jatom+2] ) Vs[jatom][2](irow, jcol) -= tmpi * jz[kgrid];
				}
				if (zeroth && first && !second){
					const double tmp = 2 * ws[kgrid] * vss[kgrid];
					const double tmpi = d1xs[kgrid] * ix[kgrid] + d1ys[kgrid] * iy[kgrid] + d1zs[kgrid] * iz[kgrid];
					const double tmpj = d1xs[kgrid] * jx[kgrid] + d1ys[kgrid] * jy[kgrid] + d1zs[kgrid] * jz[kgrid];
					if ( !dones[3*iatom+0] ) Vs[iatom][0](irow, jcol) -= tmp * (
						(
							d1xs[kgrid] * ixx[kgrid] +
							d1ys[kgrid] * ixy[kgrid] +
							d1zs[kgrid] * ixz[kgrid]
						) * jao[kgrid] + tmpj * ix[kgrid]
					);
					if ( !dones[3*iatom+1] ) Vs[jatom][0](irow, jcol) -= tmp * (
						(
							d1xs[kgrid] * jxx[kgrid] +
							d1ys[kgrid] * jxy[kgrid] +
							d1zs[kgrid] * jxz[kgrid]
						) * iao[kgrid] + tmpi * jx[kgrid]
					);
					if ( !dones[3*iatom+2] ) Vs[iatom][1](irow, jcol) -= tmp * (
						(
							d1xs[kgrid] * iyx[kgrid] +
							d1ys[kgrid] * iyy[kgrid] +
							d1zs[kgrid] * iyz[kgrid]
						) * jao[kgrid] + tmpj * iy[kgrid]
					);
					if ( !dones[3*jatom+0] ) Vs[jatom][1](irow, jcol) -= tmp * (
						(
							d1xs[kgrid] * jyx[kgrid] +
							d1ys[kgrid] * jyy[kgrid] +
							d1zs[kgrid] * jyz[kgrid]
						) * iao[kgrid] + tmpi * jy[kgrid]
					);
					if ( !dones[3*jatom+1] ) Vs[iatom][2](irow, jcol) -= tmp * (
						(
							d1xs[kgrid] * izx[kgrid] +
							d1ys[kgrid] * izy[kgrid] +
							d1zs[kgrid] * izz[kgrid]
						) * jao[kgrid] + tmpj * iz[kgrid]
					);
					if ( !dones[3*jatom+2] ) Vs[jatom][2](irow, jcol) -= tmp * (
						(
							d1xs[kgrid] * jzx[kgrid] +
							d1ys[kgrid] * jzy[kgrid] +
							d1zs[kgrid] * jzz[kgrid]
						) * iao[kgrid] + tmpi * jz[kgrid]
					);
				}
			}
		}
	}*/
	return V;
}

