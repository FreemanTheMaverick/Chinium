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
		assert(vrs && "\\frac{\\partial Exc}{\\partial \\rho} on grids array is not allocated!");
		assert(vss && "\\frac{\\partial Exc}{\\partial \\sigma} on grids array is not allocated!");
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
		assert(vrs && "\\frac{\\partial Exc}{\\partial \\rho} on grids array is not allocated!");
		assert(vss && "\\frac{\\partial Exc}{\\partial \\sigma} on grids array is not allocated!");
		assert(vls && "\\frac{\\partial Exc}{\\partial \\nabla^2 \\rho} on grids array is not allocated!");
		assert(vts && "\\frac{\\partial Exc}{\\partial \\tau} on grids array is not allocated!");
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

