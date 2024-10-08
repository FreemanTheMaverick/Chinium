#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>

#include "../Macro.h"
#include "../Multiwfn.h"

EigenMatrix VMatrix(
		double * ws,int ngrids,int nbasis,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2ls,
		double * ds,
		double * d1xs,double * d1ys,double * d1zs,double * cgs,
		double * vrs,double * vss,
		double * vls,double * vts){
	double * iao=aos;
	double * jao=aos;
	double * ix=ao1xs;
	double * jx=ao1xs;
	double * iy=ao1ys;
	double * jy=ao1ys;
	double * iz=ao1zs;
	double * jz=ao1zs;
	double * iao2=ao2ls;
	double * jao2=ao2ls;
	EigenMatrix Fxc=EigenZero(nbasis,nbasis);
	for (int irow=0;irow<nbasis;irow++){
		if (aos) iao=aos+irow*ngrids;
		if (ao1xs){
			ix=ao1xs+irow*ngrids;
			iy=ao1ys+irow*ngrids;
			iz=ao1zs+irow*ngrids;
		}
		if (ao2ls) iao2=ao2ls+irow*ngrids;
		for (int jcol=0;jcol<=irow;jcol++){
			if (aos) jao=aos+jcol*ngrids;
			if (ao1xs){
				jx=ao1xs+jcol*ngrids;
				jy=ao1ys+jcol*ngrids;
				jz=ao1zs+jcol*ngrids;
			}
			if (ao2ls) jao2=ao2ls+jcol*ngrids;
			double fij=0;
			for (int kgrid=0;kgrid<ngrids;kgrid++){
				if (! ao1xs && ! ao2ls)
					fij+=ws[kgrid]*vrs[kgrid]*iao[kgrid]*jao[kgrid];
				else if (ao1xs && ! ao2ls)
					fij+=ws[kgrid]*(vrs[kgrid]*iao[kgrid]*jao[kgrid]
					    +2*vss[kgrid]*(d1xs[kgrid]*(ix[kgrid]*jao[kgrid]+iao[kgrid]*jx[kgrid])
					                  +d1ys[kgrid]*(iy[kgrid]*jao[kgrid]+iao[kgrid]*jy[kgrid])
					                  +d1zs[kgrid]*(iz[kgrid]*jao[kgrid]+iao[kgrid]*jz[kgrid])));
				else if (ao1xs && ao2ls)
					fij+=ws[kgrid]*(vrs[kgrid]*iao[kgrid]*jao[kgrid]
					    +2*vss[kgrid]*(d1xs[kgrid]*(ix[kgrid]*jao[kgrid]+iao[kgrid]*jx[kgrid])
					                  +d1ys[kgrid]*(iy[kgrid]*jao[kgrid]+iao[kgrid]*jy[kgrid])
					                  +d1zs[kgrid]*(iz[kgrid]*jao[kgrid]+iao[kgrid]*jz[kgrid]))
					    +(0.5*vts[kgrid]+2*vls[kgrid])*(ix[kgrid]*jx[kgrid]+iy[kgrid]*jy[kgrid]+iz[kgrid]*jz[kgrid])
					    +vls[kgrid]*(iao[kgrid]*jao2[kgrid]+iao2[kgrid]*jao[kgrid]));
			}
			Fxc(irow,jcol)=Fxc(jcol,irow)=fij;
		}
	}
	return Fxc;
}

