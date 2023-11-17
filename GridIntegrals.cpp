#include <Eigen/Dense>
#include "Aliases.h"

void GetDensity(double * aos,
                double * ao1xs,double * ao1ys,double * ao1zs,
                double * ao2ls,
                int ngrids,EigenMatrix D,
                double * ds,
                double * d1xs,double * d1ys,double * d1zs,double * cgs,
                double * d2s,double * ts){
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
	double Dij;
        for (int kgrid=0;kgrid<ngrids;kgrid++){
                ds[kgrid]=0;
		if (! ao1xs) continue;
		d1xs[kgrid]=0;
		d1ys[kgrid]=0;
		d1zs[kgrid]=0;
		if (! ao2ls) continue;
		d2s[kgrid]=0;
		ts[kgrid]=0;
	}
	for (int ibasis=0;ibasis<D.cols();ibasis++){
		Dij=D(ibasis,ibasis);
		iao=aos+ibasis*ngrids; // ibasis*ngrids+jgrid
		if (ao1xs){
			ix=ao1xs+ibasis*ngrids;
			iy=ao1ys+ibasis*ngrids;
			iz=ao1zs+ibasis*ngrids;
			if (ao2ls) iao2=ao2ls+ibasis*ngrids;
		}
		for (int kgrid=0;kgrid<ngrids;kgrid++){
			ds[kgrid]+=Dij*iao[kgrid]*iao[kgrid];
			if (! ao1xs) continue;
			d1xs[kgrid]+=2*Dij*ix[kgrid]*iao[kgrid];
			d1ys[kgrid]+=2*Dij*iy[kgrid]*iao[kgrid];
			d1zs[kgrid]+=2*Dij*iz[kgrid]*iao[kgrid];
			if (! ao2ls) continue;
			ts[kgrid]+=0.5*Dij*(ix[kgrid]*ix[kgrid]+iy[kgrid]*iy[kgrid]+iz[kgrid]*iz[kgrid]);
			d2s[kgrid]+=2*Dij*iao[kgrid]*iao2[kgrid];
		}
		for (int jbasis=0;jbasis<ibasis;jbasis++){
			Dij=D(ibasis,jbasis);
			jao=aos+jbasis*ngrids;
			if (ao1xs){
				jx=ao1xs+jbasis*ngrids;
				jy=ao1ys+jbasis*ngrids;
				jz=ao1zs+jbasis*ngrids;
				if (ao2ls) jao2=ao2ls+jbasis*ngrids;
			}
			for (long int kgrid=0;kgrid<ngrids;kgrid++){
				ds[kgrid]+=2*Dij*iao[kgrid]*jao[kgrid];
				if (! ao1xs) continue;
				d1xs[kgrid]+=2*Dij*(ix[kgrid]*jao[kgrid]+iao[kgrid]*jx[kgrid]);
				d1ys[kgrid]+=2*Dij*(iy[kgrid]*jao[kgrid]+iao[kgrid]*jy[kgrid]);
				d1zs[kgrid]+=2*Dij*(iz[kgrid]*jao[kgrid]+iao[kgrid]*jz[kgrid]);
				if (! ao2ls) continue;
				ts[kgrid]+=Dij*(ix[kgrid]*jx[kgrid]+iy[kgrid]*jy[kgrid]+iz[kgrid]*jz[kgrid]);
				d2s[kgrid]+=2*Dij*(iao[kgrid]*jao2[kgrid]+iao2[kgrid]*jao[kgrid]);
			}
		}
	}
	for (int igrid=0;igrid<ngrids;igrid++){
		if (ao1xs)
			cgs[igrid]=d1xs[igrid]*d1xs[igrid]+d1ys[igrid]*d1ys[igrid]+d1zs[igrid]*d1zs[igrid];
		if (ao2ls)
			d2s[igrid]+=4*ts[igrid];
	}
}

void VectorAddition(double * as,double * bs,int ngrids){
	if (as && bs)
		for (int igrid=0;igrid<ngrids;igrid++)
			as[igrid]+=bs[igrid];
}

double SumUp(double * ds,double * ws,int ngrids){
	double n=0;
	for (int igrid=0;igrid<ngrids;igrid++)
		n+=ds[igrid]*ws[igrid];
	return n;
}

EigenMatrix FxcMatrix(
		bool u,double * ws,int ngrids,int nbasis,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2ls,
		double * ds,
		double * d1xs,double * d1ys,double * d1zs,double * cgs,
		double * vrs,double * vss,
		double * vls,double * vts,
		double * vrrs,double * vrss,double * vsss){
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
		iao=aos+irow*ngrids;
		if (ao1xs){
			ix=ao1xs+irow*ngrids;
			iy=ao1ys+irow*ngrids;
			iz=ao1zs+irow*ngrids;
			if (ao2ls) iao2=ao2ls+irow*ngrids;
		}
		for (int jcol=0;jcol<=irow;jcol++){
			jao=aos+jcol*ngrids;
			if (ao1xs){
				jx=ao1xs+jcol*ngrids;
				jy=ao1ys+jcol*ngrids;
				jz=ao1zs+jcol*ngrids;
				if (ao2ls) jao2=ao2ls+jcol*ngrids;
			}
			double fij=0;
			for (int kgrid=0;kgrid<ngrids;kgrid++){
				if (! u && ! ao1xs && ! ao2ls)
					fij+=ws[kgrid]*vrs[kgrid]*iao[kgrid]*jao[kgrid];
				else if (! u && ao1xs && ! ao2ls)
					fij+=ws[kgrid]*(vrs[kgrid]*iao[kgrid]*jao[kgrid]
					    +2*vss[kgrid]*(d1xs[kgrid]*(ix[kgrid]*jao[kgrid]+iao[kgrid]*jx[kgrid])
					                  +d1ys[kgrid]*(iy[kgrid]*jao[kgrid]+iao[kgrid]*jy[kgrid])
					                  +d1zs[kgrid]*(iz[kgrid]*jao[kgrid]+iao[kgrid]*jz[kgrid])));
				else if (! u && ao1xs && ao2ls)
					fij+=ws[kgrid]*(vrs[kgrid]*iao[kgrid]*jao[kgrid]
					    +2*vss[kgrid]*(d1xs[kgrid]*(ix[kgrid]*jao[kgrid]+iao[kgrid]*jx[kgrid])
					                  +d1ys[kgrid]*(iy[kgrid]*jao[kgrid]+iao[kgrid]*jy[kgrid])
					                  +d1zs[kgrid]*(iz[kgrid]*jao[kgrid]+iao[kgrid]*jz[kgrid]))
					    +(0.5*vts[kgrid]+2*vls[kgrid])*(ix[kgrid]*jx[kgrid]+iy[kgrid]*jy[kgrid]+iz[kgrid]*jz[kgrid])
					    +vls[kgrid]*(iao[kgrid]*jao2[kgrid]+iao2[kgrid]*jao[kgrid]));
			//	else if (u && ! ao1xs)
			//		fij+=
			}
			Fxc(irow,jcol)=fij;
			Fxc(jcol,irow)=fij;
		}
	}
	return Fxc;
}
/*
void FxcUMatrix(
		double * ws,int ngrids,int nbasis,int natoms,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ds,
		double * d1xs,double * d1ys,double * d1zs,double * cgs,
		double * vrrs,double * vrss,double * vsss,
		EigenMatrix * matrices){
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
	for (int irow=0;irow<nbasis;irow++){
		iao=aos+irow*ngrids;
		if (ao1xs){
			ix=ao1xs+irow*ngrids;
			iy=ao1ys+irow*ngrids;
			iz=ao1zs+irow*ngrids;
			if (ao2ls) iao2=ao2ls+irow*ngrids;
		}
		for (int jcol=0;jcol<=irow;jcol++){
			jao=aos+jcol*ngrids;
			if (ao1xs){
				jx=ao1xs+jcol*ngrids;
				jy=ao1ys+jcol*ngrids;
				jz=ao1zs+jcol*ngrids;
				if (ao2ls) jao2=ao2ls+jcol*ngrids;
			}
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
			Fxc(irow,jcol)=fij;
			Fxc(jcol,irow)=fij;
		}
	}
	return Fxc;
}*/
