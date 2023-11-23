#include <Eigen/Dense>
#include "Aliases.h"
#include <iostream>

void GetDensitySkeleton(
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2xxs,double * ao2yys,double * ao2zzs,
		double * ao2xys,double * ao2xzs,double * ao2yzs,
		int ngrids,EigenMatrix D,
		short int atom,short int * bf2atom,
		double * d1nxs,double * d1nys,double * d1nzs,
		double * d2nxxs,double * d2nxys,double * d2nxzs, // For example, d2nxys is the x component of grid density gradient derivative with respect to a nuclear y coordinate perturbation.
		double * d2nyxs,double * d2nyys,double * d2nyzs,
		double * d2nzxs,double * d2nzys,double * d2nzzs){
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
				if (aos && ao1xs && d1nxs){
					d1nxs[kgrid]-=Dij*((bf2atom[ibasis]==atom)*ix[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iao[kgrid]*jx[kgrid]);
					d1nys[kgrid]-=Dij*((bf2atom[ibasis]==atom)*iy[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iao[kgrid]*jy[kgrid]);
					d1nzs[kgrid]-=Dij*((bf2atom[ibasis]==atom)*iz[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iao[kgrid]*jz[kgrid]);
				}
				if (aos && ao1xs && ao2xxs && d2nxxs){
					d2nxxs[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*ixx[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*ix[kgrid]*jx[kgrid]);
					d2nxys[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*ixy[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*ix[kgrid]*jy[kgrid]);
					d2nxzs[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*ixz[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*ix[kgrid]*jz[kgrid]);
					d2nyxs[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*ixy[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iy[kgrid]*jx[kgrid]);
					d2nyys[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*iyy[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iy[kgrid]*jy[kgrid]);
					d2nyzs[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*iyz[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iy[kgrid]*jz[kgrid]);
					d2nzxs[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*ixz[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iz[kgrid]*jx[kgrid]);
					d2nzys[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*iyz[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iz[kgrid]*jy[kgrid]);
					d2nzzs[kgrid]-=2*Dij*((bf2atom[ibasis]==atom)*izz[kgrid]*jao[kgrid]+(bf2atom[jbasis]==atom)*iz[kgrid]*jz[kgrid]);
				}
			}
		}
	}
}

void GetDensity(
		double * aos,
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
	for (int ibasis=0;ibasis<D.cols();ibasis++){
		Dij=D(ibasis,ibasis);
		if (aos) iao=aos+ibasis*ngrids; // ibasis*ngrids+jgrid
		if (ao1xs){
			ix=ao1xs+ibasis*ngrids;
			iy=ao1ys+ibasis*ngrids;
			iz=ao1zs+ibasis*ngrids;
		}
		if (ao2ls) iao2=ao2ls+ibasis*ngrids;
		for (int kgrid=0;kgrid<ngrids;kgrid++){
			if (ds) ds[kgrid]+=Dij*iao[kgrid]*iao[kgrid];
			if (d1xs){
				d1xs[kgrid]+=2*Dij*ix[kgrid]*iao[kgrid];
				d1ys[kgrid]+=2*Dij*iy[kgrid]*iao[kgrid];
				d1zs[kgrid]+=2*Dij*iz[kgrid]*iao[kgrid];
			}
			if (d2s){
				ts[kgrid]+=0.5*Dij*(ix[kgrid]*ix[kgrid]+iy[kgrid]*iy[kgrid]+iz[kgrid]*iz[kgrid]);
				d2s[kgrid]+=2*Dij*iao[kgrid]*iao2[kgrid];
			}
		}
		for (int jbasis=0;jbasis<ibasis;jbasis++){
			Dij=D(ibasis,jbasis);
			if (aos) jao=aos+jbasis*ngrids;
			if (ao1xs){
				jx=ao1xs+jbasis*ngrids;
				jy=ao1ys+jbasis*ngrids;
				jz=ao1zs+jbasis*ngrids;
			}
			if (ao2ls) jao2=ao2ls+jbasis*ngrids;
			for (long int kgrid=0;kgrid<ngrids;kgrid++){
				if (ds) ds[kgrid]+=2*Dij*iao[kgrid]*jao[kgrid];
				if (d1xs){
					d1xs[kgrid]+=2*Dij*(ix[kgrid]*jao[kgrid]+iao[kgrid]*jx[kgrid]);
					d1ys[kgrid]+=2*Dij*(iy[kgrid]*jao[kgrid]+iao[kgrid]*jy[kgrid]);
					d1zs[kgrid]+=2*Dij*(iz[kgrid]*jao[kgrid]+iao[kgrid]*jz[kgrid]);
				}
				if (d2s){
					ts[kgrid]+=Dij*(ix[kgrid]*jx[kgrid]+iy[kgrid]*jy[kgrid]+iz[kgrid]*jz[kgrid]);
					d2s[kgrid]+=2*Dij*(iao[kgrid]*jao2[kgrid]+iao2[kgrid]*jao[kgrid]);
				}
			}
		}
	}
	for (int igrid=0;igrid<ngrids;igrid++){
		if (cgs)
			cgs[igrid]=d1xs[igrid]*d1xs[igrid]+d1ys[igrid]*d1ys[igrid]+d1zs[igrid]*d1zs[igrid];
		if (d2s)
			d2s[igrid]+=4*ts[igrid];
	}
}

void VectorAddition(double * as,double * bs,double * cs,int ngrids){
	if (! as || ! bs || ! cs) return;
	if (as==bs)
		for (int igrid=0;igrid<ngrids;igrid++)
			as[igrid]+=cs[igrid];
	else
		for (int igrid=0;igrid<ngrids;igrid++)
			as[igrid]=bs[igrid]+cs[igrid];
}

void VectorMultiplication(double * as,double * bs,double * cs,int ngrids){
	if (! as || ! bs || ! cs) return;
	if (as==bs)
		for (int igrid=0;igrid<ngrids;igrid++)
			as[igrid]*=cs[igrid];
	else
		for (int igrid=0;igrid<ngrids;igrid++)
			as[igrid]=bs[igrid]*cs[igrid];
}

double SumUp(double * ds,double * ws,int ngrids){
	double n=0;
	for (int igrid=0;igrid<ngrids;igrid++)
		n+=ds[igrid]*ws[igrid];
	return n;
}

EigenMatrix FxcMatrix(
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

EigenMatrix FxcUMatrix(
		double * ws,int ngrids,int nbasis,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * d1xs,double * d1ys,double * d1zs,
		double * vss,
		double * vrrs,
		double * vrss,double * vsss,
		double * dns,
		double * dn1xs,double * dn1ys,double * dn1zs){
	double * iao=aos;
	double * jao=aos;
	double * ix=ao1xs;
	double * jx=ao1xs;
	double * iy=ao1ys;
	double * jy=ao1ys;
	double * iz=ao1zs;
	double * jz=ao1zs;
	EigenMatrix Fxc=EigenZero(nbasis,nbasis);
	for (int irow=0;irow<nbasis;irow++){
		if (aos) iao=aos+irow*ngrids;
		if (ao1xs){
			ix=ao1xs+irow*ngrids;
			iy=ao1ys+irow*ngrids;
			iz=ao1zs+irow*ngrids;
		}
		for (int jcol=0;jcol<=irow;jcol++){
			if (aos) jao=aos+jcol*ngrids;
			if (ao1xs){
				jx=ao1xs+jcol*ngrids;
				jy=ao1ys+jcol*ngrids;
				jz=ao1zs+jcol*ngrids;
			}
			double fij=0;
			for (int kgrid=0;kgrid<ngrids;kgrid++){
				if (! dn1xs)
					fij+=ws[kgrid]*vrrs[kgrid]*dns[kgrid]*iao[kgrid]*jao[kgrid];
				else if (dn1xs){
					const double tmp1=d1xs[kgrid]*dn1xs[kgrid]+d1ys[kgrid]*dn1ys[kgrid]+d1zs[kgrid]*dn1zs[kgrid];
					const double tmp2=d1xs[kgrid]*(ix[kgrid]*jao[kgrid]+iao[kgrid]*jx[kgrid])
					                 +d1ys[kgrid]*(iy[kgrid]*jao[kgrid]+iao[kgrid]*jy[kgrid])
					                 +d1zs[kgrid]*(iz[kgrid]*jao[kgrid]+iao[kgrid]*jz[kgrid]);
					const double tmp3=dn1xs[kgrid]*(ix[kgrid]*jao[kgrid]+iao[kgrid]*jx[kgrid])
					                 +dn1ys[kgrid]*(iy[kgrid]*jao[kgrid]+iao[kgrid]*jy[kgrid])
					                 +dn1zs[kgrid]*(iz[kgrid]*jao[kgrid]+iao[kgrid]*jz[kgrid]);
					fij+=ws[kgrid]*(vrrs[kgrid]*dns[kgrid]+2*vrss[kgrid]*tmp1*iao[kgrid]*jao[kgrid]);
					fij+=ws[kgrid]*2*(vrss[kgrid]*dns[kgrid]+2*vsss[kgrid]*tmp1)*tmp2;
					fij+=ws[kgrid]*2*vss[kgrid]*tmp3;
				}
			}
			Fxc(irow,jcol)=Fxc(jcol,irow)=fij;
		}
	}
	return Fxc;
}

