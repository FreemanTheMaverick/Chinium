#include <Eigen/Dense>
#include <libint2.hpp>
#include <cmath>
#include <ctime>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <functional>
#include "Aliases.h"
#include "Libint2.h"
#include "sphere_lebedev_rule.hpp"

int SphericalGridNumber(std::string grid,const int natoms,double * atoms,const bool output){
	int ngrids=0;
	__Z_2_Name__
	for (int iatom=0;iatom<natoms;iatom++){
		int ngroups;
		std::ifstream gridfile(std::string(__Grid_library_path__)+"/"+grid+"/"+Z2Name[(int)(atoms[4*iatom])]+".grid");
		assert(("Missing element grid file" && gridfile.good()));
		std::string thisline;
		getline(gridfile,thisline);
		std::stringstream ss_(thisline);
		ss_>>ngroups;
		getline(gridfile,thisline);
		for (int igroup=0;igroup<ngroups;igroup++){
			getline(gridfile,thisline);
			std::stringstream ss(thisline);
			int nshells,npoints;
			ss>>nshells;ss>>npoints;
			ngrids+=nshells*npoints;
		}
	}
	if (output) std::cout<<"Number of grid points ... "<<ngrids<<std::endl;
	return ngrids;
}

double s_p_mu(double x0,double y0,double z0,double x1,double y1,double z1,double x2,double y2,double z2){
	const double x01=x0-x1;const double y01=y0-y1;const double z01=z0-z1;
	const double r01=sqrt(x01*x01+y01*y01+z01*z01);
	const double x02=x0-x2;const double y02=y0-y2;const double z02=z0-z2;
	const double r02=sqrt(x02*x02+y02*y02+z02*z02);
	const double x12=x1-x2;const double y12=y1-y2;const double z12=z1-z2;
	const double r12=sqrt(x12*x12+y12*y12+z12*z12);
	const double mu=(r01-r02)/r12;
	const double p1=1.5*mu-0.5*mu*mu*mu;
	const double p2=1.5*p1-0.5*p1*p1*p1;
	const double p3=1.5*p2-0.5*p2*p2*p2;
	return 0.5*(1-p3);
}

void SphericalGrid(std::string grid,const int natoms,double * atoms,
                   double * xs,double * ys,double * zs,double * ws,
                   const bool output){
	const clock_t start=clock();
	double * x_ranger=xs;
	double * y_ranger=ys;
	double * z_ranger=zs;
	double * w_ranger=ws;
	__Z_2_Name__
	for (int iatom=0;iatom<natoms;iatom++){
		std::ifstream gridfile(std::string(__Grid_library_path__)+"/"+grid+"/"+Z2Name[(int)(atoms[4*iatom])]+".grid");
		assert(("Missing element grid file in" __Grid_library_path__ && gridfile.good()));
		std::string thisline;
		getline(gridfile,thisline);
		std::stringstream ss_(thisline);
		int ngroups,nshells_total;
		ss_>>ngroups;ss_>>nshells_total;
		getline(gridfile,thisline); // Radial-formula-dependent line.
		std::stringstream ss__(thisline);
		std::string radialformula;ss__>>radialformula;
		std::function<double(double)> ri_func;
		std::function<double(double)> radial_weight_func;
		if (radialformula.compare("de2")==0){
			double token1,token2,token3;
			ss__>>token1;ss__>>token2;ss__>>token3;
			const double a=token1;
			const double h=(token3-token2)/(double)(nshells_total-1);
			ri_func=[=](double i){
				const double xi=h*i+token2;
				return exp(a*xi-exp(-xi));
			};
			radial_weight_func=[=](double i){
				const double xi=h*i+token2;
				return exp(3*a*xi-3*exp(-xi))*(a+exp(-xi))*h;
			};
		}else if (radialformula.compare("em")==0){
			double token1;ss__>>token1;
			const double R=token1;
			ri_func=[=](double i){
				const double ii=i+1;
				return R*pow(ii/(nshells_total+1-ii),2);
			};
			radial_weight_func=[=](double i){
				const double ii=i+1;
				return 2*pow(R,3)*(nshells_total+1)*pow(ii,5)/pow(nshells_total+1-ii,7);
			};
		}else assert(("Unrecognized radial formula!" && 0));
		int ishell_total=0;
		for (int igroup=0;igroup<ngroups;igroup++){
			getline(gridfile,thisline);
			std::stringstream ss(thisline);
			int npoints,nshells;
			ss>>npoints;ss>>nshells;
			double lebedev_xs[8192]={0};
			double lebedev_ys[8192]={0};
			double lebedev_zs[8192]={0};
			double lebedev_ws[8192]={0};
			ld_by_order(npoints,lebedev_xs,lebedev_ys,lebedev_zs,lebedev_ws);
			for (int ishell=0;ishell<nshells;ishell++,ishell_total++){
				const double ri=ri_func(ishell_total);
				const double radial_w=radial_weight_func(ishell_total);
				for (int ipoint=0;ipoint<npoints;ipoint++){
					const double x=lebedev_xs[ipoint]*ri+atoms[4*iatom+1];
					const double y=lebedev_ys[ipoint]*ri+atoms[4*iatom+2];
					const double z=lebedev_zs[ipoint]*ri+atoms[4*iatom+3];
					const double lebedev_w=4*M_PI*lebedev_ws[ipoint];
					double unnorm_becke_w_total=0;
					double unnorm_becke_w=0;
					for (int jatom=0;jatom<natoms;jatom++){
						double unnorm_becke_wj=1;
						const double xj=atoms[4*jatom+1];
						const double yj=atoms[4*jatom+2];
						const double zj=atoms[4*jatom+3];
						for (int katom=0;katom<natoms;katom++){
							if (jatom==katom) continue;
							const double xk=atoms[4*katom+1];
							const double yk=atoms[4*katom+2];
							const double zk=atoms[4*katom+3];
							unnorm_becke_wj*=s_p_mu(x,y,z,xj,yj,zj,xk,yk,zk);
						}
						unnorm_becke_w_total+=unnorm_becke_wj;
						if (iatom==jatom)
							unnorm_becke_w=unnorm_becke_wj;
					}
					const double becke_w=unnorm_becke_w/unnorm_becke_w_total;
					const double w=radial_w*lebedev_w*becke_w;
					*(x_ranger++)=x;
					*(y_ranger++)=y;
					*(z_ranger++)=z;
					*(w_ranger++)=w;
				}
			}
		}
	}
	if (output) std::cout<<"Generating grid points and weights ... "<<double(clock()-start)/CLOCKS_PER_SEC<<" s"<<std::endl;
}

#define __Uniform_Box_Grid_Number__\
	const std::vector<libint2::Atom> libint2atoms=Libint2Atoms(natoms,atoms);\
	const libint2::BasisSet obs(basisset,libint2atoms);\
	double xu=-10000;\
	double yu=-10000;\
	double zu=-10000;\
	double xl=10000;\
	double yl=10000;\
	double zl=10000;\
	for (const auto& shell:obs){\
		const double x=shell.O[0];\
		const double y=shell.O[1];\
		const double z=shell.O[2];\
		xu=xu>x?xu:x;\
		yu=yu>y?yu:y;\
		zu=zu>z?zu:z;\
		xl=xl<x?xl:x;\
		yl=yl<y?yl:y;\
		zl=zl<z?zl:z;\
	}\
	xu+=overheadlength;\
	yu+=overheadlength;\
	zu+=overheadlength;\
	xl-=overheadlength;\
	yl-=overheadlength;\
	zl-=overheadlength;\
	const double xlength=xu-xl;\
	const double ylength=yu-yl;\
	const double zlength=zu-zl;\
	const int nx=int(xlength/spacing);\
	const int ny=int(ylength/spacing);\
	const int nz=int(zlength/spacing);

long int UniformBoxGridNumber(const int natoms,double * atoms,const char * basisset,double overheadlength,double spacing){
	__Uniform_Box_Grid_Number__
	return (long int)nx*(long int)ny*(long int)nz;
}

void UniformBoxGrid(const int natoms,double * atoms,const char * basisset,double overheadlength,double spacing,double * xs,double * ys,double * zs){
	__Uniform_Box_Grid_Number__
	double * x_ranger=xs;
	double * y_ranger=ys;
	double * z_ranger=zs;
	double x,y,z;
	for (int ix=0;ix<nx;ix++){
		x=xl+ix*spacing;
		for (int iy=0;iy<ny;iy++){
			y=yl+iy*spacing;
			for (int iz=0;iz<nz;iz++){
				z=zl+iz*spacing;
				*(x_ranger++)=x;
				*(y_ranger++)=y;
				*(z_ranger++)=z;
			}
		}
	}
}

void GetAoValues(const int natoms,double * atoms,const char * basisset,
                 double * xs,double * ys,double * zs,int ngrids,
                 double * aos,
                 double * ao1xs,double * ao1ys,double * ao1zs,
                 double * ao2s){ // ibasis*ngrids+jgrid
 __Basis_From_Atoms__
 double xo,yo,zo,x,y,z,r2;
 double tmp,tmp1,tmp2;
 double A,Ax,Ay,Az,Axx,Ayy,Azz; // Basis function values;
 A=Ax=Ay=Az=Axx=Ayy=Azz=0;
 double ao2xx=0;double ao2yy=0;double ao2zz=0;
 double * ao_rangers[16];
 double * ao1x_rangers[16];
 double * ao1y_rangers[16];
 double * ao1z_rangers[16];
 double * ao2_rangers[16];
 double Q[16]={0};
 double Qx[16]={0};
 double Qy[16]={0};
 double Qz[16]={0};
 double Qxx[16]={0};
 double Qyy[16]={0};
 double Qzz[16]={0};
 int two_l_plus_one,nprims; // These in-loop variables had better be declared outside the loop. Tests showed that the codes as such are faster than those otherwise.
 int ibasis=0;
 for (const libint2::Shell & shell:obs){
  two_l_plus_one=shell.size();
  nprims=shell.nprim();
  for (int iranger=0;iranger<two_l_plus_one;iranger++,ibasis++){
   ao_rangers[iranger]=aos+ibasis*ngrids;
   if (ao1xs){
    ao1x_rangers[iranger]=ao1xs+ibasis*ngrids;
    ao1y_rangers[iranger]=ao1ys+ibasis*ngrids;
    ao1z_rangers[iranger]=ao1zs+ibasis*ngrids;
    if (ao2s)
     ao2_rangers[iranger]=ao2s+ibasis*ngrids;
   }
  }
  xo=shell.O[0];
  yo=shell.O[1];
  zo=shell.O[2];
  for (int k=0;k<ngrids;++k){
   x=xs[k]-xo; // Coordintes with the origin at the centre of the shell.
   y=ys[k]-yo;
   z=zs[k]-zo;
   r2=x*x+y*y+z*z;
   tmp=tmp1=tmp2=0;
   A=Az=Azz=0;
   for (int iprim=0;iprim<nprims;++iprim){ // Looping over primitive gaussians.
    tmp=shell.contr[0].coeff[iprim]*std::exp(-shell.alpha[iprim]*r2);
    A+=tmp; // Each primitive gaussian contributes to the radial parts of basis functions.
    if (ao1xs){
     tmp1+=tmp*shell.alpha[iprim];
     if (ao2s)
      tmp2+=tmp*shell.alpha[iprim]*shell.alpha[iprim];
    }
   }
   if (ao1xs){
    Ax=-2*x*tmp1;
    Ay=-2*y*tmp1;
    Az=-2*z*tmp1;
    if (ao2s){
     Axx=-2*tmp1+4*tmp2*x;
     Ayy=-2*tmp1+4*tmp2*y;
     Azz=-2*tmp1+4*tmp2*z;
    }
   }
   switch(two_l_plus_one){ // Basis function values are the products of the radial parts and the spherical harmonic parts. The following spherical harmonics are rescaled as suggested @ https://github.com/evaleev/libint/wiki/using-modern-CPlusPlus-API. The normalization constants here differ from those in "Symmetries of Spherical Harmonics: applications to ambisonics" @ https://iaem.at/ambisonics/symposium2009/proceedings/ambisym09-chapman-shsymmetries.pdf/@@download/file/AmbiSym09_Chapman_SHSymmetries.pdf by a factor of sqrt(2l+1).
    case 1: // S
     Q[0]=1;
     if (ao1xs){ // (uv)' = u'v + uv'
      Qx[0]=0;
      Qy[0]=0;
      Qz[0]=0;
      if (ao2s){
       Qxx[0]=0;
       Qyy[0]=0;
       Qzz[0]=0;
      }
     }
     break;
    case 3: // P
     Q[0]=y;
     Q[1]=z;
     Q[2]=x;
     if (ao1xs){
      Qx[0]=0;
      Qy[0]=1;
      Qz[0]=0;
      Qx[1]=0;
      Qy[1]=0;
      Qz[1]=1;
      Qx[2]=1;
      Qy[2]=0;
      Qz[2]=0;
      if (ao2s){
       Qxx[0]=0;
       Qyy[0]=0;
       Qzz[0]=0;
       Qxx[1]=0;
       Qyy[1]=0;
       Qzz[1]=0;
       Qxx[2]=0;
       Qyy[2]=0;
       Qzz[2]=0;
      }
     }
     break;
    case 5: // D
     Q[0]=x*y*sqrt(3);
     Q[1]=y*z*sqrt(3);
     Q[2]=(3*z*z-r2)/2;
     Q[3]=x*z*sqrt(3);
     Q[4]=(x*x-y*y)*sqrt(3)/2;
     if (ao1xs){
      Qx[0]=sqrt(3)*y;
      Qy[0]=sqrt(3)*x;
      Qz[0]=0;
      Qx[1]=0;
      Qy[1]=sqrt(3)*z;
      Qz[1]=sqrt(3)*y;
      Qx[2]=( -x );
      Qy[2]=( -y );
      Qz[2]=2*z;
      Qx[3]=sqrt(3)*z;
      Qy[3]=0;
      Qz[3]=sqrt(3)*x;
      Qx[4]=sqrt(3)*x;
      Qy[4]=( -sqrt(3)*y );
      Qz[4]=0;
      if (ao2s){
       Qxx[0]=0;
       Qyy[0]=0;
       Qzz[0]=0;
       Qxx[1]=0;
       Qyy[1]=0;
       Qzz[1]=0;
       Qxx[2]=-1;
       Qyy[2]=-1;
       Qzz[2]=2;
       Qxx[3]=0;
       Qyy[3]=0;
       Qzz[3]=0;
       Qxx[4]=sqrt(3);
       Qyy[4]=-sqrt(3);
       Qzz[4]=0;
      }
     }
     break;
    case 7: // F
     Q[0]=y*(3*x*x-y*y)*sqrt(10)/4;
     Q[1]=x*y*z*sqrt(15);
     Q[2]=y*(5*z*z-r2)*sqrt(6)/4;
     Q[3]=(5*z*z*z-3*z*r2)/2;
     Q[4]=x*(5*z*z-r2)*sqrt(6)/4;
     Q[5]=(x*x-y*y)*z*sqrt(15)/2;
     Q[6]=x*(x*x-3*y*y)*sqrt(10)/4;
     if (ao1xs){
      Qx[0]=3*sqrt(10)*x*y/2;
      Qy[0]=3*sqrt(10)*(x*x - y*y)/4;
      Qz[0]=0;
      Qx[1]=sqrt(15)*y*z;
      Qy[1]=sqrt(15)*x*z;
      Qz[1]=sqrt(15)*x*y;
      Qx[2]=( -sqrt(6)*x*y/2 );
      Qy[2]=sqrt(6)*(-x*x - 3*y*y + 4*z*z)/4;
      Qz[2]=2*sqrt(6)*y*z;
      Qx[3]=( -3*x*z );
      Qy[3]=( -3*y*z );
      Qz[3]=( -3*x*x/2 - 3*y*y/2 + 3*z*z );
      Qx[4]=sqrt(6)*(-3*x*x - y*y + 4*z*z)/4;
      Qy[4]=( -sqrt(6)*x*y/2 );
      Qz[4]=2*sqrt(6)*x*z;
      Qx[5]=sqrt(15)*x*z;
      Qy[5]=( -sqrt(15)*y*z );
      Qz[5]=sqrt(15)*(x*x - y*y)/2;
      Qx[6]=3*sqrt(10)*(x*x - y*y)/4;
      Qy[6]=( -3*sqrt(10)*x*y/2 );
      Qz[6]=0;
      if (ao2s){
       Qxx[0]=3*sqrt(10)*y/2;
       Qyy[0]=-3*sqrt(10)*y/2;
       Qzz[0]=0;
       Qxx[1]=0;
       Qyy[1]=0;
       Qzz[1]=0;
       Qxx[2]=-sqrt(6)*y/2;
       Qyy[2]=-3*sqrt(6)*y/2;
       Qzz[2]=2*sqrt(6)*y;
       Qxx[3]=-3*z;
       Qyy[3]=-3*z;
       Qzz[3]=6*z;
       Qxx[4]=-3*sqrt(6)*x/2;
       Qyy[4]=-sqrt(6)*x/2;
       Qzz[4]=2*sqrt(6)*x;
       Qxx[5]=sqrt(15)*z;
       Qyy[5]=-sqrt(15)*z;
       Qzz[5]=0;
       Qxx[6]=3*sqrt(10)*x/2;
       Qyy[6]=-3*sqrt(10)*x/2;
       Qzz[6]=0;
      }
     }
     break;
    case 9: // G
     Q[0]=x*y*(x*x-y*y)*sqrt(35)/2;
     Q[1]=y*(3*x*x-y*y)*z*sqrt(70)/4;
     Q[2]=x*y*(7*z*z-r2)*sqrt(5)/2;
     Q[3]=y*(7*z*z*z-3*z*r2)*sqrt(10)/4;
     Q[4]=(35*z*z*z*z-30*z*z*r2+3*r2*r2)/8;
     Q[5]=x*(7*z*z*z-3*z*r2)*sqrt(10)/4;
     Q[6]=(x*x-y*y)*(7*z*z-r2)*sqrt(5)/4;
     Q[7]=x*(x*x-3*y*y)*z*sqrt(70)/4;
     Q[8]=(x*x*(x*x-3*y*y)-y*y*(3*x*x-y*y))*sqrt(35)/8;
     if (ao1xs){
      Qx[0]=sqrt(35)*y*(3*x*x - y*y)/2;
      Qy[0]=sqrt(35)*x*(x*x - 3*y*y)/2;
      Qz[0]=0;
      Qx[1]=3*sqrt(70)*x*y*z/2;
      Qy[1]=3*sqrt(70)*z*(x*x - y*y)/4;
      Qz[1]=sqrt(70)*y*(3*x*x - y*y)/4;
      Qx[2]=sqrt(5)*y*(-3*x*x - y*y + 6*z*z)/2;
      Qy[2]=sqrt(5)*x*(-x*x - 3*y*y + 6*z*z)/2;
      Qz[2]=6*sqrt(5)*x*y*z;
      Qx[3]=( -3*sqrt(10)*x*y*z/2 );
      Qy[3]=sqrt(10)*z*(-3*x*x - 9*y*y + 4*z*z)/4;
      Qz[3]=3*sqrt(10)*y*(-x*x - y*y + 4*z*z)/4;
      Qx[4]=3*x*(x*x + y*y - 4*z*z)/2;
      Qy[4]=3*y*(x*x + y*y - 4*z*z)/2;
      Qz[4]=2*z*(-3*x*x - 3*y*y + 2*z*z);
      Qx[5]=sqrt(10)*z*(-9*x*x - 3*y*y + 4*z*z)/4;
      Qy[5]=( -3*sqrt(10)*x*y*z/2 );
      Qz[5]=3*sqrt(10)*x*(-x*x - y*y + 4*z*z)/4;
      Qx[6]=sqrt(5)*x*(-x*x + 3*z*z);
      Qy[6]=sqrt(5)*y*(y*y - 3*z*z);
      Qz[6]=3*sqrt(5)*z*(x*x - y*y);
      Qx[7]=3*sqrt(70)*z*(x*x - y*y)/4;
      Qy[7]=( -3*sqrt(70)*x*y*z/2 );
      Qz[7]=sqrt(70)*x*(x*x - 3*y*y)/4;
      Qx[8]=sqrt(35)*x*(x*x - 3*y*y)/2;
      Qy[8]=sqrt(35)*y*(-3*x*x + y*y)/2;
      Qz[8]=0;
      if (ao2s){
       Qxx[0]=3*sqrt(35)*x*y;
       Qyy[0]=-3*sqrt(35)*x*y;
       Qzz[0]=0;
       Qxx[1]=3*sqrt(70)*y*z/2;
       Qyy[1]=-3*sqrt(70)*y*z/2;
       Qzz[1]=0;
       Qxx[2]=-3*sqrt(5)*x*y;
       Qyy[2]=-3*sqrt(5)*x*y;
       Qzz[2]=6*sqrt(5)*x*y;
       Qxx[3]=-3*sqrt(10)*y*z/2;
       Qyy[3]=-9*sqrt(10)*y*z/2;
       Qzz[3]=6*sqrt(10)*y*z;
       Qxx[4]=9*x*x/2 + 3*y*y/2 - 6*z*z;
       Qyy[4]=3*x*x/2 + 9*y*y/2 - 6*z*z;
       Qzz[4]=-6*x*x - 6*y*y + 12*z*z;
       Qxx[5]=-9*sqrt(10)*x*z/2;
       Qyy[5]=-3*sqrt(10)*x*z/2;
       Qzz[5]=6*sqrt(10)*x*z;
       Qxx[6]=3*sqrt(5)*(-x*x + z*z);
       Qyy[6]=3*sqrt(5)*(y*y - z*z);
       Qzz[6]=3*sqrt(5)*(x*x - y*y);
       Qxx[7]=3*sqrt(70)*x*z/2;
       Qyy[7]=-3*sqrt(70)*x*z/2;
       Qzz[7]=0;
       Qxx[8]=3*sqrt(35)*(x*x - y*y)/2;
       Qyy[8]=3*sqrt(35)*(-x*x + y*y)/2;
       Qzz[8]=0;
      }
     }
     break;
    case 11: // H
     Q[0]=y*(5*x*x*x*x-10*x*x*y*y+y*y*y*y)*3*sqrt(14)/16;
     Q[1]=x*y*z*(x*x-y*y)*3*sqrt(35)/2;
     Q[2]=y*(y*y*y*y-2*x*x*y*y-3*x*x*x*x-8*y*y*z*z+24*x*x*z*z)*sqrt(70)/16;
     Q[3]=x*y*z*(2*z*z-x*x-y*y)*sqrt(105)/2;
     Q[4]=y*(x*x*x*x+2*x*x*y*y+y*y*y*y-12*x*x*z*z-12*y*y*z*z+8*z*z*z*z)*sqrt(15)/8;
     Q[5]=z*(15*x*x*x*x+15*y*y*y*y+8*z*z*z*z+30*x*x*y*y-40*x*x*z*z-40*y*y*z*z)/8;
     Q[6]=x*(x*x*x*x+2*x*x*y*y+y*y*y*y-12*x*x*z*z-12*y*y*z*z+8*z*z*z*z)*sqrt(15)/8;
     Q[7]=z*(2*x*x*z*z-2*y*y*z*z-x*x*x*x+y*y*y*y)*sqrt(105)/4;
     Q[8]=x*(2*x*x*y*y+8*x*x*z*z-24*y*y*z*z-x*x*x*x+3*y*y*y*y)*sqrt(70)/16;
     Q[9]=z*(x*x*x*x-6*x*x*y*y+y*y*y*y)*3*sqrt(35)/8;
     Q[10]=x*(x*x*x*x-10*x*x*y*y+5*y*y*y*y)*3*sqrt(14)/16;
     if (ao1xs){
      Qx[0]=15*sqrt(14)*x*y*(x*x - y*y)/4;
      Qy[0]=15*sqrt(14)*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/16;
      Qz[0]=0;
      Qx[1]=3*sqrt(35)*y*z*(3*x*x - y*y)/2;
      Qy[1]=3*sqrt(35)*x*z*(x*x - 3*y*y)/2;
      Qz[1]=3*sqrt(35)*x*y*(x*x - y*y)/2;
      Qx[2]=sqrt(70)*x*y*(-3*x*x - y*y + 12*z*z)/4;
      Qy[2]=sqrt(70)*(-3*x*x*x*x - 6*x*x*y*y + 24*x*x*z*z + 5*y*y*y*y - 24*y*y*z*z)/16;
      Qz[2]=sqrt(70)*y*z*(3*x*x - y*y);
      Qx[3]=sqrt(105)*y*z*(-3*x*x - y*y + 2*z*z)/2;
      Qy[3]=sqrt(105)*x*z*(-x*x - 3*y*y + 2*z*z)/2;
      Qz[3]=sqrt(105)*x*y*(-x*x - y*y + 6*z*z)/2;
      Qx[4]=sqrt(15)*x*y*(x*x + y*y - 6*z*z)/2;
      Qy[4]=sqrt(15)*(x*x*x*x + 6*x*x*y*y - 12*x*x*z*z + 5*y*y*y*y - 36*y*y*z*z + 8*z*z*z*z)/8;
      Qz[4]=sqrt(15)*y*z*(-3*x*x - 3*y*y + 4*z*z);
      Qx[5]=5*x*z*(3*x*x + 3*y*y - 4*z*z)/2;
      Qy[5]=5*y*z*(3*x*x + 3*y*y - 4*z*z)/2;
      Qz[5]=( 15*x*x*x*x/8 + 15*x*x*y*y/4 - 15*x*x*z*z + 15*y*y*y*y/8 - 15*y*y*z*z + 5*z*z*z*z );
      Qx[6]=sqrt(15)*(5*x*x*x*x + 6*x*x*y*y - 36*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z)/8;
      Qy[6]=sqrt(15)*x*y*(x*x + y*y - 6*z*z)/2;
      Qz[6]=sqrt(15)*x*z*(-3*x*x - 3*y*y + 4*z*z);
      Qx[7]=sqrt(105)*x*z*(-x*x + z*z);
      Qy[7]=sqrt(105)*y*z*(y*y - z*z);
      Qz[7]=sqrt(105)*(-x*x*x*x + 6*x*x*z*z + y*y*y*y - 6*y*y*z*z)/4;
      Qx[8]=sqrt(70)*(-5*x*x*x*x + 6*x*x*y*y + 24*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z)/16;
      Qy[8]=sqrt(70)*x*y*(x*x + 3*y*y - 12*z*z)/4;
      Qz[8]=sqrt(70)*x*z*(x*x - 3*y*y);
      Qx[9]=3*sqrt(35)*x*z*(x*x - 3*y*y)/2;
      Qy[9]=3*sqrt(35)*y*z*(-3*x*x + y*y)/2;
      Qz[9]=3*sqrt(35)*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/8;
      Qx[10]=15*sqrt(14)*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/16;
      Qy[10]=15*sqrt(14)*x*y*(-x*x + y*y)/4;
      Qz[10]=0;
      if (ao2s){
       Qxx[0]=15*sqrt(14)*y*(3*x*x - y*y)/4;
       Qyy[0]=15*sqrt(14)*y*(-3*x*x + y*y)/4;
       Qzz[0]=0;
       Qxx[1]=9*sqrt(35)*x*y*z;
       Qyy[1]=-9*sqrt(35)*x*y*z;
       Qzz[1]=0;
       Qxx[2]=sqrt(70)*y*(-9*x*x - y*y + 12*z*z)/4;
       Qyy[2]=sqrt(70)*y*(-3*x*x + 5*y*y - 12*z*z)/4;
       Qzz[2]=sqrt(70)*y*(3*x*x - y*y);
       Qxx[3]=-3*sqrt(105)*x*y*z;
       Qyy[3]=-3*sqrt(105)*x*y*z;
       Qzz[3]=6*sqrt(105)*x*y*z;
       Qxx[4]=sqrt(15)*y*(3*x*x + y*y - 6*z*z)/2;
       Qyy[4]=sqrt(15)*y*(3*x*x + 5*y*y - 18*z*z)/2;
       Qzz[4]=3*sqrt(15)*y*(-x*x - y*y + 4*z*z);
       Qxx[5]=5*z*(9*x*x + 3*y*y - 4*z*z)/2;
       Qyy[5]=5*z*(3*x*x + 9*y*y - 4*z*z)/2;
       Qzz[5]=10*z*(-3*x*x - 3*y*y + 2*z*z);
       Qxx[6]=sqrt(15)*x*(5*x*x + 3*y*y - 18*z*z)/2;
       Qyy[6]=sqrt(15)*x*(x*x + 3*y*y - 6*z*z)/2;
       Qzz[6]=3*sqrt(15)*x*(-x*x - y*y + 4*z*z);
       Qxx[7]=sqrt(105)*z*(-3*x*x + z*z);
       Qyy[7]=sqrt(105)*z*(3*y*y - z*z);
       Qzz[7]=3*sqrt(105)*z*(x*x - y*y);
       Qxx[8]=sqrt(70)*x*(-5*x*x + 3*y*y + 12*z*z)/4;
       Qyy[8]=sqrt(70)*x*(x*x + 9*y*y - 12*z*z)/4;
       Qzz[8]=sqrt(70)*x*(x*x - 3*y*y);
       Qxx[9]=9*sqrt(35)*z*(x*x - y*y)/2;
       Qyy[9]=9*sqrt(35)*z*(-x*x + y*y)/2;
       Qzz[9]=0;
       Qxx[10]=15*sqrt(14)*x*(x*x - 3*y*y)/4;
       Qyy[10]=15*sqrt(14)*x*(-x*x + 3*y*y)/4;
       Qzz[10]=0;
      }
     }
     break;
    case 13: // I
     Q[0]=x*y*(3*x*x*x*x-10*x*x*y*y+3*y*y*y*y)*sqrt(462)/16;
     Q[1]=y*z*(5*x*x*x*x-10*x*x*y*y+y*y*y*y)*3*sqrt(154)/16;
     Q[2]=x*y*(-x*x*x*x+y*y*y*y+10*x*x*z*z-10*y*y*z*z)*3*sqrt(7)/4;
     Q[3]=y*z*(-9*x*x*x*x+3*y*y*y*y-6*x*x*y*y+24*x*x*z*z-8*y*y*z*z)*sqrt(210)/16;
     Q[4]=x*y*(x*x*x*x+y*y*y*y+16*z*z*z*z+2*x*x*y*y-16*x*x*z*z-16*y*y*z*z)*sqrt(210)/16;
     Q[5]=y*z*(5*x*x*x*x+5*y*y*y*y+8*z*z*z*z+10*x*x*y*y-20*x*x*z*z-20*y*y*z*z)*sqrt(21)/8;
     Q[6]=(16*z*z*z*z*z*z-5*x*x*x*x*x*x-5*y*y*y*y*y*y-15*x*x*x*x*y*y+90*x*x*x*x*z*z+90*y*y*y*y*z*z-120*y*y*z*z*z*z-15*x*x*y*y*y*y-120*x*x*z*z*z*z+180*x*x*y*y*z*z)/16;
     Q[7]=x*z*(5*x*x*x*x+5*y*y*y*y+8*z*z*z*z+10*x*x*y*y-20*x*x*z*z-20*y*y*z*z)*sqrt(21)/8;
     Q[8]=(x*x*x*x*x*x-y*y*y*y*y*y+x*x*x*x*y*y-x*x*y*y*y*y-16*x*x*x*x*z*z+16*x*x*z*z*z*z+16*y*y*y*y*z*z-16*y*y*z*z*z*z)*sqrt(210)/32;
     Q[9]=x*z*(-3*x*x*x*x+6*x*x*y*y+8*x*x*z*z-24*y*y*z*z+9*y*y*y*y)*sqrt(210)/16;
     Q[10]=(-x*x*x*x*x*x+5*x*x*x*x*y*y+10*x*x*x*x*z*z+5*x*x*y*y*y*y+10*y*y*y*y*z*z-y*y*y*y*y*y-60*x*x*y*y*z*z)*3*sqrt(7)/16;
     Q[11]=x*z*(x*x*x*x-10*x*x*y*y+5*y*y*y*y)*3*sqrt(154)/16;
     Q[12]=(x*x*x*x*x*x-15*x*x*x*x*y*y+15*x*x*y*y*y*y-y*y*y*y*y*y)*sqrt(462)/32;
     if (ao1xs){
      Qx[0]=3*sqrt(462)*y*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y)/16;
      Qy[0]=3*sqrt(462)*x*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y)/16;
      Qz[0]=0;
      Qx[1]=15*sqrt(154)*x*y*z*(x*x - y*y)/4;
      Qy[1]=15*sqrt(154)*z*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/16;
      Qz[1]=3*sqrt(154)*y*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y)/16;
      Qx[2]=3*sqrt(7)*y*(-5*x*x*x*x + 30*x*x*z*z + y*y*y*y - 10*y*y*z*z)/4;
      Qy[2]=3*sqrt(7)*x*(-x*x*x*x + 10*x*x*z*z + 5*y*y*y*y - 30*y*y*z*z)/4;
      Qz[2]=15*sqrt(7)*x*y*z*(x*x - y*y);
      Qx[3]=3*sqrt(210)*x*y*z*(-3*x*x - y*y + 4*z*z)/4;
      Qy[3]=3*sqrt(210)*z*(-3*x*x*x*x - 6*x*x*y*y + 8*x*x*z*z + 5*y*y*y*y - 8*y*y*z*z)/16;
      Qz[3]=3*sqrt(210)*y*(-3*x*x*x*x - 2*x*x*y*y + 24*x*x*z*z + y*y*y*y - 8*y*y*z*z)/16;
      Qx[4]=sqrt(210)*y*(5*x*x*x*x + 6*x*x*y*y - 48*x*x*z*z + y*y*y*y - 16*y*y*z*z + 16*z*z*z*z)/16;
      Qy[4]=sqrt(210)*x*(x*x*x*x + 6*x*x*y*y - 16*x*x*z*z + 5*y*y*y*y - 48*y*y*z*z + 16*z*z*z*z)/16;
      Qz[4]=2*sqrt(210)*x*y*z*(-x*x - y*y + 2*z*z);
      Qx[5]=5*sqrt(21)*x*y*z*(x*x + y*y - 2*z*z)/2;
      Qy[5]=sqrt(21)*z*(5*x*x*x*x + 30*x*x*y*y - 20*x*x*z*z + 25*y*y*y*y - 60*y*y*z*z + 8*z*z*z*z)/8;
      Qz[5]=5*sqrt(21)*y*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z)/8;
      Qx[6]=15*x*(-x*x*x*x - 2*x*x*y*y + 12*x*x*z*z - y*y*y*y + 12*y*y*z*z - 8*z*z*z*z)/8;
      Qy[6]=15*y*(-x*x*x*x - 2*x*x*y*y + 12*x*x*z*z - y*y*y*y + 12*y*y*z*z - 8*z*z*z*z)/8;
      Qz[6]=3*z*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z)/4;
      Qx[7]=sqrt(21)*z*(25*x*x*x*x + 30*x*x*y*y - 60*x*x*z*z + 5*y*y*y*y - 20*y*y*z*z + 8*z*z*z*z)/8;
      Qy[7]=5*sqrt(21)*x*y*z*(x*x + y*y - 2*z*z)/2;
      Qz[7]=5*sqrt(21)*x*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z)/8;
      Qx[8]=sqrt(210)*x*(3*x*x*x*x + 2*x*x*y*y - 32*x*x*z*z - y*y*y*y + 16*z*z*z*z)/16;
      Qy[8]=sqrt(210)*y*(x*x*x*x - 2*x*x*y*y - 3*y*y*y*y + 32*y*y*z*z - 16*z*z*z*z)/16;
      Qz[8]=sqrt(210)*z*(-x*x*x*x + 2*x*x*z*z + y*y*y*y - 2*y*y*z*z);
      Qx[9]=3*sqrt(210)*z*(-5*x*x*x*x + 6*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 8*y*y*z*z)/16;
      Qy[9]=3*sqrt(210)*x*y*z*(x*x + 3*y*y - 4*z*z)/4;
      Qz[9]=3*sqrt(210)*x*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z)/16;
      Qx[10]=3*sqrt(7)*x*(-3*x*x*x*x + 10*x*x*y*y + 20*x*x*z*z + 5*y*y*y*y - 60*y*y*z*z)/8;
      Qy[10]=3*sqrt(7)*y*(5*x*x*x*x + 10*x*x*y*y - 60*x*x*z*z - 3*y*y*y*y + 20*y*y*z*z)/8;
      Qz[10]=15*sqrt(7)*z*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/4;
      Qx[11]=15*sqrt(154)*z*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/16;
      Qy[11]=15*sqrt(154)*x*y*z*(-x*x + y*y)/4;
      Qz[11]=3*sqrt(154)*x*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y)/16;
      Qx[12]=3*sqrt(462)*x*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y)/16;
      Qy[12]=3*sqrt(462)*y*(-5*x*x*x*x + 10*x*x*y*y - y*y*y*y)/16;
      Qz[12]=0;
      if (ao2s){
       Qx[0]=15*sqrt(462)*x*y*(x*x - y*y)/4;
       Qy[0]=15*sqrt(462)*x*y*(-x*x + y*y)/4;
       Qz[0]=0;
       Qx[1]=15*sqrt(154)*y*z*(3*x*x - y*y)/4;
       Qy[1]=15*sqrt(154)*y*z*(-3*x*x + y*y)/4;
       Qz[1]=0;
       Qx[2]=15*sqrt(7)*x*y*(-x*x + 3*z*z);
       Qy[2]=15*sqrt(7)*x*y*(y*y - 3*z*z);
       Qz[2]=15*sqrt(7)*x*y*(x*x - y*y);
       Qx[3]=3*sqrt(210)*y*z*(-9*x*x - y*y + 4*z*z)/4;
       Qy[3]=3*sqrt(210)*y*z*(-3*x*x + 5*y*y - 4*z*z)/4;
       Qz[3]=3*sqrt(210)*y*z*(3*x*x - y*y);
       Qx[4]=sqrt(210)*x*y*(5*x*x + 3*y*y - 24*z*z)/4;
       Qy[4]=sqrt(210)*x*y*(3*x*x + 5*y*y - 24*z*z)/4;
       Qz[4]=2*sqrt(210)*x*y*(-x*x - y*y + 6*z*z);
       Qx[5]=5*sqrt(21)*y*z*(3*x*x + y*y - 2*z*z)/2;
       Qy[5]=5*sqrt(21)*y*z*(3*x*x + 5*y*y - 6*z*z)/2;
       Qz[5]=5*sqrt(21)*y*z*(-3*x*x - 3*y*y + 4*z*z);
       Qx[6]=-75*x*x*x*x/8 - 45*x*x*y*y/4 + 135*x*x*z*z/2 - 15*y*y*y*y/8 + 45*y*y*z*z/2 - 15*z*z*z*z;
       Qy[6]=-15*x*x*x*x/8 - 45*x*x*y*y/4 + 45*x*x*z*z/2 - 75*y*y*y*y/8 + 135*y*y*z*z/2 - 15*z*z*z*z;
       Qz[6]=45*x*x*x*x/4 + 45*x*x*y*y/2 - 90*x*x*z*z + 45*y*y*y*y/4 - 90*y*y*z*z + 30*z*z*z*z;
       Qx[7]=5*sqrt(21)*x*z*(5*x*x + 3*y*y - 6*z*z)/2;
       Qy[7]=5*sqrt(21)*x*z*(x*x + 3*y*y - 2*z*z)/2;
       Qz[7]=5*sqrt(21)*x*z*(-3*x*x - 3*y*y + 4*z*z);
       Qx[8]=sqrt(210)*(15*x*x*x*x + 6*x*x*y*y - 96*x*x*z*z - y*y*y*y + 16*z*z*z*z)/16;
       Qy[8]=sqrt(210)*(x*x*x*x - 6*x*x*y*y - 15*y*y*y*y + 96*y*y*z*z - 16*z*z*z*z)/16;
       Qz[8]=sqrt(210)*(-x*x*x*x + 6*x*x*z*z + y*y*y*y - 6*y*y*z*z);
       Qx[9]=3*sqrt(210)*x*z*(-5*x*x + 3*y*y + 4*z*z)/4;
       Qy[9]=3*sqrt(210)*x*z*(x*x + 9*y*y - 4*z*z)/4;
       Qz[9]=3*sqrt(210)*x*z*(x*x - 3*y*y);
       Qx[10]=15*sqrt(7)*(-3*x*x*x*x + 6*x*x*y*y + 12*x*x*z*z + y*y*y*y - 12*y*y*z*z)/8;
       Qy[10]=15*sqrt(7)*(x*x*x*x + 6*x*x*y*y - 12*x*x*z*z - 3*y*y*y*y + 12*y*y*z*z)/8;
       Qz[10]=15*sqrt(7)*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/4;
       Qx[11]=15*sqrt(154)*x*z*(x*x - 3*y*y)/4;
       Qy[11]=15*sqrt(154)*x*z*(-x*x + 3*y*y)/4;
       Qz[11]=0;
       Qx[12]=15*sqrt(462)*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/16;
       Qy[12]=15*sqrt(462)*(-x*x*x*x + 6*x*x*y*y - y*y*y*y)/16;
       Qz[12]=0;
      }
     }
     break;
   }
   for (int i=0;i<two_l_plus_one;i++){
    ao_rangers[i][k]=A*Q[i];
    if (ao1xs){
     ao1x_rangers[i][k]=Ax*Q[i]+A*Qx[i];
     ao1y_rangers[i][k]=Ay*Q[i]+A*Qy[i];
     ao1z_rangers[i][k]=Az*Q[i]+A*Qz[i];
     if (ao2s){
      ao2xx=Axx*Q[i]+Ax*Qx[i]+A*Qxx[i];
      ao2yy=Ayy*Q[i]+Ay*Qy[i]+A*Qyy[i];
      ao2zz=Azz*Q[i]+Az*Qz[i]+A*Qzz[i];
      ao2_rangers[i][k]=ao2xx+ao2yy+ao2zz;
     }
    }
   }
  }
 }
}

void GetDensity(double * aos,
                double * ao1xs,double * ao1ys,double * ao1zs,
                double * ao2s,
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
	double * iao2=ao2s;
	double * jao2=ao2s;
	double Dij;
        for (int kgrid=0;kgrid<ngrids;kgrid++){
                ds[kgrid]=0;
		if (! ao1xs) continue;
		d1xs[kgrid]=0;
		d1ys[kgrid]=0;
		d1zs[kgrid]=0;
		if (! ao2s) continue;
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
			if (ao2s) iao2=ao2s+ibasis*ngrids;
		}
		for (int kgrid=0;kgrid<ngrids;kgrid++){
			ds[kgrid]+=Dij*iao[kgrid]*iao[kgrid];
			if (! ao1xs) continue;
			d1xs[kgrid]+=2*Dij*ix[kgrid]*iao[kgrid];
			d1ys[kgrid]+=2*Dij*iy[kgrid]*iao[kgrid];
			d1zs[kgrid]+=2*Dij*iz[kgrid]*iao[kgrid];
			if (! ao2s) continue;
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
				if (ao2s) jao2=ao2s+jbasis*ngrids;
			}
			for (long int kgrid=0;kgrid<ngrids;kgrid++){
				ds[kgrid]+=2*Dij*iao[kgrid]*jao[kgrid];
				if (! ao1xs) continue;
				d1xs[kgrid]+=2*Dij*(ix[kgrid]*jao[kgrid]+iao[kgrid]*jx[kgrid]);
				d1ys[kgrid]+=2*Dij*(iy[kgrid]*jao[kgrid]+iao[kgrid]*jy[kgrid]);
				d1zs[kgrid]+=2*Dij*(iz[kgrid]*jao[kgrid]+iao[kgrid]*jz[kgrid]);
				if (! ao2s) continue;
				ts[kgrid]+=Dij*(ix[kgrid]*jx[kgrid]+iy[kgrid]*jy[kgrid]+iz[kgrid]*jz[kgrid]);
				d2s[kgrid]+=2*Dij*(iao[kgrid]*jao2[kgrid]+iao2[kgrid]*jao[kgrid]);
			}
		}
	}
	for (int igrid=0;igrid<ngrids;igrid++){
		if (ao1xs)
			cgs[igrid]=d1xs[igrid]*d1xs[igrid]+d1ys[igrid]*d1ys[igrid]+d1zs[igrid]*d1zs[igrid];
		if (ao2s)
			d2s[igrid]+=4*ts[igrid];
	}
}

void VectorAddition(double * as,double * bs,int ngrids){
	for (int igrid=0;igrid<ngrids;igrid++)
		as[igrid]+=bs[igrid];
}

double SumUp(double * ds,double * ws,int ngrids){
	double n=0;
	for (int igrid=0;igrid<ngrids;igrid++)
		n+=ds[igrid]*ws[igrid];
	return n;
}

EigenMatrix FxcMatrix(double * aos,double * vrs,
                      double * d1xs,double * d1ys,double * d1zs,
                      double * ao1xs,double * ao1ys,double * ao1zs,double * vss,
                      double * ao2s,double * vls,double * vts,
                      double * ws,int ngrids,int nbasis){
	double * iao=aos;
	double * jao=aos;
	double * ix=ao1xs;
	double * jx=ao1xs;
	double * iy=ao1ys;
	double * jy=ao1ys;
	double * iz=ao1zs;
	double * jz=ao1zs;
	double * iao2=ao2s;
	double * jao2=ao2s;
	EigenMatrix Fxc=EigenZero(nbasis,nbasis);
	for (int irow=0;irow<nbasis;irow++){
		iao=aos+irow*ngrids;
		if (ao1xs){
			ix=ao1xs+irow*ngrids;
			iy=ao1ys+irow*ngrids;
			iz=ao1zs+irow*ngrids;
			if (ao2s) iao2=ao2s+irow*ngrids;
		}
		for (int jcol=0;jcol<=irow;jcol++){
			jao=aos+jcol*ngrids;
			if (ao1xs){
				jx=ao1xs+jcol*ngrids;
				jy=ao1ys+jcol*ngrids;
				jz=ao1zs+jcol*ngrids;
				if (ao2s) jao2=ao2s+jcol*ngrids;
			}
			double fij=0;
			for (int kgrid=0;kgrid<ngrids;kgrid++){
				if (! ao1xs && ! ao2s)
					fij+=ws[kgrid]*vrs[kgrid]*iao[kgrid]*jao[kgrid];
				else if (ao1xs && ! ao2s)
					fij+=ws[kgrid]*(vrs[kgrid]*iao[kgrid]*jao[kgrid]
					    +2*vss[kgrid]*(d1xs[kgrid]*(ix[kgrid]*jao[kgrid]+iao[kgrid]*jx[kgrid])
					                  +d1ys[kgrid]*(iy[kgrid]*jao[kgrid]+iao[kgrid]*jy[kgrid])
					                  +d1zs[kgrid]*(iz[kgrid]*jao[kgrid]+iao[kgrid]*jz[kgrid])));
				else if (ao1xs && ao2s)
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
}
