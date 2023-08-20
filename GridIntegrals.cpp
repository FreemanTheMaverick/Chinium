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

long int SphericalGridNumber(std::string grid,const int natoms,double * atoms,const bool output){
	long int ngrids=0;
	__Z_2_Name__
	for (int iatom=0;iatom<natoms;iatom++){
		int ngroups;
		std::ifstream gridfile(std::string(__Grid_library_path__)+"/"+grid+"/"+Z2Name[(int)(atoms[4*iatom])]+".grid");
		assert((void("Not enough element grid files in " __Grid_library_path__),gridfile.good()));
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

double s_p_miu(double x0,double y0,double z0,double x1,double y1,double z1,double x2,double y2,double z2){
	const double x01=x0-x1;const double y01=y0-y1;const double z01=z0-z1;
	const double r01=sqrt(x01*x01+y01*y01+z01*z01);
	const double x02=x0-x2;const double y02=y0-y2;const double z02=z0-z2;
	const double r02=sqrt(x02*x02+y02*y02+z02*z02);
	const double x12=x1-x2;const double y12=y1-y2;const double z12=z1-z2;
	const double r12=sqrt(x12*x12+y12*y12+z12*z12);
	const double miu=(r01-r02)/r12;
	const double p1=1.5*miu+0.5*miu*miu*miu;
	const double p2=1.5*p1+0.5*p1*p1*p1;
	const double p3=1.5*p2+0.5*p2*p2*p2;
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
		assert((void("Not enough element grid files in " __Grid_library_path__),gridfile.good()));
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
				return R*pow(i/(nshells_total+1-i),2);
			};
			radial_weight_func=[=](double i){
				return 2*pow(R,3)*(nshells_total+1)*pow(i,5)/pow(nshells_total+1-i,7);
			};
		}else assert((void("Unrecognized radial formula!"),0));
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
							unnorm_becke_wj*=s_p_miu(x,y,z,xj,yj,zj,xk,yk,zk);
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
	if (output) std::cout<<"Generating grid points and weights ... "<<double(clock()-start)/CLOCKS_PER_SEC<<" s"<<std::endl;;
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
                 double * xs,double * ys,double * zs,long int ngrids,
                 double * aos){ // ibasis*ngrids+jgrid
	const std::vector<libint2::Atom> libint2atoms=Libint2Atoms(natoms,atoms);
	const libint2::BasisSet obs(basisset,libint2atoms);
	double xo,yo,zo,x,y,z,r2,a; // Basis function values;
	double * x_ranger,* y_ranger,* z_ranger;
	double * ao_rangers[16];
	int two_l_plus_one,nprims; // These in-loop variables had better be declared outside the loop. Tests showed that the codes as such are faster than those otherwise.
	int ibasis=0;
	for (const libint2::Shell & shell:obs){
		xo=shell.O[0];
		yo=shell.O[1];
		zo=shell.O[2];
		x_ranger=xs;
		y_ranger=ys;
		z_ranger=zs;
		two_l_plus_one=shell.size();
		for (int iranger=0;iranger<two_l_plus_one;iranger++,ibasis++)
			ao_rangers[iranger]=aos+ibasis*ngrids;
		nprims=shell.nprim();
		for (long int k=0;k<ngrids;++k){
			x=*(x_ranger++)-xo; // Coordintes with the origin at the centre of the shell.
			y=*(y_ranger++)-yo;
			z=*(z_ranger++)-zo;
			r2=x*x+y*y+z*z;
			a=0;
			for (int iprim=0;iprim<nprims;++iprim) // Looping over primitive gaussians.
				a+=shell.contr[0].coeff[iprim]*std::exp(-shell.alpha[iprim]*r2); // Each primitive gaussian contributes to the radial parts of basis functions.
			switch(two_l_plus_one){ // Basis function values are the products of the radial parts and the spherical harmonic parts.
				case 1: // S
					*(ao_rangers[0]++)=a;
					break;
				case 3: // P
					*(ao_rangers[0]++)=a*x;
					*(ao_rangers[1]++)=a*y;
					*(ao_rangers[2]++)=a*z;
					break;
				case 5: // D and the following spherical harmonics are rescaled as suggested @ https://github.com/evaleev/libint/wiki/using-modern-CPlusPlus-API. The normalization constants here differ from those from "Symmetries of Spherical Harmonics: applications to ambisonics" @ https://iaem.at/ambisonics/symposium2009/proceedings/ambisym09-chapman-shsymmetries.pdf/@@download/file/AmbiSym09_Chapman_SHSymmetries.pdf by a factor of sqrt(2l+1) except if l=1.
					*(ao_rangers[0]++)=a*x*y*sqrt(3);
					*(ao_rangers[1]++)=a*y*z*sqrt(3);
					*(ao_rangers[2]++)=a*(3*z*z-r2)*0.5;
					*(ao_rangers[3]++)=a*x*z*sqrt(3);
					*(ao_rangers[4]++)=a*(x*x-y*y)*sqrt(3)*0.5;
					break;
				case 7: // F
					*(ao_rangers[0]++)=a*y*(3*x*x-y*y)*sqrt(10)/4;
					*(ao_rangers[1]++)=a*x*y*z*sqrt(15);
					*(ao_rangers[2]++)=a*y*(5*z*z-r2)*sqrt(6)/4;
					*(ao_rangers[3]++)=a*(5*z*z*z-3*z*r2)/2;
					*(ao_rangers[4]++)=a*x*(5*z*z-r2)*sqrt(6)/4;
					*(ao_rangers[5]++)=a*(x*x-y*y)*z*sqrt(15)/2;
					*(ao_rangers[6]++)=a*x*(x*x-3*y*y)*sqrt(10)/4;
					break;
				case 9: // G
					*(ao_rangers[0]++)=a*x*y*(x*x-y*y)*sqrt(35)/2;
					*(ao_rangers[1]++)=a*y*(3*x*x-y*y)*z*sqrt(70)/4;
					*(ao_rangers[2]++)=a*x*y*(7*z*z-r2)*sqrt(5)/2;
					*(ao_rangers[3]++)=a*y*(7*z*z*z-3*z*r2)*sqrt(10)/4;
					*(ao_rangers[4]++)=a*(35*z*z*z*z-30*z*z*r2+3*r2*r2)/8;
					*(ao_rangers[5]++)=a*x*(7*z*z*z-3*z*r2)*sqrt(10)/4;
					*(ao_rangers[6]++)=a*(x*x-y*y)*(7*z*z-r2)*sqrt(5)/4;
					*(ao_rangers[7]++)=a*x*(x*x-3*y*y)*z*sqrt(70)/4;
					*(ao_rangers[8]++)=a*(x*x*(x*x-3*y*y)-y*y*(3*x*x-y*y))*sqrt(35)/8;
					break;
				case 11: // H
					*(ao_rangers[0]++)=a*y*(5*x*x*x*x-10*x*x*y*y+y*y*y*y)*3*sqrt(14)/16;
					*(ao_rangers[1]++)=a*x*y*z*(x*x-y*y)*3*sqrt(35)/2;
					*(ao_rangers[2]++)=a*y*(y*y*y*y-2*x*x*y*y-3*x*x*x*x-8*y*y*z*z+24*x*x*z*z)*sqrt(70)/16;
					*(ao_rangers[3]++)=a*x*y*z*(2*z*z-x*x-y*y)*sqrt(105)/2;
					*(ao_rangers[4]++)=a*y*(x*x*x*x+2*x*x*y*y+y*y*y*y-12*x*x*z*z-12*y*y*z*z+8*z*z*z*z)*sqrt(15)/8;
					*(ao_rangers[5]++)=a*z*(15*x*x*x*x+15*y*y*y*y+8*z*z*z*z+30*x*x*y*y-40*x*x*z*z-40*y*y*z*z)/8;
					*(ao_rangers[6]++)=a*x*(x*x*x*x+2*x*x*y*y+y*y*y*y-12*x*x*z*z-12*y*y*z*z+8*z*z*z*z)*sqrt(15)/8;
					*(ao_rangers[7]++)=a*z*(2*x*x*z*z-2*y*y*z*z-x*x*x*x+y*y*y*y)*sqrt(105)/4;
					*(ao_rangers[8]++)=a*x*(2*x*x*y*y+8*x*x*z*z-24*y*y*z*z-x*x*x*x+3*y*y*y*y)*sqrt(70)/16;
					*(ao_rangers[9]++)=a*z*(x*x*x*x-6*x*x*y*y+y*y*y*y)*3*sqrt(35)/8;
					*(ao_rangers[10]++)=a*x*(x*x*x*x-10*x*x*y*y+5*y*y*y*y)*3*sqrt(14)/16;
					break;
				case 13: // I
					*(ao_rangers[0]++)=a*x*y*(3*x*x*x*x-10*x*x*y*y+3*y*y*y*y)*sqrt(462)/16;
					*(ao_rangers[1]++)=a*y*z*(5*x*x*x*x-10*x*x*y*y+y*y*y*y)*3*sqrt(154)/16;
					*(ao_rangers[2]++)=a*x*y*(-x*x*x*x+y*y*y*y+10*x*x*z*z-10*y*y*z*z)*3*sqrt(7)/4;
					*(ao_rangers[3]++)=a*y*z*(-9*x*x*x*x+3*y*y*y*y-6*x*x*y*y+24*x*x*z*z-8*y*y*z*z)*sqrt(210)/16;
					*(ao_rangers[4]++)=a*x*y*(x*x*x*x+y*y*y*y+16*z*z*z*z+2*x*x*y*y-16*x*x*z*z-16*y*y*z*z)*sqrt(210)/16;
					*(ao_rangers[5]++)=a*y*z*(5*x*x*x*x+5*y*y*y*y+8*z*z*z*z+10*x*x*y*y-20*x*x*z*z-20*y*y*z*z)*sqrt(21)/8;
					*(ao_rangers[6]++)=a*(16*z*z*z*z*z*z-5*x*x*x*x*x*x-5*y*y*y*y*y*y-15*x*x*x*x*y*y+90*x*x*x*x*z*z+90*y*y*y*y*z*z-120*y*y*z*z*z*z-15*x*x*y*y*y*y-120*x*x*z*z*z*z+180*x*x*y*y*z*z)/16;
					*(ao_rangers[7]++)=a*x*z*(5*x*x*x*x+5*y*y*y*y+8*z*z*z*z+10*x*x*y*y-20*x*x*z*z-20*y*y*z*z)*sqrt(21)/8;
					*(ao_rangers[8]++)=a*(x*x*x*x*x*x-y*y*y*y*y*y+x*x*x*x*y*y-x*x*y*y*y*y-16*x*x*x*x*z*z+16*x*x*z*z*z*z+16*y*y*y*y*z*z-16*y*y*z*z*z*z)*sqrt(210)/32;
					*(ao_rangers[9]++)=a*x*z*(-3*x*x*x*x+6*x*x*y*y+8*x*x*z*z-24*y*y*z*z+9*y*y*y*y)*sqrt(210)/16;
					*(ao_rangers[10]++)=a*(-x*x*x*x*x*x+5*x*x*x*x*y*y+10*x*x*x*x*z*z+5*x*x*y*y*y*y+10*y*y*y*y*z*z-y*y*y*y*y*y-60*x*x*y*y*z*z)*3*sqrt(7)/16;
					*(ao_rangers[11]++)=a*x*z*(x*x*x*x-10*x*x*y*y+5*y*y*y*y)*3*sqrt(154)/16;
					*(ao_rangers[12]++)=a*(x*x*x*x*x*x-15*x*x*x*x*y*y+15*x*x*y*y*y*y-y*y*y*y*y*y)*sqrt(462)/32;
					break;
			}
		}
	}
}

void GetDensity(double * aos,long int ngrids,EigenMatrix D,double * ds){
	double * i_ranger=aos; // ibasis*ngrids+jgrid
	double * j_ranger=aos;
	double * d_ranger=ds;
	for (long int i=0;i<ngrids;i++)
		*(d_ranger++)=0;
	d_ranger=ds;
	double Dij;
	for (int ibasis=0;ibasis<D.cols();ibasis++){
		for (int jbasis=0;jbasis<ibasis;jbasis++){
			i_ranger=aos+ibasis*ngrids;
			j_ranger=aos+jbasis*ngrids;
			d_ranger=ds;
			Dij=D(ibasis,jbasis);
			for (long int igrid=0;igrid<ngrids;igrid++)
				*(d_ranger++)+=2*Dij**(i_ranger++)**(j_ranger++);
		}
		i_ranger=aos+ibasis*ngrids;
		j_ranger=i_ranger;
		d_ranger=ds;
		Dij=D(ibasis,ibasis);
		for (long int igrid=0;igrid<ngrids;igrid++)
			*(d_ranger++)+=Dij**(i_ranger++)**(j_ranger++);
	}
}

double SumUp(double * ds,double * weights,long int ngrids){
	double * d_ranger=ds;
	double * weight_ranger=weights;
	double n=0;
	for (long int i=0;i<ngrids;i++)
		n+=*(d_ranger++)**(weight_ranger++);
	return n;
}

EigenMatrix VxcMatrix(double * aos,double * weights,double * vrs,long int ngrids,int nbasis){
	double * iao_ranger;
	double * jao_ranger;
	double * weight_ranger;
	double * v_ranger;
	EigenMatrix vxc=EigenZero(nbasis,nbasis);
	double vij;
	for (int irow=0;irow<nbasis;irow++){
		for (int jcol=0;jcol<irow;jcol++){
			iao_ranger=aos+irow*ngrids;
			jao_ranger=aos+jcol*ngrids;
			weight_ranger=weights;
			v_ranger=vrs;
			vij=0;
			for (int kgrid=0;kgrid<ngrids;kgrid++)
				vij+=*(iao_ranger++)**(jao_ranger++)**(weight_ranger++)**(v_ranger++);
			vxc(irow,jcol)=vij;
			vxc(jcol,irow)=vij;
		}
	}
	return vxc;
}
