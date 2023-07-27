#include <Eigen/Dense>
#include <libint2.hpp>
#include <cmath>
#include "Aliases.h"
#include "Libint2.h"
#include <iostream>

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

void GetAoValues(const int natoms,double * atoms,const char * basisset,double * xs,double * ys,double * zs,long int ngrids,double * aos){ // ibasis*ngrids+jgrid
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
				case 5: // D rescaled as suggested @ https://github.com/evaleev/libint/wiki/using-modern-CPlusPlus-API
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
					*(ao_rangers[0]++)=a*x*y*(x*x-y*y);
					*(ao_rangers[1]++)=a*y*(3*x*x-y*y)*z;
					*(ao_rangers[2]++)=a*x*y*(7*z*z-r2);
					*(ao_rangers[3]++)=a*y*(7*z*z*z-3*z*r2);
					*(ao_rangers[4]++)=a*(35*z*z*z*z-30*z*z*r2+3*r2*r2);
					*(ao_rangers[5]++)=a*x*(7*z*z*z-3*z*r2);
					*(ao_rangers[6]++)=a*(x*x-y*y)*(7*z*z-r2);
					*(ao_rangers[7]++)=a*x*(x*x-3*y*y)*z;
					*(ao_rangers[8]++)=a*(x*x*(x*x-3*y*y)-y*y*(3*x*x-y*y));
					break;
				case 11: // H
					break;
				case 13: // I
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

double GetNumElectrons(double * ds,long int ngrids,double spacing){
	double * d_ranger=ds;
	double n=0;
	for (long int i=0;i<ngrids;i++)
		n+=*(d_ranger++);
	return n*2*spacing*spacing*spacing;
}

