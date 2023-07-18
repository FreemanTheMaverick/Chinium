#include <Eigen/Dense>
#include <libint2.hpp>
#include <cmath>
#include "Aliases.h"
#include "Libint2.h"
#include <iostream>

void CalculateDensity(double * xxx,double * yyy,double * zzz,long int size,EigenMatrix D,libint2::BasisSet obs,double * results){
	const int nbasis=nBasis_from_obs(obs);
	const auto shell2bf=obs.shell2bf();
	double x,y,z,xx,yy,zz,r2,a,result,phi[8192]; // Basis function values;
	int two_l_plus_one,nprims; // These in-loop variables had better be declared outside the loop. Tests showed that the codes as such are faster than those otherwise.
	EigenVector Di_times_phii;
	/*libint2::Shell shell;
	libint2::Shell::Contraction contraction; Do not use these codes and those marked with "SLOW" below, which slow down the iterations! */
	double * x_ranger=xxx;
	double * y_ranger=yyy;
	double * z_ranger=zzz;
	double * phi_ranger;
	double * result_ranger=results;
	for (long int k=0;k<size;++k){
		xx=*(x_ranger++);
		yy=*(y_ranger++);
		zz=*(z_ranger++);
		result=0;
		phi_ranger=phi;
		for (const libint2::Shell & shell:obs){
		/*for (int s=0;s<obs.size();s++){ // SLOW
			shell=obs[s];
			contraction=shell.contr[0]; // Do not use these codes, which slow down the iterations! */
			two_l_plus_one=shell.size();
			nprims=shell.nprim();
			x=xx-shell.O[0]; // Coordintes with the origin at the centre of the shell.
			y=yy-shell.O[1];
			z=zz-shell.O[2];
			r2=x*x+y*y+z*z;
			a=0;
			for (int iprim=0;iprim<nprims;++iprim) // Looping over primitive gaussians.
				a+=shell.contr[0].coeff[iprim]*std::exp(-shell.alpha[iprim]*r2); // Each primitive gaussian contributes to the radial parts of basis functions.
			switch(two_l_plus_one){ // Basis function values are the products of the radial parts and the spherical harmonic parts.
				case 1: // S
					*(phi_ranger++)=a;
					break;
				case 3: // P
					*(phi_ranger++)=a*x;
					*(phi_ranger++)=a*y;
					*(phi_ranger++)=a*z;
					break;
				case 5: // D rescaled as suggested @ https://github.com/evaleev/libint/wiki/using-modern-CPlusPlus-API
					*(phi_ranger++)=a*x*y*sqrt(3);
					*(phi_ranger++)=a*y*z*sqrt(3);
					*(phi_ranger++)=a*(3*z*z-r2)*0.5;
					*(phi_ranger++)=a*x*z*sqrt(3);
					*(phi_ranger++)=a*(x*x-y*y)*sqrt(3)*0.5;
					break;
				case 7: // F
					*(phi_ranger++)=a*y*(3*x*x-y*y)*sqrt(10)/4;
					*(phi_ranger++)=a*x*y*z*sqrt(15);
					*(phi_ranger++)=a*y*(5*z*z-r2)*sqrt(6)/4;
					*(phi_ranger++)=a*(5*z*z*z-3*z*r2)/2;
					*(phi_ranger++)=a*x*(5*z*z-r2)*sqrt(6)/4;
					*(phi_ranger++)=a*(x*x-y*y)*z*sqrt(15)/2;
					*(phi_ranger++)=a*x*(x*x-3*y*y)*sqrt(10)/4;
					break;
				case 9: // G
					*(phi_ranger++)=a*x*y*(x*x-y*y);
					*(phi_ranger++)=a*y*(3*x*x-y*y)*z;
					*(phi_ranger++)=a*x*y*(7*z*z-r2);
					*(phi_ranger++)=a*y*(7*z*z*z-3*z*r2);
					*(phi_ranger++)=a*(35*z*z*z*z-30*z*z*r2+3*r2*r2);
					*(phi_ranger++)=a*x*(7*z*z*z-3*z*r2);
					*(phi_ranger++)=a*(x*x-y*y)*(7*z*z-r2);
					*(phi_ranger++)=a*x*(x*x-3*y*y)*z;
					*(phi_ranger++)=a*(x*x*(x*x-3*y*y)-y*y*(3*x*x-y*y));
					break;
				case 11: // H
					break;
				case 13: // I
					break;
			}
		}
		for (int i=0;i<nbasis;i++){
			Di_times_phii=D.row(i)*phi[i];
			for (int j=0;j<i;j++)
				result+=Di_times_phii(j)*phi[j]*2;
			result+=Di_times_phii(i)*phi[i];
		}
		*(result_ranger++)=result;
	}
}

[[deprecated("This function is for test only.")]]
double UniformBoxGridDensity(const int natoms,double * atoms,const char * basisset,EigenMatrix D,double overheadlength,double griddensity){
	const std::vector<libint2::Atom> libint2atoms=Libint2Atoms(natoms,atoms);
	const libint2::BasisSet obs(basisset,libint2atoms);
	double xu=-10000;
	double yu=-10000;
	double zu=-10000;
	double xl=10000;
	double yl=10000;
	double zl=10000;
	for (const auto& shell:obs){
		const double x=shell.O[0];
		const double y=shell.O[1];
		const double z=shell.O[2];
		xu=xu>x?xu:x;
		yu=yu>y?yu:y;
		zu=zu>z?zu:z;
		xl=xl<x?xl:x;
		yl=yl<y?yl:y;
		zl=zl<z?zl:z;
	}
	xu+=overheadlength;
	yu+=overheadlength;
	zu+=overheadlength;
	xl-=overheadlength;
	yl-=overheadlength;
	zl-=overheadlength;
	const double xlength=xu-xl;
	const double ylength=yu-yl;
	const double zlength=zu-zl;
	const int nx=int(xlength*griddensity)+1;
	const int ny=int(ylength*griddensity)+1;
	const int nz=int(zlength*griddensity)+1;
	double * xx=new double[nx*ny*nz];double * x_ranger=xx;
	double * yy=new double[nx*ny*nz];double * y_ranger=yy;
	double * zz=new double[nx*ny*nz];double * z_ranger=zz;
	double * results=new double[nx*ny*nz];double * result_ranger=results;
	double x,y,z;
	for (int ix=0;ix<nx;ix++){
		x=xl+ix/griddensity;
		for (int iy=0;iy<ny;iy++){
			y=yl+iy/griddensity;
			for (int iz=0;iz<nz;iz++){
				z=zl+iz/griddensity;
				*(x_ranger++)=x;
				*(y_ranger++)=y;
				*(z_ranger++)=z;
			}
		}
	}
	CalculateDensity(xx,yy,zz,nx*ny*nz,D,obs,results);
	double nele=0;
	for (long int i=0;i<nx*ny*nz;i++)
		nele+=*(result_ranger++);
	nele*=2./griddensity/griddensity/griddensity;
	delete [] xx;
	delete [] yy;
	delete [] zz;
	delete [] results;
	return nele;
}
