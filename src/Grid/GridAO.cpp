#include <Eigen/Core>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdio>
#include <string>

#include "../Macro.h"
#include "../Multiwfn.h"

#include <iostream>


void GetAoValues(
		std::vector<MwfnCenter>& centers,
		double* xs, double* ys, double* zs, long int ngrids,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		double* ao2xxs, double* ao2yys, double* ao2zzs,
		double* ao2xys, double* ao2xzs, double* ao2yzs){ // ibasis*ngrids+jgrid
	double xo, yo, zo, x, y, z, r2;
	double tmp, tmp0, tmp1, tmp2;
	double A, Ax, Ay, Az, Axx, Ayy, Azz, Axy, Axz, Ayz; // Basis function values;
	A = Ax = Ay = Az = Axx = Ayy = Azz = Axy = Axz = Ayz = 0;
	double ao2xx = 0; double ao2yy = 0; double ao2zz = 0;
	double* ao_rangers[16];
	double* ao1x_rangers[16];
	double* ao1y_rangers[16];
	double* ao1z_rangers[16];
	double* ao2l_rangers[16];
	double* ao2xx_rangers[16];
	double* ao2yy_rangers[16];
	double* ao2zz_rangers[16];
	double* ao2xy_rangers[16];
	double* ao2xz_rangers[16];
	double* ao2yz_rangers[16];
	double Q[16] = {0};
	double Qx[16] = {0};
	double Qy[16] = {0};
	double Qz[16] = {0};
	double Qxx[16] = {0};
	double Qyy[16] = {0};
	double Qzz[16] = {0};
	double Qxy[16] = {0};
	double Qxz[16] = {0};
	double Qyz[16] = {0};
	int shellsize, nprims; // These in-loop variables had better be declared outside the loop. Tests showed that the codes as such are faster than those otherwise.
	int ibasis = 0;
	for ( MwfnCenter& center : centers ) for ( MwfnShell& shell : center.Shells ){
		shellsize = shell.getSize();
		nprims = shell.getNumPrims();
		for ( int iranger = 0; iranger < shellsize; iranger++, ibasis++ ){
			ao_rangers[iranger] = aos + ibasis * ngrids;
			if (ao1xs){
				ao1x_rangers[iranger] = ao1xs + ibasis * ngrids;
				ao1y_rangers[iranger] = ao1ys + ibasis * ngrids;
				ao1z_rangers[iranger] = ao1zs + ibasis * ngrids;
				if (ao2ls) ao2l_rangers[iranger] = ao2ls + ibasis * ngrids;
				if (ao2xxs){
					ao2xx_rangers[iranger] = ao2xxs + ibasis * ngrids;
					ao2yy_rangers[iranger] = ao2yys + ibasis * ngrids;
					ao2zz_rangers[iranger] = ao2zzs + ibasis * ngrids;
					ao2xy_rangers[iranger] = ao2xys + ibasis * ngrids;
					ao2xz_rangers[iranger] = ao2xzs + ibasis * ngrids;
					ao2yz_rangers[iranger] = ao2yzs + ibasis * ngrids;
				}
			}
		}
		xo = center.Coordinates[0];
		yo = center.Coordinates[1];
		zo = center.Coordinates[2];
		for ( int k = 0; k < ngrids; ++k ){
			x = xs[k] - xo; // Coordinates with the origin at the centre of the shell.
			y = ys[k] - yo;
			z = zs[k] - zo;
			r2 = x * x + y * y + z * z;
			tmp = tmp0 = tmp1 = tmp2 = 0;
			for ( int iprim = 0; iprim < nprims; ++iprim ){ // Looping over primitive gaussians.
				tmp = shell.NormalizedCoefficients[iprim] * std::exp(- shell.Exponents[iprim] * r2);
				tmp0 += tmp; // Each primitive gaussian contributes to the radial parts of basis functions.
				if (ao1xs){
					tmp1 += tmp * shell.Exponents[iprim];
					if (ao2ls || ao2xxs)
						tmp2 += tmp * shell.Exponents[iprim] * shell.Exponents[iprim];
				}
			}
			A = tmp0;
			if (ao1xs){
				Ax = - 2 * x * tmp1;
				Ay = - 2 * y * tmp1;
				Az = - 2 * z * tmp1;
				if (ao2ls || ao2xxs){
					Axx = - 2 * tmp1 + 4 * x * x * tmp2;
					Ayy = - 2 * tmp1 + 4 * y * y * tmp2;
					Azz = - 2 * tmp1 + 4 * z * z * tmp2;
					if (ao2xxs){
						Axy = 4 * tmp2 * x * y;
						Axz = 4 * tmp2 * x * z;
						Ayz = 4 * tmp2 * y * z;
					}
				}
			}
			switch(shell.Type){ // Basis function values are the products of the radial parts and the spherical harmonic parts. The following spherical harmonics are rescaled as suggested @ https://github.com/evaleev/libint/wiki/using-modern-CPlusPlus-API. The normalization constants here differ from those in "Symmetries of Spherical Harmonics: applications to ambisonics" @ https://iaem.at/ambisonics/symposium2009/proceedings/ambisym09-chapman-shsymmetries.pdf/@@download/file/AmbiSym09_Chapman_SHSymmetries.pdf by a factor of sqrt(2l+1).
				case 0:
					#include "S"
					break;
				case 1:
					#include "CartP"
					break;
				case -1:
					#include "PureP"
					break;
				case -2:
					#include "PureD"
					break;
				case -3:
					#include "PureF"
					break;
				case -4:
					#include "PureG"
					break;
				case -5:
					#include "PureH"
					break;
				case -6:
					#include "PureI"
					break;
			}
			for ( int i = 0; i < shell.getSize(); i++ ){
				ao_rangers[i][k] = A * Q[i];
				if (ao1xs){
					ao1x_rangers[i][k] = Ax * Q[i] + A * Qx[i];
					ao1y_rangers[i][k] = Ay * Q[i] + A * Qy[i];
					ao1z_rangers[i][k] = Az * Q[i] + A * Qz[i];
					if (ao2ls || ao2xxs){
						ao2xx = Axx * Q[i] + 2 * Ax * Qx[i] + A * Qxx[i];
						ao2yy = Ayy * Q[i] + 2 * Ay * Qy[i] + A * Qyy[i];
						ao2zz = Azz * Q[i] + 2 * Az * Qz[i] + A * Qzz[i];
						if (ao2ls) ao2l_rangers[i][k] = ao2xx + ao2yy + ao2zz;
						if (ao2xxs){
							ao2xx_rangers[i][k] = ao2xx;
							ao2yy_rangers[i][k] = ao2yy;
							ao2zz_rangers[i][k] = ao2zz;
							ao2xy_rangers[i][k] = Axy * Q[i] + Ax * Qy[i] + Ay * Qx[i] + A * Qxy[i];
							ao2xz_rangers[i][k] = Axz * Q[i] + Ax * Qz[i] + Az * Qx[i] + A * Qxz[i];
							ao2yz_rangers[i][k] = Ayz * Q[i] + Ay * Qz[i] + Az * Qy[i] + A * Qyz[i];
						}
					}
				}
			}
		}
	}
}

void Multiwfn::getGridAO(int order, int output){
	if (output) std::printf("Generating grids to order %d of basis functions ... ", order);
	auto start = __now__;
	assert(this->Xs && "Grid is not generated yet!");
	assert(this->Ys && "Grid is not generated yet!");
	assert(this->Zs && "Grid is not generated yet!");
	if ( order >= 0 )
		this->AOs = new double[this->NumGrids * this->getNumBasis()];
	if ( order >= 1 ){
		this->AO1Xs = new double[this->NumGrids * this->getNumBasis()];
		this->AO1Ys = new double[this->NumGrids * this->getNumBasis()];
		this->AO1Zs = new double[this->NumGrids * this->getNumBasis()];
	}
	if ( order >= 2 ){
		this->AO2Ls = new double[this->NumGrids * this->getNumBasis()];
		this->AO2XXs = new double[this->NumGrids * this->getNumBasis()];
		this->AO2YYs = new double[this->NumGrids * this->getNumBasis()];
		this->AO2ZZs = new double[this->NumGrids * this->getNumBasis()];
		this->AO2XYs = new double[this->NumGrids * this->getNumBasis()];
		this->AO2XZs = new double[this->NumGrids * this->getNumBasis()];
		this->AO2YZs = new double[this->NumGrids * this->getNumBasis()];
	}
	GetAoValues(
			this->Centers,
			this->Xs, this->Ys, this->Zs, this->NumGrids,
			this->AOs,
			this->AO1Xs, this->AO1Ys, this->AO1Zs,
			this->AO2Ls,
			this->AO2XXs, this->AO2YYs, this->AO2ZZs,
			this->AO2XYs, this->AO2XZs, this->AO2YZs
	);
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
}
