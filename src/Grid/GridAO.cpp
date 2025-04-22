#include <Eigen/Core>
#include <cmath>
#include <vector>
#include <array>
#include <chrono>
#include <cstdio>
#include <string>

#include "../Macro.h"
#include "../Multiwfn/Multiwfn.h"


void GetAoValues(
		std::vector<MwfnCenter>& centers,
		double* xs, double* ys, double* zs, long int ngrids,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		double* ao2xxs, double* ao2yys, double* ao2zzs,
		double* ao2xys, double* ao2xzs, double* ao2yzs,
		double* ao3xxxs, double* ao3xxys, double* ao3xxzs,
		double* ao3xyys, double* ao3xyzs, double* ao3xzzs,
		double* ao3yyys, double* ao3yyzs, double* ao3yzzs, double* ao3zzzs){ // ibasis*ngrids+jgrid
	double xo, yo, zo, x, y, z, r2;
	double tmp, tmp0, tmp1, tmp2, tmp3;
	auto [A, Ax, Ay, Az, Axx, Ayy, Azz, Axy, Axz, Ayz, Axxx, Axxy, Axxz, Axyy, Axyz, Axzz, Ayyy, Ayyz, Ayzz, Azzz] = std::array<double, 20>{}; // Basis function values
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
	double* ao3xxx_rangers[16];
	double* ao3xxy_rangers[16];
	double* ao3xxz_rangers[16];
	double* ao3xyy_rangers[16];
	double* ao3xyz_rangers[16];
	double* ao3xzz_rangers[16];
	double* ao3yyy_rangers[16];
	double* ao3yyz_rangers[16];
	double* ao3yzz_rangers[16];
	double* ao3zzz_rangers[16];
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
	double Qxxx[16] = {0};
	double Qxxy[16] = {0};
	double Qxxz[16] = {0};
	double Qxyy[16] = {0};
	double Qxyz[16] = {0};
	double Qxzz[16] = {0};
	double Qyyy[16] = {0};
	double Qyyz[16] = {0};
	double Qyzz[16] = {0};
	double Qzzz[16] = {0};
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
					if (ao3xxxs){
						ao3xxx_rangers[iranger] = ao3xxxs + ibasis * ngrids;
						ao3xxy_rangers[iranger] = ao3xxys + ibasis * ngrids;
						ao3xxz_rangers[iranger] = ao3xxzs + ibasis * ngrids;
						ao3xyy_rangers[iranger] = ao3xyys + ibasis * ngrids;
						ao3xyz_rangers[iranger] = ao3xyzs + ibasis * ngrids;
						ao3xzz_rangers[iranger] = ao3xzzs + ibasis * ngrids;
						ao3yyy_rangers[iranger] = ao3yyys + ibasis * ngrids;
						ao3yyz_rangers[iranger] = ao3yyzs + ibasis * ngrids;
						ao3yzz_rangers[iranger] = ao3yzzs + ibasis * ngrids;
						ao3zzz_rangers[iranger] = ao3zzzs + ibasis * ngrids;
					}
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
			tmp = tmp0 = tmp1 = tmp2 = tmp3 = 0;
			for ( int iprim = 0; iprim < nprims; ++iprim ){ // Looping over primitive gaussians.
				tmp = shell.NormalizedCoefficients[iprim] * std::exp(- shell.Exponents[iprim] * r2);
				tmp0 += tmp; // Each primitive gaussian contributes to the radial parts of basis functions.
				if (ao1xs){
					tmp1 += tmp * shell.Exponents[iprim];
					if (ao2ls || ao2xxs){
						tmp2 += tmp * shell.Exponents[iprim] * shell.Exponents[iprim];
						if (ao3xxxs)
							tmp3 += tmp * shell.Exponents[iprim] * shell.Exponents[iprim] * shell.Exponents[iprim];
					}
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
						if (ao3xxxs){
							Axxx = 12 * x * tmp2 - 8 * x * x * x * tmp3;
							Axxy = 4 * y * tmp2 - 8 * x * x * y * tmp3;
							Axxz = 4 * z * tmp2 - 8 * x * x * z * tmp3;
							Axyy = 4 * tmp2 * x - 8 * y * tmp3 * x * y;
							Axyz = - 8 * tmp3 * x * y * z;
							Axzz = 4 * tmp2 * x - 8 * z * tmp3 * x * z;
							Ayyy = 12 * y * tmp2 - 8 * y * y * y * tmp3;
							Ayyz = 4 * z * tmp2 - 8 * y * y * z * tmp3;
							Ayzz = 4 * tmp2 * y - 8 * z * tmp3 * y * z;
							Azzz = 12 * z * tmp2 - 8 * z * z * z * tmp3;
						}
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
							if (ao3xxxs){
								ao3xxx_rangers[i][k] = Axxx * Q[i] + 3 * Axx * Qx[i] + 3 * Ax * Qxx[i] + A * Qxxx[i];
								ao3xxy_rangers[i][k] = Axxy * Q[i] + 2 * Axy * Qx[i] + Axx * Qy[i] + 2 * Ax * Qxy[i] + Ay * Qxx[i] + A * Qxxy[i];
								ao3xxz_rangers[i][k] = Axxz * Q[i] + 2 * Axz * Qx[i] + Axx * Qz[i] + 2 * Ax * Qxz[i] + Az * Qxx[i] + A * Qxxz[i];
								ao3xyy_rangers[i][k] = Axyy * Q[i] + 2 * Axy * Qy[i] + Ayy * Qx[i] + Ax * Qyy[i] + 2 * Ay * Qxy[i] + A * Qxyy[i];
								ao3xyz_rangers[i][k] = Axyz * Q[i] + Axy * Qz[i] + Axz * Qy[i] + Ayz * Qx[i] + Ax * Qyz[i] + Ay * Qxz[i] + Az * Qxy[i] + A * Qxyz[i];
								ao3xzz_rangers[i][k] = Axzz * Q[i] + 2 * Axz * Qz[i] + Azz * Qx[i] + Ax * Qzz[i] + 2 * Az * Qxz[i] + A * Qxzz[i];
								ao3yyy_rangers[i][k] = Ayyy * Q[i] + 3 * Ayy * Qy[i] + 3 * Ay * Qyy[i] + A * Qyyy[i];
								ao3yyz_rangers[i][k] = Ayyz * Q[i] + 2 * Ayz * Qy[i] + Ayy * Qz[i] + 2 * Ay * Qyz[i] + Az * Qyy[i] + A * Qyyz[i];
								ao3yzz_rangers[i][k] = Ayzz * Q[i] + 2 * Ayz * Qz[i] + Azz * Qy[i] + Ay * Qzz[i] + 2 * Az * Qyz[i] + A * Qyzz[i];
								ao3zzz_rangers[i][k] = Azzz * Q[i] + 3 * Azz * Qz[i] + 3 * Az * Qzz[i] + A * Qzzz[i];
							}
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
	if ( order >= 3 ){
		this->AO3XXXs = new double[this->NumGrids * this->getNumBasis()];
		this->AO3XXYs = new double[this->NumGrids * this->getNumBasis()];
		this->AO3XXZs = new double[this->NumGrids * this->getNumBasis()];
		this->AO3XYYs = new double[this->NumGrids * this->getNumBasis()];
		this->AO3XYZs = new double[this->NumGrids * this->getNumBasis()];
		this->AO3XZZs = new double[this->NumGrids * this->getNumBasis()];
		this->AO3YYYs = new double[this->NumGrids * this->getNumBasis()];
		this->AO3YYZs = new double[this->NumGrids * this->getNumBasis()];
		this->AO3YZZs = new double[this->NumGrids * this->getNumBasis()];
		this->AO3ZZZs = new double[this->NumGrids * this->getNumBasis()];
	}
	GetAoValues(
			this->Centers,
			this->Xs, this->Ys, this->Zs, this->NumGrids,
			this->AOs,
			this->AO1Xs, this->AO1Ys, this->AO1Zs,
			this->AO2Ls,
			this->AO2XXs, this->AO2YYs, this->AO2ZZs,
			this->AO2XYs, this->AO2XZs, this->AO2YZs,
			this->AO3XXXs, this->AO3XXYs, this->AO3XXZs,
			this->AO3XYYs, this->AO3XYZs, this->AO3XZZs,
			this->AO3YYYs, this->AO3YYZs, this->AO3YZZs, this->AO3ZZZs
	);
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
}
