#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <functional>
#include <cassert>
#include <libmwfn.h>

#include "../Macro.h"
#include "Tensor.h"
#include "Grid.h"
#include "sphere_lebedev_rule.hpp"

int SphericalGridNumber(std::string path, std::vector<MwfnCenter>& centers){
	int ngrids = 0;
	__Z_2_Name__
	for ( MwfnCenter& center : centers ){
		int ngroups;
		std::ifstream gridfile( path + "/" + Z2Name[center.Index] + ".grid" );
		assert("Missing element grid file in folder!" && gridfile.good());
		std::string thisline;
		std::getline(gridfile, thisline);
		std::stringstream ss_(thisline);
		ss_ >> ngroups;
		std::getline(gridfile, thisline);
		for ( int igroup = 0; igroup < ngroups; igroup++ ){
			std::getline(gridfile, thisline);
			std::stringstream ss(thisline);
			int nshells, npoints;
			ss >> nshells; ss >> npoints;
			ngrids += nshells * npoints;
		}
	}
	return ngrids;
}

double s_p_mu(
		double x0, double y0, double z0,
		double x1, double y1, double z1,
		double x2, double y2, double z2){
	const double x01 = x0 - x1;
	const double y01 = y0 - y1;
	const double z01 = z0 - z1;
	const double r01 = std::sqrt( x01 * x01 + y01 * y01 + z01 * z01 );
	const double x02 = x0 - x2;
	const double y02 = y0 - y2;
	const double z02 = z0 - z2;
	const double r02 = std::sqrt( x02 * x02 + y02 * y02 + z02 * z02 );
	const double x12 = x1 - x2;
	const double y12 = y1 - y2;
	const double z12 = z1 - z2;
	const double r12 = std::sqrt( x12 * x12 + y12 * y12 + z12 * z12 );
	const double mu = ( r01 - r02 ) / r12;
	const double p1 = 1.5 * mu - 0.5 * mu * mu * mu;
	const double p2 = 1.5 * p1 - 0.5 * p1 * p1 * p1;
	const double p3 = 1.5 * p2 - 0.5 * p2 * p2 * p2;
	return 0.5 * ( 1 - p3 );
}

void SphericalGrid(
		std::string path, std::vector<MwfnCenter>& centers,
		double* xs, double* ys, double* zs, double* ws){
	double* x_ranger = xs;
	double* y_ranger = ys;
	double* z_ranger = zs;
	double* w_ranger = ws;
	__Z_2_Name__
	for ( MwfnCenter& centera : centers ){
		std::ifstream gridfile(path + "/" + Z2Name[centera.Index] + ".grid");
		assert("Missing element grid file in folder!" && gridfile.good());
		std::string thisline;
		std::getline(gridfile, thisline);
		std::stringstream ss_(thisline);
		int ngroups, nshells_total;
		ss_ >> ngroups; ss_ >> nshells_total;
		std::getline(gridfile, thisline); // Radial-formula-dependent line.
		std::stringstream ss__(thisline);
		std::string radialformula; ss__ >> radialformula;
		std::function<double(double)> ri_func;
		std::function<double(double)> radial_weight_func;
		if ( radialformula == "de2" ){
			double token1, token2, token3;
			ss__ >> token1; ss__ >> token2; ss__ >> token3;
			const double a = token1;
			const double h = ( token3 - token2 ) / (double)( nshells_total - 1 );
			ri_func = [=](double i){
				const double xi = h * i + token2;
				return std::exp( a * xi - std::exp( - xi ) );
			};
			radial_weight_func = [=](double i){
				const double xi = h * i + token2;
				return std::exp( 3 * a * xi - 3 * std::exp( - xi ) ) * ( a +std:: exp( - xi ) ) * h;
			};
		}else if ( radialformula == "em" ){
			double token1; ss__ >> token1;
			const double R = token1;
			ri_func = [=](double i){
				const double ii = i + 1;
				return R * std::pow( ii / ( nshells_total + 1 - ii ), 2);
			};
			radial_weight_func = [=](double i){
				const double ii = i + 1;
				return 2 * std::pow(R, 3) * ( nshells_total + 1 ) * std::pow(ii, 5) / std::pow(nshells_total + 1 - ii, 7);
			};
		}else assert("Unrecognized radial formula!" && 0);
		int ishell_total = 0;
		for ( int igroup = 0; igroup < ngroups; igroup++ ){
			std::getline(gridfile, thisline);
			std::stringstream ss(thisline);
			int npoints, nshells;
			ss >> npoints; ss >> nshells;
			double lebedev_xs[8192] = {0};
			double lebedev_ys[8192] = {0};
			double lebedev_zs[8192] = {0};
			double lebedev_ws[8192] = {0};
			ld_by_order(npoints, lebedev_xs, lebedev_ys, lebedev_zs, lebedev_ws);
			for ( int ishell = 0; ishell < nshells; ishell++, ishell_total++ ){
				const double ri = ri_func(ishell_total);
				const double radial_w = radial_weight_func(ishell_total);
				for ( int ipoint = 0; ipoint < npoints; ipoint++ ){
					const double x = lebedev_xs[ipoint] * ri + centera.Coordinates[0];
					const double y = lebedev_ys[ipoint] * ri + centera.Coordinates[1];
					const double z = lebedev_zs[ipoint] * ri + centera.Coordinates[2];
					const double lebedev_w = 4 * M_PI * lebedev_ws[ipoint];
					double unnorm_becke_w_total = 0;
					double unnorm_becke_w = 0;
					for ( MwfnCenter& centerb : centers ){
						double unnorm_becke_wj = 1;
						const double xj = centerb.Coordinates[0];
						const double yj = centerb.Coordinates[1];
						const double zj = centerb.Coordinates[2];
						for ( MwfnCenter& centerc : centers ){
							if ( &centerb == &centerc ) continue;
							const double xk = centerc.Coordinates[0];
							const double yk = centerc.Coordinates[1];
							const double zk = centerc.Coordinates[2];
							unnorm_becke_wj *= s_p_mu(x, y, z, xj, yj, zj, xk, yk, zk);
						}
						unnorm_becke_w_total += unnorm_becke_wj;
						if ( &centera == &centerb )
							unnorm_becke_w = unnorm_becke_wj;
					}
					const double becke_w = unnorm_becke_w / unnorm_becke_w_total;
					const double w = radial_w * lebedev_w * becke_w;
					*(x_ranger++) = x;
					*(y_ranger++) = y;
					*(z_ranger++) = z;
					*(w_ranger++) = w;
				}
			}
		}
	}
}

/*
#define __Uniform_Box_Grid_Number__\
	__Libint2_Atoms__\
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
*/

Grid::Grid(Mwfn* mwfn, std::string grid, int output){
	this->MWFN = mwfn;
	if ( grid.size() == 0 ) return;
	std::string path = std::getenv("CHINIUM_PATH");
	path += "/Grids/" + grid + "/";
	if (output) std::printf("Reading grid files in %s\n", path.c_str());

	this->NumGrids = SphericalGridNumber(path, mwfn->Centers);
	if (output) std::printf("Number of grid points ... %d\n", this->NumGrids);

	if (output) std::printf("Generating grid points and weights ... ");
	auto start = __now__;
	assert(this->Xs.size() == 0 && "Grid is already generated!");
	assert(this->Ys.size() == 0 && "Grid is already generated!");
	assert(this->Zs.size() == 0 && "Grid is already generated!");
	assert(this->Weights.size() == 0 && "Grid is already generated!");
	this->Xs.resize(this->NumGrids);
	this->Ys.resize(this->NumGrids);
	this->Zs.resize(this->NumGrids);
	this->Weights = Eigen::Tensor<double, 1>(this->NumGrids);
	SphericalGrid(
			path, mwfn->Centers,
			this->Xs.data(), this->Ys.data(),
			this->Zs.data(), this->Weights.data()
	);
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
}

template<int ndim> inline Eigen::Tensor<double, ndim> SliceGrid(
		const Eigen::Tensor<double, ndim>& tensor,
		int from, int length){
	Eigen::array<Eigen::Index, ndim> offsets; offsets[0] = from;
	Eigen::array<Eigen::Index, ndim> extents; extents[0] = length;
	for ( int i = 1; i < ndim; i++ ){
		offsets[i] = 0;
		extents[i] = tensor.dimension(i);
	}
	return tensor.slice(offsets, extents);
}

#define __Copy_Sliced_Grid__(tensor)\
	if ( grid.tensor.size() > 0 )\
		this->tensor = SliceGrid(grid.tensor, from, length);

Grid::Grid(Grid& grid, int from, int length, int output){
	if (output > 0) std::printf("Copying sub-grids %d - %d from grids ...\n", from, from + length - 1);
	assert(from + length - 1 < grid.NumGrids && "Invalid range!");

	this->MWFN = grid.MWFN;
	this->Type = grid.Type;

	this->NumGrids = length;
	this->Xs = std::vector<double>(&grid.Xs[from], &grid.Xs[from + length]);
	this->Ys = std::vector<double>(&grid.Ys[from], &grid.Ys[from + length]);
	this->Zs = std::vector<double>(&grid.Zs[from], &grid.Zs[from + length]);

	__Copy_Sliced_Grid__(Weights);
	__Copy_Sliced_Grid__(AOs);
	__Copy_Sliced_Grid__(AO1s);
	__Copy_Sliced_Grid__(AO2Ls);
	__Copy_Sliced_Grid__(AO2s);
	__Copy_Sliced_Grid__(AO3s);

	__Copy_Sliced_Grid__(Rhos);
	__Copy_Sliced_Grid__(Rho1s);
	__Copy_Sliced_Grid__(Sigmas);
	__Copy_Sliced_Grid__(Lapls);
	__Copy_Sliced_Grid__(Taus);

	__Copy_Sliced_Grid__(RhoGrads);
	__Copy_Sliced_Grid__(Rho1Grads);
	__Copy_Sliced_Grid__(SigmaGrads);

	__Copy_Sliced_Grid__(RhoHesss);
	__Copy_Sliced_Grid__(Rho1Hesss);
	__Copy_Sliced_Grid__(SigmaHesss);

	__Copy_Sliced_Grid__(Es);
	__Copy_Sliced_Grid__(E1Rhos);
	__Copy_Sliced_Grid__(E1Sigmas);
	__Copy_Sliced_Grid__(E1Lapls);
	__Copy_Sliced_Grid__(E1Taus);
	__Copy_Sliced_Grid__(E2Rho2s);
	__Copy_Sliced_Grid__(E2RhoSigmas);
	__Copy_Sliced_Grid__(E2Sigma2s);
}
