#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Maniverse/Manifold/Stiefel.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <array>
#include <tuple>
#include <functional>
#include <cassert>
#include <libmwfn.h>

#include "../Macro.h"
#include "Grid.h"
#include "sphere_lebedev_rule.hpp"

#define max_size 128
#define ao_threshold -10

int SphericalGridNumber(std::string path, std::vector<MwfnCenter>& centers){
	int ngrids = 0;
	__Z_2_Name__
	for ( MwfnCenter& center : centers ){
		int ngroups;
		std::ifstream gridfile( path + "/" + Z2Name[center.Index] + ".grid" );
		if ( !gridfile.good() ) throw std::runtime_error("Missing element grid file in folder!");
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
		double* data){
	__Z_2_Name__
	for ( MwfnCenter& centera : centers ){
		std::ifstream gridfile(path + "/" + Z2Name[centera.Index] + ".grid");
		if ( !gridfile.good() ) throw std::runtime_error("Missing element grid file in folder!");
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
			ri_func = [R, nshells_total](double i){
				const double ii = i + 1;
				return R * std::pow( ii / ( nshells_total + 1 - ii ), 2);
			};
			radial_weight_func = [R, nshells_total](double i){
				const double ii = i + 1;
				return 2 * std::pow(R, 3) * ( nshells_total + 1 ) * std::pow(ii, 5) / std::pow(nshells_total + 1 - ii, 7);
			};
		}else throw std::runtime_error("Unrecognized radial formula!");
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
					*(data++) = x;
					*(data++) = y;
					*(data++) = z;
					*(data++) = w;
				}
			}
		}
	}
}

class CutFunc: public Maniverse::Objective{ public:
	EigenMatrix P2;
	CutFunc(EigenMatrix P2): P2(P2){};
	void Calculate(std::vector<EigenMatrix> Ws, std::vector<int> /*derivatives*/) override{
		const EigenMatrix& W = Ws[0];
		Value = - ( W.transpose() * P2 * W ).sum();
		Gradient = { - 2 * P2 * W };
	};
	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> Vs) const override{
		return std::vector<EigenMatrix>{ - 2 * P2 * Vs[0] };
	};
};

std::tuple<EigenMatrix, EigenMatrix> Cut(EigenMatrix P){
	EigenMatrix Pcentered = P.topRows(3);
	std::array<double, 3> center = {0, 0, 0};
	for ( int i = 0; i < 3; i++ ){
		center[i] = P.row(i).mean();
		Pcentered.row(i).array() -= center[i];
	}
	CutFunc obj(Pcentered * Pcentered.transpose());
	Maniverse::Stiefel stiefel(EigenMatrix::Ones(3, 1) / std::sqrt(3));
	Maniverse::Iterate M(obj, {stiefel.Share()}, 1);
	Maniverse::TrustRegion tr;
	Maniverse::TruncatedNewton(
			M, tr, {1.e-6, 1, 1},
			0.001, 100, 0
	);
	const double A = M.Point(0);
	const double B = M.Point(1);
	const double C = M.Point(2);
	const double D = - A * center[0] - B * center[1] - C * center[2];
	auto* points = reinterpret_cast<std::array<double, 4>*>(P.data());
	std::partition(points, points + P.rows(), [A, B, C, D](const auto& X){
		return A * X[0] + B * X[1] + C * X[2] + D > 0;
	});
	return std::make_tuple(P.leftCols(P.cols() / 2).eval(), P.rightCols(P.cols() - P.cols() / 2).eval());
}

void RecursiveCut(EigenMatrix parent, std::vector<EigenMatrix>& all_batches){
	if ( parent.cols() <= max_size ) all_batches.push_back(parent);
	else{
		auto [child1, child2] = Cut(parent);
		RecursiveCut(child1, all_batches);
		RecursiveCut(child2, all_batches);
	}
}

SubGrid::SubGrid(EigenMatrix P){
	const int ngrids = P.cols();
	const int new_ngrids = ( ngrids + 7 ) & ~7;
	const EigenVector x_vec = P.row(0);
	const EigenVector y_vec = P.row(1);
	const EigenVector z_vec = P.row(2);
	const EigenVector w_vec = P.row(3);
	this->X.assign(new_ngrids, 0);
	this->Y.assign(new_ngrids, 0);
	this->Z.assign(new_ngrids, 0);
	this->W.resize(new_ngrids); this->W.setZero();
	std::memcpy(X.data(), x_vec.data(), ngrids * 8);
	std::memcpy(Y.data(), y_vec.data(), ngrids * 8);
	std::memcpy(Z.data(), z_vec.data(), ngrids * 8);
	std::memcpy(W.data(), w_vec.data(), ngrids * 8);
	std::memcpy(X.data() + ngrids, X.data(), ( new_ngrids - ngrids ) * 8);
	std::memcpy(Y.data() + ngrids, Y.data(), ( new_ngrids - ngrids ) * 8);
	std::memcpy(Z.data() + ngrids, Z.data(), ( new_ngrids - ngrids ) * 8);
}

Grid::Grid(Mwfn* mwfn, std::string grid, int nthreads, int output){
	if ( grid.size() == 0 ) return;
	std::string path = std::getenv("CHINIUM_PATH");
	path += "/Grids/" + grid + "/";
	if (output) std::printf("Reading grid files in %s\n", path.c_str());

	const int ngrids = SphericalGridNumber(path, mwfn->Centers);
	if (output) std::printf("Number of grid points ... %d\n", ngrids);

	if (output) std::printf("Generating grid points and weights ... ");
	auto start = __now__;
	EigenMatrix P = EigenZero(4, ngrids);
	SphericalGrid(path, mwfn->Centers, P.data());
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));

	if (output) std::printf("Clutersing grid points ... ");
	start = __now__;
	std::vector<EigenMatrix> batches;
	RecursiveCut(P, batches);
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
	if (output) std::printf("The grid points are clustered into %d groups.\n", (int)batches.size());

	// First touch allocation of subgrids among OMP threads
	std::vector<SubGrid> subgrids; subgrids.resize(batches.size());
	std::vector<int> complexity; complexity.resize(batches.size());
	double total = 0;
	for ( int i = 0; i < (int)subgrids.size(); i++ ){
		SubGrid& subgrid = subgrids[i] = SubGrid(batches[i]);
		subgrid.MWFN = mwfn;
		subgrid.Spin = mwfn->getNumSpins();
		subgrid.NumGrids = subgrid.W.dimension(0);
		subgrid.BasisList.resize(mwfn->getNumBasis()); for ( int k = 0; k < mwfn->getNumBasis(); k++ ) subgrid.BasisList[k] = k;
		subgrid.getAO(0);
		subgrid.BasisList.resize(0);
		for ( int mu = 0; mu < mwfn->getNumBasis(); mu++ ){
			const EigenTensor<0> max = subgrid.AO.chip(mu, 1).abs().maximum();
			if ( max() > ao_threshold ){
				subgrid.BasisList.push_back(mu);
			}
		}
		subgrid.AO.resize(0, 0);
		complexity[i] = subgrid.NumGrids * subgrid.BasisList.size() * subgrid.BasisList.size();
		total += complexity[i];

		std::vector<int> basis2atom_tot = mwfn->Basis2Atom();
		std::vector<int> basis2atom; basis2atom.reserve(subgrid.getNumBasis());
		for ( int basis : subgrid.BasisList ){
			basis2atom.push_back(basis2atom_tot[basis]);
		}
		subgrid.AtomList = basis2atom;
		auto last_unique = std::unique(subgrid.AtomList.begin(), subgrid.AtomList.end());
		subgrid.AtomList.erase(last_unique, subgrid.AtomList.end());
		int nbasis = 0;
		for ( int atom : subgrid.AtomList ){
			subgrid.AtomHeads.push_back(nbasis);
			const int length = std::count(basis2atom.begin(), basis2atom.end(), atom);
			nbasis += length;
			subgrid.AtomLengths.push_back(length);
		}
	}

	std::vector<std::vector<SubGrid>> bins; bins.resize(nthreads);
	int kbin = 0;
	double current_complexity = 0;
	for ( int i = 0; i < (int)subgrids.size(); i++ ){
		bins[kbin].push_back(std::move(subgrids[i]));
		current_complexity += complexity[i];
		if ( current_complexity > ( kbin + 1 ) * total / nthreads ) kbin++;
	}

	this->SubGridBatches.resize(nthreads);
	#pragma omp parallel for schedule(static) num_threads(nthreads)
	for ( int kthread = 0; kthread < nthreads; kthread++ ){
		this->SubGridBatches[kthread].reserve(bins[kthread].size());
		for ( SubGrid& subgrid : bins[kthread] ){
			this->SubGridBatches[kthread].push_back(std::make_unique<SubGrid>(subgrid));
		}
	}
}
