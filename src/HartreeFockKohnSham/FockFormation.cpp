#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <omp.h>

#include "../Macro.h"
#include "../Multiwfn/Multiwfn.h"
#include "../Grid/GridDensity.h"
#include "../Grid/GridPotential.h"

#include <iostream>

#define __Check_and_Zero__(array, multiple)\
	assert(array && " array is not allocated!");\
	std::memset(array, 0, ngrids * multiple * sizeof(double));

std::tuple<double, EigenMatrix> Gxc(
		ExchangeCorrelation& xc,
		double* ws, long int ngrids, int nbasis,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		std::vector<EigenMatrix> Ds,
		double* rhos,
		double* rho1xs, double* rho1ys, double* rho1zs, double* sigmas,
		double* lapls, double* taus,
		double* es,
		double* erhos, double* esigmas,
		double* elapls, double* etaus,
		int nthreads){

	// Step 1: Checking AOs
	if ( xc.XCfamily.compare("LDA") == 0 ){
		assert(aos && "AOs on grids do not exist!");
	}else if ( xc.XCfamily.compare("GGA") == 0 ){
		assert(aos && "AOs on grids do not exist!");
		assert(ao1xs && "First order x-derivatives of AOs on grids do not exist!");
		assert(ao1ys && "First order y-derivatives of AOs on grids do not exist!");
		assert(ao1zs && "First order z-derivatives of AOs on grids do not exist!");
	}else if ( xc.XCfamily.compare("mGGA") == 0 ){
		assert(aos && "AOs on grids do not exist!");
		assert(ao1xs && "First order x-derivatives of AOs on grids do not exist!");
		assert(ao1ys && "First order y-derivatives of AOs on grids do not exist!");
		assert(ao1zs && "First order z-derivatives of AOs on grids do not exist!");
		assert(ao2ls && "Laplacians of AOs on grids do not exist!");
	}

	// Step 2: Calculating density
	std::vector<int> dorders = {};
	if ( xc.XCfamily.compare("LDA") == 0 ){
		dorders = {0};
		__Check_and_Zero__(rhos, Ds.size());
	}else if ( xc.XCfamily.compare("GGA") == 0 ){
		dorders = {0, 1};
		__Check_and_Zero__(rhos, Ds.size());
		__Check_and_Zero__(rho1xs, Ds.size());
		__Check_and_Zero__(rho1ys, Ds.size());
		__Check_and_Zero__(rho1zs, Ds.size());
		__Check_and_Zero__(sigmas, 2 * Ds.size() - 1);
	}else if ( xc.XCfamily.compare("mGGA") == 0 ){
		dorders = {0, 1, 2};
		__Check_and_Zero__(rhos, Ds.size());
		__Check_and_Zero__(rho1xs, Ds.size());
		__Check_and_Zero__(rho1ys, Ds.size());
		__Check_and_Zero__(rho1zs, Ds.size());
		__Check_and_Zero__(sigmas, 2 * Ds.size() - 1);
		__Check_and_Zero__(lapls, Ds.size());
		__Check_and_Zero__(taus, Ds.size());
	}
	for ( int iD = 0; iD < (int)Ds.size(); iD++ )
		GetDensity(
			dorders,
			aos,
			ao1xs, ao1ys, ao1zs,
			ao2ls,
			ngrids, Ds[iD],
			rhos + ngrids * iD,
			rho1xs + ngrids * iD,
			rho1ys + ngrids * iD,
			rho1zs + ngrids * iD,
			lapls + ngrids * iD,
			taus + ngrids * iD,
			nthreads);
	if (std::find(dorders.begin(), dorders.end(), 1) != dorders.end()){
		if ( Ds.size() == 1 )
			for ( long int igrid = 0; igrid < ngrids; igrid++ )
				sigmas[igrid] = std::pow(rho1xs[igrid], 2) + std::pow(rho1ys[igrid], 2) + std::pow(rho1zs[igrid], 2);
		else{
			long int jgrid = 0;
			long int kgrid = 0;
			for ( long int igrid = 0; igrid < ngrids; igrid++ ){
				jgrid = igrid + ngrids;
				kgrid = jgrid + ngrids;
				sigmas[igrid] = std::pow(rho1xs[igrid], 2) + std::pow(rho1ys[igrid], 2) + std::pow(rho1zs[igrid], 2);
				sigmas[jgrid] = rho1xs[igrid] * rho1xs[jgrid] + rho1ys[igrid] * rho1ys[jgrid] + rho1zs[igrid] * rho1zs[jgrid];
				sigmas[kgrid] = std::pow(rho1xs[jgrid], 2) + std::pow(rho1ys[jgrid], 2) + std::pow(rho1zs[jgrid], 2);
			}
		}
	}

	// Step 3: Calculating XC data on grids
	if ( xc.XCfamily.compare("LDA") == 0 ){
		__Check_and_Zero__(es, 1);
		__Check_and_Zero__(erhos, Ds.size());
	}else if ( xc.XCfamily.compare("GGA") == 0 ){
		__Check_and_Zero__(es, 1);
		__Check_and_Zero__(erhos, Ds.size());
		__Check_and_Zero__(esigmas, 2 * Ds.size() - 1);
	}else if ( xc.XCfamily.compare("mGGA") == 0 ){
		__Check_and_Zero__(es, 1);
		__Check_and_Zero__(erhos, Ds.size());
		__Check_and_Zero__(esigmas, 2 * Ds.size() - 1);
		__Check_and_Zero__(elapls, Ds.size());
		__Check_and_Zero__(etaus, Ds.size());
	}
	xc.Evaluate(
			"ev", ngrids,
			rhos,
			sigmas,
			lapls, taus,
			es,
			erhos, esigmas, elapls, etaus,
			nullptr, nullptr, nullptr,
			nullptr, nullptr, nullptr, nullptr,
			nullptr, nullptr, nullptr, nullptr, nullptr);

	// Step 4: Calculating the XC energy
	double Exc = 0;
	for ( long int igrid = 0; igrid < ngrids; igrid++ )
		Exc += ws[igrid] * rhos[igrid] * es[igrid];

	// Step 5: Calculating the XC part of the Fock matrix
	// Needs to be modified of unrestricted SCF
	std::vector<int> vorders = {
		xc.XCfamily.compare("LDA") == 0 ? 0 :
			( xc.XCfamily.compare("GGA") == 0 ? 1 : 2 )
	};
	EigenMatrix Gxc = VMatrix(
			vorders,
			ws, ngrids, nbasis,
			aos,
			ao1xs, ao1ys, ao1zs,
			ao2ls,
			rho1xs, rho1ys, rho1zs,
			erhos, esigmas,
			elapls, etaus);

	return std::make_tuple(Exc, Gxc);
}
