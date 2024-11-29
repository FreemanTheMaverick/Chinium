#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <omp.h>

#include "../Macro.h"
#include "../Multiwfn.h"
#include "../Grid/GridDensity.h"
#include "../Grid/GridPotential.h"

#include <iostream>


std::vector<long int> getThreadPointers(long int nitems, int nthreads){
	const long int nitem_fewer = nitems / nthreads;
	const int nfewers = nthreads - nitems + nitem_fewer * nthreads;
	std::vector<long int> heads(nthreads, 0);
	long int n = 0;
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		heads[ithread] = n;
		n += (ithread < nfewers) ? nitem_fewer : nitem_fewer + 1;
	}
	return heads;
}

void GhfMultipleReduction(
		std::vector<std::vector<EigenArray>>& omp_out,
		std::vector<std::vector<EigenArray>>& omp_in){
	for ( int i = 0; i < (int)omp_out.size(); i++ )
		for ( int j = 0; j < (int)omp_out[i].size(); j++ )
			omp_out[i][j] += omp_in[i][j];
}

std::vector<EigenMatrix> GhfMultiple(
		short int* is, short int* js, short int* ks, short int* ls,
		char* degs, double* ints, long int length,
		std::vector<EigenMatrix>& Ds, double kscale, int nthreads){
	std::vector<long int> heads = getThreadPointers(length, nthreads);
	const int nbasis = Ds[0].rows();
	const int nmatrices = Ds.size();
	const int nmatrices_redun = std::ceil((double)nmatrices / 8.) * 8;
	std::vector<std::vector<EigenArray>> vvvD(nbasis);
	std::vector<std::vector<EigenArray>> vvvrawJ(nbasis);
	std::vector<std::vector<EigenArray>> vvvrawK(nbasis);
	for ( int i = 0; i < nbasis; i++ ){
		vvvD[i].resize(nbasis);
		vvvrawJ[i].resize(nbasis);
		vvvrawK[i].resize(nbasis);
		for ( int j = 0; j < nbasis; j++ ){
			vvvD[i][j] = EigenZero(nmatrices_redun, 1).array();
			vvvrawJ[i][j] = EigenZero(nmatrices_redun, 1).array();
			vvvrawK[i][j] = EigenZero(nmatrices_redun, 1).array();
			for ( int k = 0; k < nmatrices; k++ )
				vvvD[i][j](k) = Ds[k](i, j);
		}
	}
	omp_set_num_threads(nthreads);
	#pragma omp declare reduction(vvv_reduction: std::vector<std::vector<EigenArray>>: GhfMultipleReduction(omp_out, omp_in)) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(vvv_reduction: vvvrawJ, vvvrawK)
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		const long int head = heads[ithread];
		const long int nints = ( (ithread == nthreads - 1) ? length : heads[ithread + 1] ) - head;
		short int* iranger = is + head;
		short int* jranger = js + head;
		short int* kranger = ks + head;
		short int* lranger = ls + head;
		char* degranger = degs + head;
		double* repulsionranger = ints + head;
		for ( long int iint = 0; iint < nints; iint++ ){
			const short int i = *(iranger++);
			const short int j = *(jranger++);
			const short int k = *(kranger++);
			const short int l = *(lranger++);
			const double deg_value = *(degranger++) * *(repulsionranger++);
			vvvrawJ[i][j] += vvvD[k][l] * deg_value;
			vvvrawJ[k][l] += vvvD[i][j] * deg_value;
			if ( kscale > 0. ){
				vvvrawK[i][k] += vvvD[j][l] * deg_value;
				vvvrawK[j][l] += vvvD[i][k] * deg_value;
				vvvrawK[i][l] += vvvD[j][k] * deg_value;
				vvvrawK[j][k] += vvvD[i][l] * deg_value;
			}
		}
	}
	std::vector<EigenMatrix> rawGs(nmatrices, EigenZero(nbasis, nbasis));
	for ( int k = 0; k < nmatrices; k++ ){
		EigenMatrix rawJ = EigenZero(nbasis, nbasis);
		EigenMatrix rawK = EigenZero(nbasis, nbasis);
		for ( int i = 0; i < nbasis; i++ ){
			for ( int j = 0; j < nbasis; j++ ){
				rawJ(i, j) = vvvrawJ[i][j](k);
				rawK(i, j) = vvvrawK[i][j](k);
			}
		}
		const EigenMatrix J = 0.5 *  ( rawJ + rawJ.transpose() );
		const EigenMatrix K = 0.25 * ( rawK + rawK.transpose() );
		rawGs[k] = J - 0.5 * kscale * K;
	}
	return rawGs;
}


EigenMatrix Ghf(
		short int* is, short int* js, short int* ks, short int* ls,
		char* degs, double* ints, long int length,
		EigenMatrix D, double kscale, int nthreads){
	std::vector<long int> heads = getThreadPointers(length, nthreads);
	EigenMatrix rawJ = EigenZero(D.rows(), D.cols());
	EigenMatrix rawK = EigenZero(D.rows(), D.cols());
	omp_set_num_threads(nthreads);
	#pragma omp declare reduction(EigenMatrixSum: EigenMatrix: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(EigenMatrixSum: rawJ, rawK)
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		const long int head = heads[ithread];
		const long int nints = ( (ithread == nthreads - 1) ? length : heads[ithread + 1] ) - head;
		short int* iranger = is + head;
		short int* jranger = js + head;
		short int* kranger = ks + head;
		short int* lranger = ls + head;
		char* degranger = degs + head;
		double* repulsionranger = ints + head;
		for ( long int iint = 0; iint < nints; iint++ ){
			const short int i = *(iranger++);
			const short int j = *(jranger++);
			const short int k = *(kranger++);
			const short int l = *(lranger++);
			const double deg_value = *(degranger++) * *(repulsionranger++);
			rawJ(i, j) += D(k, l) * deg_value;
			rawJ(k, l) += D(i, j) * deg_value;
			if ( kscale > 0. ){
				rawK(i, k) += D(j, l) * deg_value;
				rawK(j, l) += D(i, k) * deg_value;
				rawK(i, l) += D(j, k) * deg_value;
				rawK(j, k) += D(i, l) * deg_value;
			}
		}
	}
	const EigenMatrix J = 0.5 * ( rawJ + rawJ.transpose() );
	const EigenMatrix K = 0.25 * ( rawK + rawK.transpose() );
	return J - 0.5 * kscale * K;
}

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
		double* elapls, double* etaus){

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
			taus + ngrids * iD);
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

// Needs modification for unrestricted SCF
std::tuple<EigenMatrix, EigenMatrix, double> Multiwfn::calcFock(EigenMatrix D, int nthreads){
	const EigenMatrix Fhf = this->Kinetic + this->Nuclear + Ghf(
		this->RepulsionIs, this->RepulsionJs,
		this->RepulsionKs, this->RepulsionLs,
		this->RepulsionDegs, this->Repulsions, this->RepulsionLength,
		D, this->XC.EXX, nthreads);
	double Exc = 0;
	EigenMatrix Fxc = EigenZero(Fhf.rows(), Fhf.cols());
	if (this->XC.Xcode != 0 || this->XC.XCcode != 0)
		std::tie(Exc, Fxc) = Gxc(
			this->XC,
			this->Ws, this->NumGrids, this->getNumBasis(),
			this->AOs,
			this->AO1Xs, this->AO1Ys, this->AO1Zs,
			this->AO2Ls,
			{D*2},
			this->Rhos,
			this->Rho1Xs, this->Rho1Ys, this->Rho1Zs, this->Sigmas,
			this->Lapls, this->Taus,
			this->Es,
			this->E1Rhos, this->E1Sigmas,
			nullptr, nullptr);
	return std::make_tuple(Fhf, Fxc, Exc);
}


