#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>

#include "../Macro.h"
#include "../Multiwfn.h"


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

EigenMatrix Ghf(
		short int* is, short int* js, short int* ks, short int* ls,
		char* degs, double* ints, long int length,
		EigenMatrix D, double kscale, int nthreads){
	std::vector<long int> heads = getThreadPointers(length, nthreads);
	std::vector<EigenMatrix> rawJs(nthreads, EigenZero(D.rows(), D.cols()));
	std::vector<EigenMatrix> rawKs(nthreads, EigenZero(D.rows(), D.cols()));
	omp_set_num_threads(nthreads);
	#pragma omp parallel for
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		const long int head = heads[ithread];
		const long int nints = ( (ithread == nthreads - 1) ? length : heads[ithread + 1] ) - head;
		short int* iranger = is + head;
		short int* jranger = js + head;
		short int* kranger = ks + head;
		short int* lranger = ls + head;
		char* degranger = degs + head;
		double* repulsionranger = ints + head;
		EigenMatrix* rawJ = &rawJs[ithread];
		EigenMatrix* rawK = &rawKs[ithread];
		for ( long int iint = 0; iint < nints; iint++ ){
			const short int i = *(iranger++);
			const short int j = *(jranger++);
			const short int k = *(kranger++);
			const short int l = *(lranger++);
			const double deg_value = *(degranger++) * *(repulsionranger++);
			(*rawJ)(i, j) += D(k, l) * deg_value;
			(*rawJ)(k, l) += D(i, j) * deg_value;
			if ( kscale > 0. ){
				(*rawK)(i, k) += D(j, l) * deg_value;
				(*rawK)(j, l) += D(i, k) * deg_value;
				(*rawK)(i, l) += D(j, k) * deg_value;
				(*rawK)(j, k) += D(i, l) * deg_value;
			}
		}
	}
	EigenMatrix rawJ = EigenZero(D.rows(), D.cols());
	EigenMatrix rawK = EigenZero(D.rows(), D.cols());
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		rawJ += rawJs[ithread];
		if ( kscale > 0. ) rawK += rawKs[ithread];
	}
	const EigenMatrix J = 0.5 * ( rawJ + rawJ.transpose() );
	const EigenMatrix K = 0.25 * ( rawK + rawK.transpose() );
	return J - 0.5 * kscale * K;
}

EigenMatrix Multiwfn::calcFock(int nthreads){
	return this->Kinetic + this->Nuclear + Ghf(
		this->RepulsionIs, this->RepulsionJs,
		this->RepulsionKs, this->RepulsionLs,
		this->RepulsionDegs, this->Repulsions, this->RepulsionLength,
		this->getDensity(), 1., nthreads);
}

EigenMatrix Multiwfn::calcFock(EigenMatrix D, int nthreads){
	return this->Kinetic + this->Nuclear + Ghf(
		this->RepulsionIs, this->RepulsionJs,
		this->RepulsionKs, this->RepulsionLs,
		this->RepulsionDegs, this->Repulsions, this->RepulsionLength,
		D, 1., nthreads);
}
