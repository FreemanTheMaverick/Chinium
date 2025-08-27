#include <Eigen/Dense>
#include <cmath>
#include <deque>
#include <vector>
#include <cstdio>

#include "../Macro.h"
#include "AugmentedRoothaanHall.h"

#define current_size (int)this->Ps.size()

void AugmentedRoothaanHall::Init(int max_size, bool verbose){
	if (verbose) std::printf("Initialized Augmented Roothaan Hall hessian with maximal size %d.\n", max_size);
	this->MaxSize = max_size;
	this->Verbose = verbose;
	this->Pdiffs.reserve(max_size);
	this->Gdiffs.reserve(max_size);
}

void AugmentedRoothaanHall::Append(EigenMatrix P, EigenMatrix G){
	const int size = (int)this->Ps.size();
	this->Pdiffs.resize(size);
	this->Gdiffs.resize(size);
	for ( int i = 0; i < size; i++ ){
		this->Pdiffs[i] = P - this->Ps[i];
		this->Gdiffs[i] = G - this->Gs[i];
	}
	EigenMatrix T = EigenZero(size, size);
	for ( int i = 0; i < size; i++ ){
		for ( int j = i; j < size; j++ ){
			T(i, j) = T(j, i) = Pdiffs[i].cwiseProduct(Pdiffs[j]).sum();
		}
	}
	this->Tinv = T.inverse();
	if ( size == this->MaxSize ){
		this->Ps.pop_front();
		this->Gs.pop_front();
	}
	this->Ps.push_back(P);
	this->Gs.push_back(G);
}

EigenMatrix AugmentedRoothaanHall::Hessian(EigenMatrix v){
	const int size = (int)this->Pdiffs.size();
	if ( size <= 1 ) return v;

	std::vector<double> TrDsv(size);
	for ( int j = 0; j < size; j++ ){
		TrDsv[j] = Pdiffs[j].cwiseProduct(v).sum();
	}
	EigenMatrix Hv = EigenZero(v.rows(), v.cols());
	for ( int i = 0; i < size; i++ ){
		double coeff = 0;
		for ( int j = 0; j < size; j++ ){
			coeff += Tinv(i, j) * TrDsv[j];
		}
		Hv += coeff * Gdiffs[i];
	}
	return Hv;
}
