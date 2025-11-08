#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <vector>
#include <array>
#include <chrono>
#include <cstdio>
#include <string>
#include <libmwfn.h>

#include "../Macro.h"
#include "Grid.h"

void Grid::setType(int type){
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->Type = type;
		}
	}
}

void Grid::getAO(int derivative, int output){
	const int order = derivative + this->SubGridBatches[0][0]->Type;
	if (output) std::printf("Generating grids to order %d of basis functions ... ", order);
	auto start = __now__;
	#pragma omp parallel for schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getAO(derivative);
		}
	}
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
}

void Grid::getDensity(std::vector<EigenMatrix> D_){
	const int nspins = this->SubGridBatches[0][0]->Spin;
	const int nbasis = this->SubGridBatches[0][0]->MWFN->getNumBasis();
	assert( nspins == (int)D_.size() && "Wrong number of spin types!" );
	EigenTensor<3> D(nbasis, nbasis, nspins);
	for ( int spin = 0; spin < nspins; spin++ ){
		std::memcpy(&D(0, 0, spin), D_[spin].data(), nbasis * nbasis * 8);
	}
	#pragma omp parallel for schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getDensity(D);
		}
	}
}

double Grid::getNumElectrons(){
	EigenTensor<0> N; N() = 0;
	#pragma omp declare reduction(EigenTensorSum: EigenTensor<0>: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(EigenTensorSum: N) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getNumElectrons(N);
		}
	}
	return N();
}

void Grid::getDensityU(std::vector<std::vector<EigenMatrix>> D_){
	const int nspins = this->SubGridBatches[0][0]->Spin;
	const int nbasis = this->SubGridBatches[0][0]->MWFN->getNumBasis();
	const int nmats = (int)D_[0].size();
	assert( nspins == (int)D_.size() && "Wrong number of spin types!" );
	EigenTensor<4> D(nbasis, nbasis, nmats, nspins);
	for ( int spin = 0; spin < nspins; spin++ ) for ( int mat = 0; mat < nmats; mat++ ){
		std::memcpy(&D(0, 0, mat, spin), D_[spin][mat].data(), nbasis * nbasis * 8);
	}
	#pragma omp parallel for schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getDensityU(D);
		}
	}
}

void Grid::getDensitySkeleton(std::vector<EigenMatrix> D_){
	const int nspins = this->SubGridBatches[0][0]->Spin;
	const int nbasis = this->SubGridBatches[0][0]->MWFN->getNumBasis();
	assert( nspins == (int)D_.size() && "Wrong number of spin types!" );
	EigenTensor<3> D(nbasis, nbasis, nspins);
	for ( int spin = 0; spin < nspins; spin++ ){
		std::memcpy(&D(0, 0, spin), D_[spin].data(), nbasis * nbasis * 8);
	}
	#pragma omp parallel for schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getDensitySkeleton(D);
		}
	}
}

void Grid::getDensitySkeleton2(std::vector<EigenMatrix> D_){
	const int nspins = this->SubGridBatches[0][0]->Spin;
	const int nbasis = this->SubGridBatches[0][0]->MWFN->getNumBasis();
	assert( nspins == (int)D_.size() && "Wrong number of spin types!" );
	EigenTensor<3> D(nbasis, nbasis, nspins);
	for ( int spin = 0; spin < nspins; spin++ ){
		std::memcpy(&D(0, 0, spin), D_[spin].data(), nbasis * nbasis * 8);
	}
	#pragma omp parallel for schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getDensitySkeleton2(D);
		}
	}
}

double Grid::getEnergy(){
	EigenTensor<0> E; E() = 0;
	#pragma omp declare reduction(EigenTensorSum: EigenTensor<0>: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(EigenTensorSum: E) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getEnergy(E);
		}
	}
	return E();
}

std::vector<double> Grid::getEnergyGrad(){
	const int natoms = this->SubGridBatches[0][0]->MWFN->getNumCenters();
	EigenTensor<2> E(3, natoms); E.setZero();
	#pragma omp declare reduction(EigenTensorSum: EigenTensor<2>: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(EigenTensorSum: E) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getEnergyGrad(E);
		}
	}
	std::vector<double> grad(3 * natoms);
	grad.assign(&E(0, 0), &E(0, 0) + 3 * natoms);
	return grad;
}

std::vector<std::vector<double>> Grid::getEnergyHess(){
	const int natoms = this->SubGridBatches[0][0]->MWFN->getNumCenters();
	EigenTensor<4> E(3, natoms, 3, natoms); E.setZero();
	#pragma omp declare reduction(EigenTensorSum: EigenTensor<4>: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(EigenTensorSum: E) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getEnergyHess(E);
		}
	}
	std::vector<std::vector<double>> hess(3 * natoms);
	for ( int atom = 0, pert = 0; atom < natoms; atom++ )
		for ( int t = 0; t < 3; t++, pert++ )
			hess[pert].assign(&E(0, 0, t, atom), &E(0, 0, t, atom) + 3 * natoms);
	return hess;
}

std::vector<EigenMatrix> Grid::getFock(){
	const int nbasis = this->SubGridBatches[0][0]->MWFN->getNumBasis();
	const int nspins = this->SubGridBatches[0][0]->Spin;
	EigenTensor<3> F(nbasis, nbasis, nspins); F.setZero();
	#pragma omp declare reduction(EigenTensorSum: EigenTensor<3>: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(EigenTensorSum: F) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getFock(F);
		}
	}
	std::vector<EigenMatrix> Fs(nspins, EigenZero(nbasis, nbasis));
	for ( int spin = 0; spin < nspins; spin++ )
		std::memcpy(Fs[spin].data(), &F(0, 0, spin), nbasis * nbasis * 8);
	return Fs;
}

std::vector<std::vector<EigenMatrix>> Grid::getFockSkeleton(){
	const int natoms = this->SubGridBatches[0][0]->MWFN->getNumCenters();
	const int nbasis = this->SubGridBatches[0][0]->MWFN->getNumBasis();
	const int nspins = this->SubGridBatches[0][0]->Spin;
	EigenTensor<5> F(nbasis, nbasis, 3, natoms, nspins); F.setZero();
	#pragma omp declare reduction(EigenTensorSum: EigenTensor<5>: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(EigenTensorSum: F) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getFockSkeleton(F);
		}
	}
	std::vector<std::vector<EigenMatrix>> Fs(nspins, std::vector<EigenMatrix>(3 * natoms, EigenZero(nbasis, nbasis)));
	for ( int spin = 0; spin < nspins; spin++ ) for ( int iatom = 0, jpert = 0; iatom < natoms; iatom++ ) for ( int t = 0; t < 3; t++, jpert++ ){
		std::memcpy(Fs[spin][jpert].data(), &F(0, 0, t, iatom, spin), nbasis * nbasis * 8);
	}
	return Fs;
}

// enum D_t{ s_t, u_t };
template <D_t d_t>
std::vector<std::vector<EigenMatrix>> Grid::getFockU(){
	const int nbasis = this->SubGridBatches[0][0]->MWFN->getNumBasis();
	const int natoms = this->SubGridBatches[0][0]->MWFN->getNumCenters();
	const int nmats =  [this, natoms]() -> int {
			if constexpr ( d_t == s_t ) return natoms * 3;
			else return this->SubGridBatches[0][0]->RhoU.dimension(1); // Crap. I have to go inside a SubGrid.
;
	}();
	const int nspins = this->SubGridBatches[0][0]->Spin;
	EigenTensor<4> F(nbasis, nbasis, nmats, nspins); F.setZero();
	#pragma omp declare reduction(EigenTensorSum: EigenTensor<4>: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(EigenTensorSum: F) schedule(static) num_threads(this->getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : this->SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){
			subgrid->getFockU<d_t>(F);
		}
	}
	std::vector<std::vector<EigenMatrix>> Fs(nspins, std::vector<EigenMatrix>(nmats, EigenZero(nbasis, nbasis)));
	for ( int spin = 0; spin < nspins; spin++ ) for ( int mat = 0; mat < nmats; mat++ ){
		std::memcpy(Fs[spin][mat].data(), &F(0, 0, mat, spin), nbasis * nbasis * 8);
	}
	return Fs;
}
template std::vector<std::vector<EigenMatrix>> Grid::getFockU<s_t>(); // Using skeleton derivatives of density
template std::vector<std::vector<EigenMatrix>> Grid::getFockU<u_t>(); // Using U derivatives of density
