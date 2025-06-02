#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <vector>
#include <set>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cassert>
#include <omp.h>

#include "../Macro.h"
#include "../Multiwfn/Multiwfn.h"
#include "Grid.h"

void Grid::getDensity(EigenMatrix D_){
	const int ngrids = this->NumGrids;
	const int nbasis = (int)D_.cols();
	Eigen::Tensor<double, 1>& Rhos = this->Rhos_Cache;
	Eigen::Tensor<double, 2>& Rho1s = this->Rho1s_Cache;
	Eigen::Tensor<double, 1>& Sigmas = this->Sigmas_Cache;
	Eigen::Tensor<double, 1>& Lapls = this->Lapls_Cache;
	Eigen::Tensor<double, 1>& Taus = this->Taus_Cache;
	Eigen::Tensor<double, 2> D = Eigen::TensorMap<Eigen::Tensor<double, 2>>(D_.data(), nbasis, nbasis);
	if ( this->Type >= 0 ){
		Rhos.resize(ngrids); Rhos.setZero();
		#include "DensityEinSum/D_mu,nu...AOs_g,mu...AOs_g,nu---Rhos_g.hpp"
	}
	if ( this->Type >= 1 ){
		Rho1s.resize(ngrids, 3); Rho1s.setZero();
		#include "DensityEinSum/D_mu,nu...AO1s_g,mu,r...AOs_g,nu---Rho1s_g,r.hpp"
		ScaleTensor(Rho1s, 2);
		Sigmas.resize(ngrids); Sigmas.setZero();
		#include "DensityEinSum/Rho1s_g,r...Rho1s_g,r---Sigmas_g.hpp"
	}
	if ( this->Type >= 2 ){
		Taus.resize(ngrids); Taus.setZero();
		#include "DensityEinSum/D_mu,nu...AO1s_g,mu,r...AO1s_g,nu,r---Taus_g.hpp"
		ScaleTensor(Taus, 0.5);
		Lapls = 4. * Taus;
		#include "DensityEinSum/D_mu,nu...AOs_g,mu...AO2Ls_g,nu---Lapls_g.hpp"
		#include "DensityEinSum/D_mu,nu...AO2Ls_g,mu...AOs_g,nu---Lapls_g.hpp"
	}
}

void Grid::getDensityU(
		std::vector<EigenMatrix> DUs_,
		std::vector<Eigen::Tensor<double, 1>>& RhoUss,
		std::vector<Eigen::Tensor<double, 2>>& Rho1Uss,
		std::vector<Eigen::Tensor<double, 1>>& SigmaUss){
	const int ngrids = this->NumGrids;
	const int nbasis = (int)DUs_[0].cols();
	const int nmats = (int)DUs_.size();
	Eigen::Tensor<double, 1> dummy1;
	Eigen::Tensor<double, 2> dummy2;
	if ( this->Type >= 0 ){
		RhoUss.resize(nmats);
	}
	if ( this->Type >= 1 ){
		Rho1Uss.resize(nmats);
		SigmaUss.resize(nmats);
	}
	for ( int imat = 0; imat < nmats; imat++ ){
		Eigen::Tensor<double, 1>& RhoUs = Type >= 0 ? RhoUss[imat] : dummy1;
		Eigen::Tensor<double, 2>& Rho1Us = Type >= 1 ? Rho1Uss[imat] : dummy2;
		Eigen::Tensor<double, 1>& SigmaUs = Type >= 1 ? SigmaUss[imat] : dummy1;
		Eigen::Tensor<double, 2> DU = Eigen::TensorMap<Eigen::Tensor<double, 2>>(DUs_[imat].data(), nbasis, nbasis);
		if ( this->Type >= 0 ){
			RhoUs.resize(ngrids); RhoUs.setZero();
			#include "DensityEinSum/DU_mu,nu...AOs_g,mu...AOs_g,nu---RhoUs_g.hpp"
		}
		if ( this->Type >= 1 ){
			Rho1Us.resize(ngrids, 3); Rho1Us.setZero();
			#include "DensityEinSum/DU_mu,nu...AO1s_g,mu,r...AOs_g,nu---Rho1Us_g,r.hpp"
			ScaleTensor(Rho1Us, 2);
			SigmaUs.resize(ngrids); SigmaUs.setZero();
			#include "DensityEinSum/Rho1Us_g,r...Rho1s_g,r---SigmaUs_g.hpp"
			ScaleTensor(SigmaUs, 2);
		}
	}
}

void Grid::SaveDensity(){
	this->Rhos = this->Rhos_Cache;
	this->Rho1s = this->Rho1s_Cache;
	this->Sigmas = this->Sigmas_Cache;
	this->Lapls = this->Lapls_Cache;
	this->Taus = this->Taus_Cache;
}

void Grid::RetrieveDensity(){
	this->Rhos_Cache = this->Rhos;
	this->Rho1s_Cache = this->Rho1s;
	this->Sigmas_Cache = this->Sigmas;
	this->Lapls_Cache = this->Lapls;
	this->Taus_Cache = this->Taus;
}

double Grid::getNumElectrons(){
	assert( this->NumGrids == Weights.dimension(0) );
	assert( this->Rhos.size() ? this->NumGrids == Rhos.dimension(0) : this->NumGrids == Rhos_Cache.dimension(0) );
	Eigen::Tensor<double, 1>& Rhos = this->Rhos.size() ? this->Rhos : this->Rhos_Cache;
	const Eigen::Tensor<double, 0> n = (Weights * Rhos).sum();
	return n();
}

void Grid::getDensitySkeleton(EigenMatrix D_){
	const int ngrids = this->NumGrids;
	const int nbasis = this->Mwfn->getNumBasis();
	const int natoms = this->Mwfn->getNumCenters();
	const std::vector<int> atom2bf = this->Mwfn->Atom2Basis();
	Eigen::Tensor<double, 2> D = Eigen::TensorMap<Eigen::Tensor<double, 2>>(D_.data(), nbasis, nbasis);
	if ( this->Type >= 0 ){
		RhoGrads.resize(ngrids, 3, natoms); RhoGrads.setZero();
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int head = atom2bf[iatom];
			const int length = this->Mwfn->Centers[iatom].getNumBasis();
			const Eigen::Tensor<double, 2> Da = SliceTensor(D, {head, 0}, {length, nbasis});
			const Eigen::Tensor<double, 3> AO1sa = SliceTensor(AO1s, {0, head, 0}, {ngrids, length, 3});
			Eigen::Tensor<double, 2> RhoGradsa(ngrids, 3);
			RhoGradsa.setZero();
			#include "DensityEinSum/Da_mu,nu...AO1sa_g,mu,t...AOs_g,nu---RhoGradsa_g,t.hpp"
			RhoGrads.chip(iatom, 2) = RhoGradsa;
		}
		ScaleTensor(RhoGrads, -2);
	}
	if ( this->Type >= 1 ){
		Rho1Grads.resize(ngrids, 3, 3, natoms); Rho1Grads.setZero();
		SigmaGrads.resize(ngrids, 3, natoms); SigmaGrads.setZero();
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int head = atom2bf[iatom];
			const int length = this->Mwfn->Centers[iatom].getNumBasis();
			Eigen::Tensor<double, 2> Da = SliceTensor(D, {head, 0}, {length, nbasis});
			const Eigen::Tensor<double, 3> AO2sa = SliceTensor(AO2s, {0, head, 0}, {ngrids, length, 6});
			Eigen::Tensor<double, 3> Rho1Gradsa(ngrids, 3, 3);
			Rho1Gradsa.setZero();
			#include "DensityEinSum/Da_mu,nu...AO2sa_g,mu,r,t...AOs_g,nu---Rho1Gradsa_g,r,t.hpp"
			const Eigen::Tensor<double, 3> AO1sa = SliceTensor(AO1s, {0, head, 0}, {ngrids, length, 3});
			#include "DensityEinSum/Da_mu,nu...AO1sa_g,mu,t...AO1s_g,nu,r---Rho1Gradsa_g,r,t.hpp"
			ScaleTensor(Rho1Gradsa, -2);
			Rho1Grads.chip(iatom, 3) = Rho1Gradsa;
			Eigen::Tensor<double, 2> SigmaGradsa(ngrids, 3);
			SigmaGradsa.setZero();
			#include "DensityEinSum/Rho1s_g,r...Rho1Gradsa_g,r,t---SigmaGradsa_g,t.hpp"
			SigmaGrads.chip(iatom, 2) = 2 * SigmaGradsa;
		}
	}
}

void Grid::getDensitySkeleton2(EigenMatrix D_){
	const int ngrids = this->NumGrids;
	const int nbasis = (int)D_.cols();
	const int natoms = this->Mwfn->getNumCenters();
	const std::vector<int> atom2bf = this->Mwfn->Atom2Basis();
	Eigen::Tensor<double, 2> D = Eigen::TensorMap<Eigen::Tensor<double, 2>>(D_.data(), nbasis, nbasis);
	if ( this->Type >= 0 ){
		RhoHesss.resize(ngrids, 3, natoms, 3, natoms); RhoHesss.setZero();
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = atom2bf[iatom];
			const int ilength = this->Mwfn->Centers[iatom].getNumBasis();
			const Eigen::Tensor<double, 2> Da = SliceTensor(D, {ihead, 0}, {ilength, nbasis});
			const Eigen::Tensor<double, 3> AO2sa = SliceTensor(AO2s, {0, ihead, 0}, {ngrids, ilength, 6});
			Eigen::Tensor<double, 3> RhoHesssaa(ngrids, 3, 3);
			RhoHesssaa.setZero();
			#include "DensityEinSum/Da_mu,nu...AO2sa_g,mu,t,s...AOs_g,nu---RhoHesssaa_g,t,s.hpp"
			const Eigen::Tensor<double, 2> Daa = SliceTensor(D, {ihead, ihead}, {ilength, ilength});
			const Eigen::Tensor<double, 3> AO1sa = SliceTensor(AO1s, {0, ihead, 0}, {ngrids, ilength, 3});
			#include "DensityEinSum/Daa_mu,nu...AO1sa_g,mu,t...AO1sa_g,nu,s---RhoHesssaa_g,t,s.hpp"
			RhoHesss.chip(iatom, 4).chip(iatom, 2) = RhoHesssaa;
			RhoHesssaa.resize(0, 0, 0);
			for ( int jatom = 0; jatom < iatom; jatom++ ){
				const int jhead = atom2bf[jatom];
				const int jlength = this->Mwfn->Centers[jatom].getNumBasis();
				const Eigen::Tensor<double, 2> Dab = SliceTensor(D, {ihead, jhead}, {ilength, jlength});
				const Eigen::Tensor<double, 3> AO1sb = SliceTensor(AO1s, {0, jhead, 0}, {ngrids, jlength, 3});
				Eigen::Tensor<double, 3> RhoHesssab(ngrids, 3, 3);
				RhoHesssab.setZero();
				#include "DensityEinSum/Dab_mu,nu...AO1sa_g,mu,t...AO1sb_g,nu,s---RhoHesssab_g,t,s.hpp"
				RhoHesss.chip(jatom, 4).chip(iatom, 2) = RhoHesssab;
				RhoHesss.chip(iatom, 4).chip(jatom, 2) = RhoHesssab.shuffle(Eigen::array<int, 3>{0, 2, 1});
			}
		}
		ScaleTensor(RhoHesss, 2);
	}
	if ( this->Type >= 1 ){
		Rho1Hesss.resize(ngrids, 3, 3, natoms, 3, natoms); Rho1Hesss.setZero();
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = atom2bf[iatom];
			const int ilength = this->Mwfn->Centers[iatom].getNumBasis();
			Eigen::Tensor<double, 4> Rho1Hesssaa(ngrids, 3, 3, 3);
			Rho1Hesssaa.setZero();
			Eigen::Tensor<double, 2> Daa = SliceTensor(D, {ihead, ihead}, {ilength, ilength});
			const Eigen::Tensor<double, 3> AO1sa = SliceTensor(AO1s, {0, ihead, 0}, {ngrids, ilength, 3});
			const Eigen::Tensor<double, 3> AO2sa = SliceTensor(AO2s, {0, ihead, 0}, {ngrids, ilength, 6});
			#include "DensityEinSum/Daa_mu,nu...AO2sa_g,mu,r,t...AO1sa_g,nu,s---Rho1Hesssaa_g,r,t,s.hpp" // 2, 4
			Rho1Hesss.chip(iatom, 5).chip(iatom, 3) = Rho1Hesssaa + Rho1Hesssaa.shuffle(Eigen::array<int, 4>{0, 1, 3, 2});
			Rho1Hesssaa.setZero();
			Eigen::Tensor<double, 2> Da = SliceTensor(D, {ihead, 0}, {ilength, nbasis});
			const Eigen::Tensor<double, 3> AO3sa = SliceTensor(AO3s, {0, ihead, 0}, {ngrids, ilength, 10});
			#include "DensityEinSum/Da_mu,nu...AO3sa_g,mu,r,t,s...AOs_g,nu---Rho1Hesssaa_g,r,t,s.hpp" // 1
			#include "DensityEinSum/Da_mu,nu...AO2sa_g,mu,t,s...AO1s_g,nu,r---Rho1Hesssaa_g,r,t,s.hpp" // 3
			Rho1Hesss.chip(iatom, 5).chip(iatom, 3) += Rho1Hesssaa;
			Rho1Hesssaa.resize(0, 0, 0, 0);
			for ( int jatom = 0; jatom < iatom; jatom++ ){
				const int jhead = atom2bf[jatom];
				const int jlength = this->Mwfn->Centers[jatom].getNumBasis();
				Eigen::Tensor<double, 4> Rho1Hesssab(ngrids, 3, 3, 3);
				Rho1Hesssab.setZero();
				Eigen::Tensor<double, 2> Dab = SliceTensor(D, {ihead, jhead}, {ilength, jlength});
				const Eigen::Tensor<double, 3> AO1sb = SliceTensor(AO1s, {0, jhead, 0}, {ngrids, jlength, 3});
				#include "DensityEinSum/Dab_mu,nu...AO2sa_g,mu,r,t...AO1sb_g,nu,s---Rho1Hesssab_g,r,t,s.hpp" // 2
				const Eigen::Tensor<double, 3> AO2sb = SliceTensor(AO2s, {0, jhead, 0}, {ngrids, jlength, 6});
				#include "DensityEinSum/Dab_mu,nu...AO1sa_g,mu,t...AO2sb_g,nu,r,s---Rho1Hesssab_g,r,t,s.hpp" // 4
				Rho1Hesss.chip(jatom, 5).chip(iatom, 3) = Rho1Hesssab;
				Rho1Hesss.chip(iatom, 5).chip(jatom, 3) = Rho1Hesssab.shuffle(Eigen::array<int, 4>{0, 1, 3, 2});
			}
		}
		ScaleTensor(Rho1Hesss, 2);
		SigmaHesss.resize(ngrids, 3, natoms, 3, natoms); SigmaHesss.setZero();
		#include "DensityEinSum/Rho1Hesss_g,r,t,a,s,b...Rho1s_g,r---SigmaHesss_g,t,a,s,b.hpp"
		#include "DensityEinSum/Rho1Grads_g,r,t,a...Rho1Grads_g,r,s,b---SigmaHesss_g,t,a,s,b.hpp"
		ScaleTensor(SigmaHesss, 2);
	}
}
