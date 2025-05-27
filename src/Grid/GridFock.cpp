#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>

#include "../Macro.h"
#include "../Multiwfn/Multiwfn.h"
#include "Grid.h"
#include <iostream>
EigenMatrix Grid::getFock(int type){ // Default type = -1
	if ( type == -1 ) type = this->Type;
	const Eigen::Tensor<double, 1>& Ws = Weights;
	const int nbasis = this->Mwfn->getNumBasis();
	Eigen::Tensor<double, 2> F(nbasis, nbasis); F.setZero();
	if ( type >= 0 ){
		Eigen::Tensor<double, 2> F0(nbasis, nbasis); F0.setZero();
		#include "FockEinSum/Ws_g...E1Rhos_g...AOs_g,mu...AOs_g,nu---F0_mu,nu.hpp"
		F += 0.5 * F0;
	}
	if ( type >= 1 ){
		Eigen::Tensor<double, 2> F1(nbasis, nbasis); F1.setZero();
		Eigen::Tensor<double, 2>& Rho1s = this->Rho1s.size() ? this->Rho1s : this->Rho1s_Cache;
		#include "FockEinSum/Ws_g...E1Sigmas_g...Rho1s_g,r...AO1s_g,mu,r...AOs_g,nu---F1_mu,nu.hpp"
		F += 2. * F1;
	}
	if ( type >= 2 ){
		Eigen::Tensor<double, 2> F2(nbasis, nbasis); F2.setZero();
		Eigen::Tensor<double, 1> V = 0.5 * E1Taus + 2 * E1Lapls;
		#include "FockEinSum/Ws_g...V_g...AO1s_g,mu,r...AO1s_g,nu,r---F2_mu,nu.hpp"
		F += 0.5 * F2;
		F2.setZero();
		#include "FockEinSum/Ws_g...E1Lapls_g...AOs_g,mu...AO2Ls_g,nu---F2_mu,nu.hpp"
		F += F2;
	}
	F += F.shuffle(Eigen::array<int, 2>{1, 0}).eval();
	const EigenMatrix Fmat = Eigen::Map<EigenMatrix>(F.data(), nbasis, nbasis);
	return Fmat;
}

std::vector<EigenMatrix> Grid::getFockSkeleton(){
	const Eigen::Tensor<double, 1>& Ws = Weights;
	const int nbasis = this->Mwfn->getNumBasis();
	const int ngrids = this->NumGrids;
	const int natoms = this->Mwfn->getNumCenters();
	const std::vector<int> atom2bf = this->Mwfn->Atom2Basis();
	Eigen::Tensor<double, 4> F(nbasis, nbasis, 3, natoms); F.setZero();
	if ( this->Type >= 0 ){
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = atom2bf[iatom];
			const int ilength = this->Mwfn->Centers[iatom].getNumBasis();
			const Eigen::Tensor<double, 3> AO1sa = SliceTensor(AO1s, {0, ihead, 0}, {ngrids, ilength, 3});
			Eigen::Tensor<double, 3> F0a(nbasis, nbasis, 3); F0a.setZero();
			#include "FockEinSum/Ws_g...E1Rhos_g...AO1sa_g,mu,t...AOs_g,nu---F0a_mu,nu,t.hpp"
			F.chip(iatom, 3) -= F0a;
		}
	}
	if ( this->Type >= 1 ){
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = atom2bf[iatom];
			const int ilength = this->Mwfn->Centers[iatom].getNumBasis();
			Eigen::Tensor<double, 3> Fa1(nbasis, nbasis, 3); Fa1.setZero();
			const Eigen::Tensor<double, 3> AO2sa = SliceTensor(AO2s, {0, ihead, 0}, {ngrids, ilength, 6});
			#include "FockEinSum/Ws_g...E1Sigmas_g...Rho1s_g,r...AO2sa_g,mu,r,t...AOs_g,nu---Fa1_mu,nu,t.hpp"
			const Eigen::Tensor<double, 3> AO1sa = SliceTensor(AO1s, {0, ihead, 0}, {ngrids, ilength, 3});
			#include "FockEinSum/Ws_g...E1Sigmas_g...Rho1s_g,r...AO1sa_g,mu,t...AO1s_g,nu,r---Fa1_mu,nu,t.hpp"
			F.chip(iatom, 3) -= 2. * Fa1;
		}
	}
	F += F.shuffle(Eigen::array<int, 4>{1, 0, 2, 3}).eval();
	std::vector<EigenMatrix> Fmats(3 * natoms, EigenZero(nbasis, nbasis));
	for ( int iatom = 0, kpert = 0; iatom < natoms; iatom++ ) for ( int t = 0; t < 3; t++, kpert++ ){
		Eigen::Tensor<double, 2> Fmat = F.chip(iatom, 3).chip(t, 2);
		Fmats[kpert] = Eigen::Map<EigenMatrix>(Fmat.data(), nbasis, nbasis);
	}
	return Fmats;
}

std::vector<EigenMatrix> Grid::getFockDensity(){
	const Eigen::Tensor<double, 1>& Ws = Weights;
	const int nbasis = this->Mwfn->getNumBasis();
	const int natoms = this->Mwfn->getNumCenters();
	const int nxyz = 3;
	Eigen::Tensor<double, 4> F(nbasis, nbasis, nxyz, natoms); F.setZero();
	if ( this->Type >= 0 ){
		Eigen::Tensor<double, 4> F0(nbasis, nbasis, nxyz, natoms); F0.setZero();
		#include "FockEinSum/Ws_g...E2Rho2s_g...RhoGrads_g,t,a...AOs_g,mu...AOs_g,nu---F0_mu,nu,t,a.hpp"
		F += 0.5 * F0;
	}
	if ( this->Type >= 1 ){
		Eigen::Tensor<double, 4> F1(nbasis, nbasis, nxyz, natoms); F1.setZero();
		#include "FockEinSum/Ws_g...E2RhoSigmas_g...SigmaGrads_g,t,a...AOs_g,mu...AOs_g,nu---F1_mu,nu,t,a.hpp"
		F += 0.5 * F1;
		F1.setZero();
		#include "FockEinSum/Ws_g...E2RhoSigmas_g...RhoGrads_g,t,a...Rho1s_g,r...AO1s_g,mu,r...AOs_g,nu---F1_mu,nu,t,a.hpp"
		F += 2. * F1;
		F1.setZero();
		#include "FockEinSum/Ws_g...E2Sigma2s_g...SigmaGrads_g,t,a...Rho1s_g,r...AO1s_g,mu,r...AOs_g,nu---F1_mu,nu,t,a.hpp"
		F += 2. * F1;
		F1.setZero();
		#include "FockEinSum/Ws_g...E1Sigmas_g...Rho1Grads_g,r,t,a...AO1s_g,mu,r...AOs_g,nu---F1_mu,nu,t,a.hpp"
		F += 2. * F1;
	}
	F += F.shuffle(Eigen::array<int, 4>{1, 0, 2, 3}).eval();
	std::vector<EigenMatrix> Fmats(nxyz * natoms, EigenZero(nbasis, nbasis));
	for ( int iatom = 0, kpert = 0; iatom < natoms; iatom++ ) for ( int t = 0; t < nxyz; t++, kpert++ ){
		Eigen::Tensor<double, 2> Fmat = F.chip(iatom, 3).chip(t, 2);
		Fmats[kpert] = Eigen::Map<EigenMatrix>(Fmat.data(), nbasis, nbasis);
	}
	return Fmats;
}

EigenMatrix Grid::getFockDensitySelf(){
	const Eigen::Tensor<double, 1>& Ws = Weights;
	const int nbasis = this->Mwfn->getNumBasis();
	const int natoms = 1;
	const int nxyz = 1;
	const int ngrids = this->NumGrids;
	Eigen::Tensor<double, 3> RhoGrads(ngrids, 1, 1); RhoGrads.chip(0, 2).chip(0, 1) = this->Rhos_Cache;
	Eigen::Tensor<double, 4> Rho1Grads(ngrids, 3, 1, 1); Rho1Grads.chip(0, 3).chip(0, 2) = this->Rho1s_Cache;
	Eigen::Tensor<double, 3> SigmaGrads(ngrids, 1, 1); SigmaGrads.chip(0, 2).chip(0, 1) = this->Sigmas_Cache;
	Eigen::Tensor<double, 4> F(nbasis, nbasis, nxyz, natoms); F.setZero();
	if ( this->Type >= 0 ){
		Eigen::Tensor<double, 4> F0(nbasis, nbasis, nxyz, natoms); F0.setZero();
		#include "FockEinSum/Ws_g...E2Rho2s_g...RhoGrads_g,t,a...AOs_g,mu...AOs_g,nu---F0_mu,nu,t,a.hpp"
		F += 0.5 * F0;
	}
	if ( this->Type >= 1 ){
		Eigen::Tensor<double, 4> F1(nbasis, nbasis, nxyz, natoms); F1.setZero();
		#include "FockEinSum/Ws_g...E2RhoSigmas_g...SigmaGrads_g,t,a...AOs_g,mu...AOs_g,nu---F1_mu,nu,t,a.hpp"
		F += 0.5 * F1;
		F1.setZero();
		#include "FockEinSum/Ws_g...E2RhoSigmas_g...RhoGrads_g,t,a...Rho1s_g,r...AO1s_g,mu,r...AOs_g,nu---F1_mu,nu,t,a.hpp"
		F += 2. * F1;
		F1.setZero();
		#include "FockEinSum/Ws_g...E2Sigma2s_g...SigmaGrads_g,t,a...Rho1s_g,r...AO1s_g,mu,r...AOs_g,nu---F1_mu,nu,t,a.hpp"
		F += 2. * F1;
		F1.setZero();
		#include "FockEinSum/Ws_g...E1Sigmas_g...Rho1Grads_g,r,t,a...AO1s_g,mu,r...AOs_g,nu---F1_mu,nu,t,a.hpp"
		F += 2. * F1;
	}
	F += F.shuffle(Eigen::array<int, 4>{1, 0, 2, 3}).eval();
	std::vector<EigenMatrix> Fmats(nxyz * natoms, EigenZero(nbasis, nbasis));
	for ( int iatom = 0, kpert = 0; iatom < natoms; iatom++ ) for ( int t = 0; t < nxyz; t++, kpert++ ){
		Eigen::Tensor<double, 2> Fmat = F.chip(iatom, 3).chip(t, 2);
		Fmats[kpert] = Eigen::Map<EigenMatrix>(Fmat.data(), nbasis, nbasis);
	}
	return Fmats[0];
}
