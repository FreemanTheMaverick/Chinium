#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>

#include "../Macro.h"
#include "../MwfnIO/MwfnIO.h"
#include "Grid.h"

EigenMatrix Grid::getFock(int type){ // Default type = -1
	if ( type == -1 ) type = this->Type;
	const Eigen::Tensor<double, 1>& Ws = Weights;
	const int nbasis = this->MWFN->getNumBasis();
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
	const int nbasis = this->MWFN->getNumBasis();
	const int ngrids = this->NumGrids;
	const int natoms = this->MWFN->getNumCenters();
	const std::vector<int> atom2bf = this->MWFN->Atom2Basis();
	Eigen::Tensor<double, 4> F(nbasis, nbasis, 3, natoms); F.setZero();
	if ( this->Type >= 0 ){
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = atom2bf[iatom];
			const int ilength = this->MWFN->Centers[iatom].getNumBasis();
			const Eigen::Tensor<double, 3> AO1sa = SliceTensor(AO1s, {0, ihead, 0}, {ngrids, ilength, 3});
			Eigen::Tensor<double, 3> F0a(ilength, nbasis, 3); F0a.setZero();
			#include "FockEinSum/Ws_g...E1Rhos_g...AO1sa_g,mu,t...AOs_g,nu---F0a_mu,nu,t.hpp"
			F.chip(iatom, 3) -= PadTensor(F0a, {{ihead, nbasis - ihead - ilength}, {0, 0}, {0, 0}});
		}
	}
	if ( this->Type >= 1 ){
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = atom2bf[iatom];
			const int ilength = this->MWFN->Centers[iatom].getNumBasis();
			Eigen::Tensor<double, 3> Fa1(ilength, nbasis, 3); Fa1.setZero();
			const Eigen::Tensor<double, 3> AO2sa = SliceTensor(AO2s, {0, ihead, 0}, {ngrids, ilength, 6});
			#include "FockEinSum/Ws_g...E1Sigmas_g...Rho1s_g,r...AO2sa_g,mu,r,t...AOs_g,nu---Fa1_mu,nu,t.hpp"
			const Eigen::Tensor<double, 3> AO1sa = SliceTensor(AO1s, {0, ihead, 0}, {ngrids, ilength, 3});
			#include "FockEinSum/Ws_g...E1Sigmas_g...Rho1s_g,r...AO1sa_g,mu,t...AO1s_g,nu,r---Fa1_mu,nu,t.hpp"
			F.chip(iatom, 3) -= 2. * PadTensor(Fa1, {{ihead, nbasis - ihead - ilength}, {0, 0}, {0, 0}});
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

void getFockU(
		Eigen::Tensor<double, 1>& Ws,
		Eigen::Tensor<double, 2>& AOs,
		Eigen::Tensor<double, 3>& AO1s,
		Eigen::Tensor<double, 2>& Rho1s,
		Eigen::Tensor<double, 1>& E1Sigmas,
		Eigen::Tensor<double, 1>& E2Rho2s,
		Eigen::Tensor<double, 1>& E2RhoSigmas,
		Eigen::Tensor<double, 1>& E2Sigma2s,
		Eigen::Tensor<double, 1>& RhoGrad,
		Eigen::Tensor<double, 2>& Rho1Grad,
		Eigen::Tensor<double, 1>& SigmaGrad,
		EigenMatrix& Fmat){
	const int ngrids = Ws.dimension(0);
	const int nbasis = Fmat.cols();
	Eigen::Tensor<double, 2> F(nbasis, nbasis); F.setZero();
	if (
			( E2Rho2s.size() && RhoGrad.size() ) ||
			( E2RhoSigmas.size() && SigmaGrad.size() )
	){
		Eigen::Tensor<double, 1> TMP0(ngrids); TMP0.setZero();
		if ( E2Rho2s.size() && RhoGrad.size() && AOs.size() ){
			TMP0 += E2Rho2s * RhoGrad;
		}
		if ( E2RhoSigmas.size() && SigmaGrad.size() && AOs.size() ){
			TMP0 += E2RhoSigmas * SigmaGrad;
		}
		TMP0 *= Ws;
		Eigen::Tensor<double, 2> F0(nbasis, nbasis); F0.setZero();
		#include "FockEinSum/TMP0_g...AOs_g,mu...AOs_g,nu---F0_mu,nu.hpp"
		F += 0.5 * F0;
	}
	if ( E2RhoSigmas.size() && RhoGrad.size() && E2Sigma2s.size() && SigmaGrad.size() && Rho1s.size() && Rho1Grad.size() && AO1s.size() && AOs.size() ){
		Eigen::Tensor<double, 1> TMP1 = E2RhoSigmas * RhoGrad + E2Sigma2s * SigmaGrad;
		Eigen::Tensor<double, 2> TMP2(ngrids, 3); TMP2.setZero();
		#include "FockEinSum/TMP1_g...Rho1s_g,r---TMP2_g,r.hpp"
		Eigen::Tensor<double, 2> TMP3(ngrids, 3); TMP3.setZero();
		#include "FockEinSum/E1Sigmas_g...Rho1Grad_g,r---TMP3_g,r.hpp"
		Eigen::Tensor<double, 2> TMP4 = TMP2 + TMP3;
		Eigen::Tensor<double, 2> F1(nbasis, nbasis); F1.setZero();
		#include "FockEinSum/Ws_g...TMP4_g,r...AO1s_g,mu,r...AOs_g,nu---F1_mu,nu.hpp"
		F += 2. * F1;
	}
	F += F.shuffle(Eigen::array<int, 2>{1, 0}).eval();
	Fmat = Eigen::Map<EigenMatrix>(F.data(), nbasis, nbasis);
}

std::vector<EigenMatrix> Grid::getFockU(){
	const int nbasis = this->MWFN->getNumBasis();
	const int natoms = this->MWFN->getNumCenters();
	Eigen::Tensor<double, 1>& Ws = Weights;
	std::vector<EigenMatrix> Fmats(3 * natoms, EigenZero(nbasis, nbasis));
	Eigen::Tensor<double, 1> dummy1;
	Eigen::Tensor<double, 2> dummy2;
	for ( int a = 0; a < this->MWFN->getNumCenters(); a++ ) for ( int t = 0; t < 3; t++ ){
		Eigen::Tensor<double, 1> RhoGrad = dummy1;
		Eigen::Tensor<double, 2> Rho1Grad = dummy2;
		Eigen::Tensor<double, 1> SigmaGrad = dummy1;
		if ( this->Type >= 0 ){
			RhoGrad = RhoGrads.chip(a, 2).chip(t, 1);
		}
		if ( this->Type >= 1 ){
			Rho1Grad = Rho1Grads.chip(a, 3).chip(t, 2);
			SigmaGrad = SigmaGrads.chip(a, 2).chip(t, 1);
		}
		::getFockU(
			Ws, AOs, AO1s, Rho1s,
			E1Sigmas, E2Rho2s, E2RhoSigmas, E2Sigma2s,
			RhoGrad, Rho1Grad, SigmaGrad,
			Fmats[3 * a + t]
		);
	}
	return Fmats;
}

std::vector<EigenMatrix> Grid::getFockU(
		std::vector<Eigen::Tensor<double, 1>>& Rhoss,
		std::vector<Eigen::Tensor<double, 2>>& Rho1ss,
		std::vector<Eigen::Tensor<double, 1>>& Sigmass){
	const int nmats = (int)Rhoss.size();
	if ( this->Type >= 1 ){
		assert( nmats == (int)Rho1ss.size() );
		assert( nmats == (int)Sigmass.size() );
	}
	const int nbasis = this->MWFN->getNumBasis();
	Eigen::Tensor<double, 1>& Ws = Weights;
	std::vector<EigenMatrix> Fmats(nmats, EigenZero(nbasis, nbasis));
	for ( int imat = 0; imat < nmats; imat++ ){
		::getFockU(
			Ws, AOs, AO1s, Rho1s,
			E1Sigmas, E2Rho2s, E2RhoSigmas, E2Sigma2s,
			Rhoss[imat], Rho1ss[imat], Sigmass[imat],
			Fmats[imat]
		);
	}
	return Fmats;
}
