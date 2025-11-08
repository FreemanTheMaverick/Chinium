#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <string>
#include <vector>
#include <libmwfn.h>

#include "../Macro.h"
#include "Grid.h"

void SubGrid::getFock(EigenTensor<3>& F_){
	const int ngrids = this->NumGrids;
	const int nspins = this->Spin;
	const int nbasis = this->getNumBasis();
	EigenTensor<3> F(nbasis, nbasis, nspins); F.setZero();
	if ( this->Type >= 0 ){
		EigenTensor<3> F0(nbasis, nbasis, nspins); F0.setZero();
		#include "FockEinSum/W_g...Eps1Rho_g,w...AO_g,mu...AO_g,nu---F0_mu,nu,w.hpp"
		F += 0.5 * F0;
	}
	if ( this->Type >= 1 ){
		EigenTensor<1> S(nspins * ( nspins + 1 ) / 2); // Scaling factors
		S.setConstant(1);
		for ( int w = 0; w < nspins; w++ ) S(( w + 1 ) * ( w + 2 ) / 2 - 1) = 2;
		EigenTensor<2> ScaledEps1Sigma(ngrids, nspins * ( nspins + 1 ) / 2);
		ScaledEps1Sigma.setZero();
		#include "FockEinSum/S_w...Eps1Sigma_g,w---ScaledEps1Sigma_g,w.hpp"
		EigenTensor<3> F1(nbasis, nbasis, nspins); F1.setZero();
		#include "FockEinSum/W_g...ScaledEps1Sigma_g,u+v...Rho1_g,r,v...AO1_g,mu,r...AO_g,nu---F1_mu,nu,u.hpp"
		F += F1;
	}
	if ( this->Type >= 2 ){
		EigenTensor<3> F2(nbasis, nbasis, nspins); F2.setZero();
		EigenTensor<2> V = 0.5 * Eps1Tau + 2 * Eps1Lapl;
		#include "FockEinSum/W_g...V_g,w...AO1_g,mu,r...AO1_g,nu,r---F2_mu,nu,w.hpp"
		F += 0.5 * F2;
		F2.setZero();
		#include "FockEinSum/W_g...Eps1Lapl_g,w...AO_g,mu...AO2L_g,nu---F2_mu,nu,w.hpp"
		F += F2;
	}
	F += F.shuffle(Eigen::array<int, 3>{1, 0, 2}).eval();
	for ( int spin = 0; spin < nspins; spin++ ){
		for ( int nu = 0; nu < nbasis; nu++ ){
			for ( int mu = 0; mu < nbasis; mu++ ){
				F_(this->BasisList[mu], this->BasisList[nu], spin) += F(mu, nu, spin);
			}
		}
	}
}

void SubGrid::getFockSkeleton(EigenTensor<5>& F_){
	const int nbasis = this->getNumBasis();
	const int ngrids = this->NumGrids;
	const int natoms = this->getNumAtoms();
	const int nspins = this->Spin;
	EigenTensor<5> F(nbasis, nbasis, 3, natoms, nspins); F.setZero();
	if ( this->Type >= 0 ){
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = this->AtomHeads[iatom];
			const int ilength = this->AtomLengths[iatom];
			const EigenTensor<3> AO1a = SliceTensor(AO1, {0, ihead, 0}, {ngrids, ilength, 3});
			EigenTensor<4> F0a(ilength, nbasis, 3, nspins); F0a.setZero();
			#include "FockEinSum/W_g...Eps1Rho_g,w...AO1a_g,mu,t...AO_g,nu---F0a_mu,nu,t,w.hpp"
			F.chip(iatom, 3) -= PadTensor(F0a, {{ihead, nbasis - ihead - ilength}, {0, 0}, {0, 0}, {0, 0}});
		}
	}
	if ( this->Type >= 1 ){
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			EigenTensor<1> S(nspins * ( nspins + 1 ) / 2); // Scaling factors
			S.setConstant(1);
			for ( int w = 0; w < nspins; w++ ) S(( w + 1 ) * ( w + 2 ) / 2 - 1) = 2;
			EigenTensor<2> ScaledEps1Sigma(ngrids, nspins * ( nspins + 1 ) / 2);
			ScaledEps1Sigma.setZero();
			#include "FockEinSum/S_w...Eps1Sigma_g,w---ScaledEps1Sigma_g,w.hpp"
			const int ihead = this->AtomHeads[iatom];
			const int ilength = this->AtomLengths[iatom];
			EigenTensor<4> Fa1(ilength, nbasis, 3, nspins); Fa1.setZero();
			const EigenTensor<3> AO2a = SliceTensor(AO2, {0, ihead, 0}, {ngrids, ilength, 6});
			#include "FockEinSum/W_g...ScaledEps1Sigma_g,u+v...Rho1_g,r,v...AO2a_g,mu,r+t...AO_g,nu---Fa1_mu,nu,t,u.hpp"
			const EigenTensor<3> AO1a = SliceTensor(AO1, {0, ihead, 0}, {ngrids, ilength, 3});
			#include "FockEinSum/W_g...ScaledEps1Sigma_g,u+v...Rho1_g,r,v...AO1a_g,mu,t...AO1_g,nu,r---Fa1_mu,nu,t,u.hpp"
			F.chip(iatom, 3) -= PadTensor(Fa1, {{ihead, nbasis - ihead - ilength}, {0, 0}, {0, 0}, {0, 0}});
		}
	}
	F += F.shuffle(Eigen::array<int, 5>{1, 0, 2, 3, 4}).eval();
	for ( int spin = 0; spin < nspins; spin++ ){
		for ( int iatom = 0; iatom < natoms; iatom++ ) for ( int t = 0; t < 3; t++ ){
			for ( int mu = 0; mu < nbasis; mu++ ){
				for ( int nu = 0; nu < nbasis; nu++ ){
					F_(this->BasisList[mu], this->BasisList[nu], t, this->AtomList[iatom], spin) += F(mu, nu, t, iatom, spin);
				}
			}
		}
	}
}

// enum D_t{ s_t, u_t };
template <D_t d_t>
void SubGrid::getFockU(EigenTensor<4>& F_){
	const int ngrids = this->NumGrids;
	const int nmats = F_.dimension(2);
	const int nbasis = this->getNumBasis();
	const int nspins = F_.dimension(3);
	const Eigen::TensorMap<EigenTensor<3>> RhoG( [&RhoGrad = this->RhoGrad, &RhoU = this->RhoU]() -> double* {
			if constexpr ( d_t == s_t ) return RhoGrad.data();
			else return RhoU.data();
	}(), this->Type >= 0 ? ngrids : 0, nmats, nspins );
	const Eigen::TensorMap<EigenTensor<4>> Rho1G( [&Rho1Grad = this->Rho1Grad, &Rho1U = this->Rho1U]() -> double* {
			if constexpr ( d_t == s_t ) return Rho1Grad.data();
			else return Rho1U.data();
	}(), this->Type >= 1 ? ngrids : 0, 3, nmats, nspins );
	const Eigen::TensorMap<EigenTensor<3>> SigmaG( [&SigmaGrad = this->SigmaGrad, &SigmaU = this->SigmaU]() -> double* {
			if constexpr ( d_t == s_t ) return SigmaGrad.data();
			else return SigmaU.data();
	}(), this->Type >= 1 ? ngrids : 0, nmats, nspins );
	EigenTensor<4> F(nbasis, nbasis, nmats, nspins); F.setZero();
	if ( this->Type >= 0 ){
		EigenTensor<3> TMP0(ngrids, nmats, nspins); TMP0.setZero();
		#include "FockEinSum/Eps2Rho2_g,u,v...RhoG_g,mat,v---TMP0_g,mat,u.hpp"
		if ( this->Type >= 1 ){
			#include "FockEinSum/Eps2RhoSigma_g,w,uv...SigmaG_g,mat,uv---TMP0_g,mat,w.hpp"
		}
		EigenTensor<3> TMP1(ngrids, nmats, nspins); TMP1.setZero();
		#include "FockEinSum/W_g...TMP0_g,mat,w---TMP1_g,mat,w.hpp"
		EigenTensor<4> F0(nbasis, nbasis, nmats, nspins); F0.setZero();
		#include "FockEinSum/TMP1_g,mat,w...AO_g,mu...AO_g,nu---F0_mu,nu,mat,w.hpp"
//std::cout<<F0.chip(0, 3)<<std::endl;
		F += 0.5 * F0;
	}
	if ( this->Type >= 1 ){
		EigenTensor<1> S(nspins * ( nspins + 1 ) / 2); // Scaling factors
		S.setConstant(1);
		for ( int w = 0; w < nspins; w++ ) S(( w + 1 ) * ( w + 2 ) / 2 - 1) = 2;
		EigenTensor<3> ScaledEps2RhoSigma(ngrids, nspins, nspins * ( nspins + 1 ) / 2);
		ScaledEps2RhoSigma.setZero();
		#include "FockEinSum/S_x...Eps2RhoSigma_g,w,x---ScaledEps2RhoSigma_g,w,x.hpp"
		EigenTensor<3> ScaledEps2Sigma2(ngrids, nspins * ( nspins + 1 ) / 2, nspins * ( nspins + 1 ) / 2);
		ScaledEps2Sigma2.setZero();
		#include "FockEinSum/S_x...Eps2Sigma2_g,w,x---ScaledEps2Sigma2_g,w,x.hpp"
		EigenTensor<2> ScaledEps1Sigma(ngrids, nspins * ( nspins + 1 ) / 2);
		ScaledEps1Sigma.setZero();
		#include "FockEinSum/S_w...Eps1Sigma_g,w---ScaledEps1Sigma_g,w.hpp"

		EigenTensor<3> TMP1(ngrids, nmats, nspins * ( nspins + 1 ) / 2); TMP1.setZero();
		#include "FockEinSum/ScaledEps2RhoSigma_g,w,uv...RhoG_g,mat,w---TMP1_g,mat,uv.hpp"
		#include "FockEinSum/ScaledEps2Sigma2_g,u,v...SigmaG_g,mat,v---TMP1_g,mat,u.hpp"
		EigenTensor<4> TMP2(ngrids, 3, nmats, nspins); TMP2.setZero();
		#include "FockEinSum/TMP1_g,mat,u+v...Rho1_g,r,v---TMP2_g,r,mat,u.hpp"
		EigenTensor<4> TMP3(ngrids, 3, nmats, nspins); TMP3.setZero();
		#include "FockEinSum/ScaledEps1Sigma_g,u+v...Rho1G_g,r,mat,v---TMP3_g,r,mat,u.hpp"
		EigenTensor<4> TMP4 = TMP2 + TMP3;
		EigenTensor<4> F1(nbasis, nbasis, nmats, nspins); F1.setZero();
		#include "FockEinSum/W_g...TMP4_g,r,mat,u...AO1_g,mu,r...AO_g,nu---F1_mu,nu,mat,u.hpp"
		F += F1;
	}
	F += F.shuffle(Eigen::array<int, 4>{1, 0, 2, 3}).eval();
	for ( int spin = 0; spin < nspins; spin++ ) for ( int mat = 0; mat < nmats; mat++ ){
		for ( int mu = 0; mu < nbasis; mu++ ) for ( int nu = 0; nu < nbasis; nu++ ){
			F_(this->BasisList[mu], this->BasisList[nu], mat, spin) += F(mu, nu, mat, spin);
		}
	}
}
template void SubGrid::getFockU<s_t>(EigenTensor<4>& F_); // Using skeleton derivatives of density
template void SubGrid::getFockU<u_t>(EigenTensor<4>& F_); // Using U derivatives of density
