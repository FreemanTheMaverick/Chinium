#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <string>
#include <vector>
#include <libmwfn.h>

#include "../Macro.h"
#include "Tensor.h"
#include "Grid.h"

void SubGrid::getFock(EigenMatrix& F_){
	const int nbasis = this->getNumBasis();
	EigenTensor<2> F(nbasis, nbasis); F.setZero();
	if ( this->Type >= 0 ){
		EigenTensor<2> F0(nbasis, nbasis); F0.setZero();
		#include "FockEinSum/W_g...E1Rho_g...AO_g,mu...AO_g,nu---F0_mu,nu.hpp"
		F += 0.5 * F0;
	}
	if ( this->Type >= 1 ){
		EigenTensor<2> F1(nbasis, nbasis); F1.setZero();
		#include "FockEinSum/W_g...E1Sigma_g...Rho1_g,r...AO1_g,mu,r...AO_g,nu---F1_mu,nu.hpp"
		F += 2. * F1;
	}
	if ( this->Type >= 2 ){
		EigenTensor<2> F2(nbasis, nbasis); F2.setZero();
		EigenTensor<1> V = 0.5 * E1Tau + 2 * E1Lapl;
		#include "FockEinSum/W_g...V_g...AO1_g,mu,r...AO1_g,nu,r---F2_mu,nu.hpp"
		F += 0.5 * F2;
		F2.setZero();
		#include "FockEinSum/W_g...E1Lapl_g...AO_g,mu...AO2L_g,nu---F2_mu,nu.hpp"
		F += F2;
	}
	F += F.shuffle(Eigen::array<int, 2>{1, 0}).eval();
	for ( int mu = 0; mu < nbasis; mu++ ){
		for ( int nu = 0; nu < nbasis; nu++ ){
			F_(this->BasisList[mu], this->BasisList[nu]) += F(mu, nu);
		}
	}
}

void SubGrid::getFockSkeleton(std::vector<EigenMatrix>& Fs){
	const int nbasis = this->getNumBasis();
	const int ngrids = this->NumGrids;
	const int natoms = this->getNumAtoms();
	EigenTensor<4> F(nbasis, nbasis, 3, natoms); F.setZero();
	if ( this->Type >= 0 ){
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = this->AtomHeads[iatom];
			const int ilength = this->AtomLengths[iatom];
			const EigenTensor<3> AO1a = SliceTensor(AO1, {0, ihead, 0}, {ngrids, ilength, 3});
			EigenTensor<3> F0a(ilength, nbasis, 3); F0a.setZero();
			#include "FockEinSum/W_g...E1Rho_g...AO1a_g,mu,t...AO_g,nu---F0a_mu,nu,t.hpp"
			F.chip(iatom, 3) -= PadTensor(F0a, {{ihead, nbasis - ihead - ilength}, {0, 0}, {0, 0}});
		}
	}
	if ( this->Type >= 1 ){
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = this->AtomHeads[iatom];
			const int ilength = this->AtomLengths[iatom];
			EigenTensor<3> Fa1(ilength, nbasis, 3); Fa1.setZero();
			const EigenTensor<3> AO2a = SliceTensor(AO2, {0, ihead, 0}, {ngrids, ilength, 6});
			#include "FockEinSum/W_g...E1Sigma_g...Rho1_g,r...AO2a_g,mu,r,t...AO_g,nu---Fa1_mu,nu,t.hpp"
			const EigenTensor<3> AO1a = SliceTensor(AO1, {0, ihead, 0}, {ngrids, ilength, 3});
			#include "FockEinSum/W_g...E1Sigma_g...Rho1_g,r...AO1a_g,mu,t...AO1_g,nu,r---Fa1_mu,nu,t.hpp"
			F.chip(iatom, 3) -= 2. * PadTensor(Fa1, {{ihead, nbasis - ihead - ilength}, {0, 0}, {0, 0}});
		}
	}
	F += F.shuffle(Eigen::array<int, 4>{1, 0, 2, 3}).eval();
	for ( int iatom = 0; iatom < natoms; iatom++ ) for ( int t = 0; t < 3; t++ ){
		for ( int mu = 0; mu < nbasis; mu++ ){
			for ( int nu = 0; nu < nbasis; nu++ ){
				Fs[3 * this->AtomList[iatom] + t](this->BasisList[mu], this->BasisList[nu]) += F(mu, nu, t, iatom);
			}
		}
	}
}

// enum D_t{ s_t, u_t };
template <D_t d_t>
void SubGrid::getFockU(std::vector<EigenMatrix>& Fs){
	const int ngrids = this->NumGrids;
	const int nmats = (int)Fs.size();
	const int nbasis = this->getNumBasis();
	const Eigen::TensorMap<EigenTensor<2>> RhoG( []() -> double* {
			if constexpr (std::is_same_v<d_t, s_t>) return RhoGrad.data();
			else return RhoU.data();
	}, this->Type >= 0 ? ngrids : 0, nmats );
	const Eigen:TensorMap<EigenTensor<3>> Rho1G( []() -> double* {
			if constexpr (std::is_same_v<d_t, s_t>) return Rho1Grad.data();
			else return Rho1U.data();
	}, this->Type >= 1 ? ngrids : 0, 3, nmats );
	const Eigen:TensorMap<EigenTensor<2>> SigmaG( []() -> double* {
			if constexpr (std::is_same_v<d_t, s_t>) return SigmaGrad.data();
			else return SigmaU.data();
	}, this->Type >= 1 ? ngrids : 0, nmats );
	EigenTensor<3> F(nbasis, nbasis, nmats); F.setZero();
	if ( this->Type >= 0 ){
		EigenTensor<2> TMP0(ngrids, nmats); TMP0.setZero();
		#include "FockEinSum/E2Rho2_g...RhoG_g,mat---TMP0_mat.hpp"
		if ( this->Type >= 1 ){
			#include "FockEinSum/E2RhoSigma_g...SigmaG_g,mat---TMP0_mat.hpp"
		}
		EigenTensor<2> TMP1(ngrids, nmats); TMP1.setZero();
		#include "FockEinSum/W_g...TMP0_g,mat---TMP1_g,mat.hpp"
		EigenTensor<3> F0(nbasis, nbasis, nmats); F0.setZero();
		#include "FockEinSum/TMP1_g,mat...AO_g,mu...AO_g,nu---F0_mu,nu,mat.hpp"
		F += 0.5 * F0;
	}
	if ( this->Type >= 1 ){
		EigenTensor<2> TMP1(ngrids, nmats); TMP1.setZero();
		#include "FockEinSum/E2RhoSigma_g...RhoG_g,mat---TMP1_g,mat.hpp"
		#include "FockEinSum/E2Sigma2_g...SigmaG_g,mat---TMP1_g,mat.hpp"
		EigenTensor<3> TMP2(ngrids, 3, nmats); TMP2.setZero();
		#include "FockEinSum/TMP1_g,mat...Rho1_g,r---TMP2_g,r,mat.hpp"
		EigenTensor<3> TMP3(ngrids, 3, nmats); TMP3.setZero();
		#include "FockEinSum/E1Sigma_g...Rho1Grad_g,r,mat---TMP3_g,r,mat.hpp"
		EigenTensor<3> TMP4 = TMP2 + TMP3;
		EigenTensor<3> F1(nbasis, nbasis, nmats); F1.setZero();
		#include "FockEinSum/W_g...TMP4_g,r,mat...AO1_g,mu,r...AO_g,nu---F1_mu,nu,mat.hpp"
		F += 2. * F1;
	}
	F += F.shuffle(Eigen::array<int, 3>{1, 0, 2}).eval();
	for ( int imat = 0; imat < nmats; imat++ ){
		for ( int mu = 0; mu < nbasis; mu++ ){
			for ( int nu = 0; nu < nbasis; nu++ ){
				Fs[imat](this->BasisList[mu], this->BasisList[nu]) += F(mu, nu, imat);
			}
		}
	}
}
template void SubGrid::getFockU<s_t>(std::vector<EigenMatrix>& Fs); // Using skeleton derivatives of density
template void SubGrid::getFockU<u_t>(std::vector<EigenMatrix>& Fs); // Using U derivatives of density
