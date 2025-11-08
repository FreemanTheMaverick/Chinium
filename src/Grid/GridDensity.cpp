#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cassert>
#include <libmwfn.h>

#include "../Macro.h"
#include "Grid.h"

void SubGrid::getDensity(EigenTensor<3>& D_){
	const int ngrids = this->NumGrids;
	const int nspins = this->Spin;
	const int nbasis = this->getNumBasis();
	EigenTensor<3> D(nbasis, nbasis, nspins);
	for ( int spin = 0; spin < nspins; spin++ ){
		for ( int nu = 0; nu < nbasis; nu++ ) for ( int mu = 0; mu < nbasis; mu++ ){
			D(mu, nu, spin) = D_(this->BasisList[mu], this->BasisList[nu], spin);
		}
	}
	if ( this->Type >= 0 ){
		Rho.resize(ngrids, nspins); Rho.setZero();
		#include "DensityEinSum/D_mu,nu,w...AO_g,mu...AO_g,nu---Rho_g,w.hpp"
	}
	if ( this->Type >= 1 ){
		Rho1.resize(ngrids, 3, nspins); Rho1.setZero();
		#include "DensityEinSum/D_mu,nu,w...AO1_g,mu,r...AO_g,nu---Rho1_g,r,w.hpp"
		ScaleTensor(Rho1, 2);
		Sigma.resize(ngrids, nspins * ( nspins + 1 ) / 2); Sigma.setZero();
		#include "DensityEinSum/Rho1_g,r,u...Rho1_g,r,v---Sigma_g,u+v.hpp"
	}
	if ( this->Type >= 2 ){
		Tau.resize(ngrids, nspins); Tau.setZero();
		#include "DensityEinSum/D_mu,nu,w...AO1_g,mu,r...AO1_g,nu,r---Tau_g,w.hpp"
		ScaleTensor(Tau, 0.5);
		Lapl = 4. * Tau;
		#include "DensityEinSum/D_mu,nu,w...AO_g,mu...AO2L_g,nu---Lapl_g,w.hpp"
		#include "DensityEinSum/D_mu,nu,w...AO2L_g,mu...AO_g,nu---Lapl_g,w.hpp"
	}
}

void SubGrid::getNumElectrons(EigenTensor<0>& n){
	#include "DensityEinSum/W_g...Rho_g,w---n_.hpp"
}

void SubGrid::getDensityU(EigenTensor<4>& D_){
	const int nbasis = this->getNumBasis();
	const int ngrids = this->NumGrids;
	const int nspins = this->Spin;
	const int nmats = D_.dimension(2);
	assert( nspins == D_.dimension(3) && "Inconsistent number of spin types!" );
	EigenTensor<4> D(nbasis, nbasis, nmats, nspins);
	for ( int spin = 0; spin < nspins; spin++ ){
		for ( int mat = 0; mat < nmats; mat++ ){
			for ( int nu = 0; nu < nbasis; nu++ ) for ( int mu = 0; mu < nbasis; mu++ ){
				D(mu, nu, mat, spin) = D_(this->BasisList[mu], this->BasisList[nu], mat, spin);
			}
		}
	}
	if ( this->Type >= 0 ){
		RhoU.resize(ngrids, nmats, nspins); RhoU.setZero();
		#include "DensityEinSum/D_mu,nu,mat,w...AO_g,mu...AO_g,nu---RhoU_g,mat,w.hpp"
	}
	if ( this->Type >= 1 ){
		Rho1U.resize(ngrids, 3, nmats, nspins); Rho1U.setZero();
		#include "DensityEinSum/D_mu,nu,mat,w...AO1_g,mu,r...AO_g,nu---Rho1U_g,r,mat,w.hpp"
		ScaleTensor(Rho1U, 2);
		SigmaU.resize(ngrids, nmats, nspins * ( nspins + 1 ) / 2); SigmaU.setZero();
		#include "DensityEinSum/Rho1U_g,r,mat,u...Rho1_g,r,v---SigmaU_g,mat,u+v.hpp"
		ScaleTensor(SigmaU, 2);
	}
}

void SubGrid::getDensitySkeleton(EigenTensor<3>& D_){
	const int nspins = this->Spin;
	const int nbasis = this->getNumBasis();
	const int ngrids = this->NumGrids;
	const int natoms = this->getNumAtoms();
	EigenTensor<3> D(nbasis, nbasis, nspins);
	for ( int spin = 0; spin < nspins; spin++ ){
		for ( int nu = 0; nu < nbasis; nu++ ) for ( int mu = 0; mu < nbasis; mu++ ){
			D(mu, nu, spin) = D_(this->BasisList[mu], this->BasisList[nu], spin);
		}
	}
	if ( this->Type >= 0 ){
		RhoGrad.resize(ngrids, 3, natoms, nspins); RhoGrad.setZero();
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int head = this->AtomHeads[iatom];
			const int length = this->AtomLengths[iatom];
			const EigenTensor<3> Da = SliceTensor(D, {head, 0, 0}, {length, nbasis, nspins});
			const EigenTensor<3> AO1a = SliceTensor(AO1, {0, head, 0}, {ngrids, length, 3});
			EigenTensor<3> RhoGrada(ngrids, 3, nspins);
			RhoGrada.setZero();
			#include "DensityEinSum/Da_mu,nu,w...AO1a_g,mu,t...AO_g,nu---RhoGrada_g,t,w.hpp"
			RhoGrad.chip(iatom, 2) = RhoGrada;
		}
		ScaleTensor(RhoGrad, -2);
	}
	if ( this->Type >= 1 ){
		Rho1Grad.resize(ngrids, 3, 3, natoms, nspins); Rho1Grad.setZero();
		SigmaGrad.resize(ngrids, 3, natoms, nspins * ( nspins + 1 ) / 2); SigmaGrad.setZero();
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int head = this->AtomHeads[iatom];
			const int length = this->AtomLengths[iatom];
			const EigenTensor<3> Da = SliceTensor(D, {head, 0, 0}, {length, nbasis, nspins});
			const EigenTensor<3> AO2a = SliceTensor(AO2, {0, head, 0}, {ngrids, length, 6});
			EigenTensor<4> Rho1Grada(ngrids, 3, 3, nspins);
			Rho1Grada.setZero();
			#include "DensityEinSum/Da_mu,nu,w...AO2a_g,mu,r+t...AO_g,nu---Rho1Grada_g,r,t,w.hpp"
			const EigenTensor<3> AO1a = SliceTensor(AO1, {0, head, 0}, {ngrids, length, 3});
			#include "DensityEinSum/Da_mu,nu,w...AO1a_g,mu,t...AO1_g,nu,r---Rho1Grada_g,r,t,w.hpp"
			ScaleTensor(Rho1Grada, -2);
			Rho1Grad.chip(iatom, 3) = Rho1Grada;
			EigenTensor<4> SigmaGradaFull(ngrids, 3, nspins, nspins);
			SigmaGradaFull.setZero();
			#include "DensityEinSum/Rho1_g,r,u...Rho1Grada_g,r,t,v---SigmaGradaFull_g,t,u,v.hpp"
			SigmaGradaFull += SigmaGradaFull.shuffle(Eigen::array<int, 4>{0, 1, 3, 2});
			EigenTensor<3> SigmaGrada(ngrids, 3, nspins * ( nspins + 1 ) / 2);
			SigmaGrada.setZero();
			EigenTensor<1> One(nspins * ( nspins + 1 ) / 2); One.setConstant(1);
			#include "DensityEinSum/SigmaGradaFull_g,t,u,v...One_u+v---SigmaGrada_g,t,u+v.hpp"
			SigmaGrad.chip(iatom, 2) = SigmaGrada; // Putting SigmaGrad in upper triangular form
		}
	}
}

void SubGrid::getDensitySkeleton2(EigenTensor<3>& D_){
	const int nspins = this->Spin;
	const int nbasis = this->getNumBasis();
	const int ngrids = this->NumGrids;
	const int natoms = this->getNumAtoms();
	EigenTensor<3> D(nbasis, nbasis, nspins);
	for ( int spin = 0; spin < nspins; spin++ ){
		for ( int nu = 0; nu < nbasis; nu++ ) for ( int mu = 0; mu < nbasis; mu++ ){
			D(mu, nu, spin) = D_(this->BasisList[mu], this->BasisList[nu], spin);
		}
	}
	if ( this->Type >= 0 ){
		RhoHess.resize(ngrids, 3, natoms, 3, natoms, nspins); RhoHess.setZero();
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = this->AtomHeads[iatom];
			const int ilength = this->AtomLengths[iatom];
			const EigenTensor<3> Da = SliceTensor(D, {ihead, 0, 0}, {ilength, nbasis, nspins});
			const EigenTensor<3> AO2a = SliceTensor(AO2, {0, ihead, 0}, {ngrids, ilength, 6});
			EigenTensor<4> RhoHessaa(ngrids, 3, 3, nspins);
			RhoHessaa.setZero();
			#include "DensityEinSum/Da_mu,nu,w...AO2a_g,mu,t+s...AO_g,nu---RhoHessaa_g,t,s,w.hpp"
			const EigenTensor<3> Daa = SliceTensor(D, {ihead, ihead, 0}, {ilength, ilength, nspins});
			const EigenTensor<3> AO1a = SliceTensor(AO1, {0, ihead, 0}, {ngrids, ilength, 3});
			#include "DensityEinSum/Daa_mu,nu,w...AO1a_g,mu,t...AO1a_g,nu,s---RhoHessaa_g,t,s,w.hpp"
			RhoHess.chip(iatom, 4).chip(iatom, 2) = RhoHessaa;
			RhoHessaa.resize(0, 0, 0, 0);
			for ( int jatom = 0; jatom < iatom; jatom++ ){
				const int jhead = this->AtomHeads[jatom];
				const int jlength = this->AtomLengths[jatom];
				const EigenTensor<3> Dab = SliceTensor(D, {ihead, jhead, 0}, {ilength, jlength, nspins});
				const EigenTensor<3> AO1b = SliceTensor(AO1, {0, jhead, 0}, {ngrids, jlength, 3});
				EigenTensor<4> RhoHessab(ngrids, 3, 3, nspins);
				RhoHessab.setZero();
				#include "DensityEinSum/Dab_mu,nu,w...AO1a_g,mu,t...AO1b_g,nu,s---RhoHessab_g,t,s,w.hpp"
				RhoHess.chip(jatom, 4).chip(iatom, 2) = RhoHessab;
				RhoHess.chip(iatom, 4).chip(jatom, 2) = RhoHessab.shuffle(Eigen::array<int, 4>{0, 2, 1, 3});
			}
		}
		ScaleTensor(RhoHess, 2);
	}
	if ( this->Type >= 1 ){
		Rho1Hess.resize(ngrids, 3, 3, natoms, 3, natoms, nspins); Rho1Hess.setZero();
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = this->AtomHeads[iatom];
			const int ilength = this->AtomLengths[iatom];
			EigenTensor<5> Rho1Hessaa(ngrids, 3, 3, 3, nspins);
			Rho1Hessaa.setZero();
			const EigenTensor<3> Daa = SliceTensor(D, {ihead, ihead, 0}, {ilength, ilength, nspins});
			const EigenTensor<3> AO1a = SliceTensor(AO1, {0, ihead, 0}, {ngrids, ilength, 3});
			const EigenTensor<3> AO2a = SliceTensor(AO2, {0, ihead, 0}, {ngrids, ilength, 6});
			#include "DensityEinSum/Daa_mu,nu,w...AO2a_g,mu,r+t...AO1a_g,nu,s---Rho1Hessaa_g,r,t,s,w.hpp" // 2, 4
			Rho1Hess.chip(iatom, 5).chip(iatom, 3) = Rho1Hessaa + Rho1Hessaa.shuffle(Eigen::array<int, 5>{0, 1, 3, 2, 4});
			Rho1Hessaa.setZero();
			const EigenTensor<3> Da = SliceTensor(D, {ihead, 0, 0}, {ilength, nbasis, nspins});
			const EigenTensor<3> AO3a = SliceTensor(AO3, {0, ihead, 0}, {ngrids, ilength, 10});
			#include "DensityEinSum/Da_mu,nu,w...AO3a_g,mu,r+t+s...AO_g,nu---Rho1Hessaa_g,r,t,s,w.hpp" // 1
			#include "DensityEinSum/Da_mu,nu,w...AO2a_g,mu,t+s...AO1_g,nu,r---Rho1Hessaa_g,r,t,s,w.hpp" // 3
			Rho1Hess.chip(iatom, 5).chip(iatom, 3) += Rho1Hessaa;
			Rho1Hessaa.resize(0, 0, 0, 0, 0);
			for ( int jatom = 0; jatom < iatom; jatom++ ){
				const int jhead = this->AtomHeads[jatom];
				const int jlength = this->AtomLengths[jatom];
				EigenTensor<5> Rho1Hessab(ngrids, 3, 3, 3, nspins);
				Rho1Hessab.setZero();
				const EigenTensor<3> Dab = SliceTensor(D, {ihead, jhead, 0}, {ilength, jlength, nspins});
				const EigenTensor<3> AO1b = SliceTensor(AO1, {0, jhead, 0}, {ngrids, jlength, 3});
				#include "DensityEinSum/Dab_mu,nu,w...AO2a_g,mu,r+t...AO1b_g,nu,s---Rho1Hessab_g,r,t,s,w.hpp" // 2
				const EigenTensor<3> AO2b = SliceTensor(AO2, {0, jhead, 0}, {ngrids, jlength, 6});
				#include "DensityEinSum/Dab_mu,nu,w...AO2b_g,nu,r+s...AO1a_g,mu,t---Rho1Hessab_g,r,t,s,w.hpp" // 4
				Rho1Hess.chip(jatom, 5).chip(iatom, 3) = Rho1Hessab;
				Rho1Hess.chip(iatom, 5).chip(jatom, 3) = Rho1Hessab.shuffle(Eigen::array<int, 5>{0, 1, 3, 2, 4});
			}
		}
		ScaleTensor(Rho1Hess, 2);
		SigmaHess.resize(ngrids, 3, natoms, 3, natoms, nspins * ( 1 + nspins ) / 2); SigmaHess.setZero();
		#include "DensityEinSum/Rho1Hess_g,r,t,a,s,b,u...Rho1_g,r,v---SigmaHess_g,t,a,s,b,u+v.hpp"
		#include "DensityEinSum/Rho1Grad_g,r,t,a,u...Rho1Grad_g,r,s,b,v---SigmaHess_g,t,a,s,b,u+v.hpp"
		ScaleTensor(SigmaHess, 2); // Incorrect for U/RO. To be corrected in the future.
	}
}
