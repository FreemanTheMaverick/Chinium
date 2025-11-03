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

void SubGrid::getDensity(EigenMatrix& D_){
	const EigenMatrix Dblock = D_(this->BasisList, this->BasisList);
	const Eigen::TensorMap<const EigenTensor<2>> D(Dblock.data(), this->getNumBasis(), this->getNumBasis());
	const int ngrids = this->NumGrids;
	if ( this->Type >= 0 ){
		Rho.resize(ngrids); Rho.setZero();
		#include "DensityEinSum/D_mu,nu...AO_g,mu...AO_g,nu---Rho_g.hpp"
	}
	if ( this->Type >= 1 ){
		Rho1.resize(ngrids, 3); Rho1.setZero();
		#include "DensityEinSum/D_mu,nu...AO1_g,mu,r...AO_g,nu---Rho1_g,r.hpp"
		ScaleTensor(Rho1, 2);
		Sigma.resize(ngrids); Sigma.setZero();
		#include "DensityEinSum/Rho1_g,r...Rho1_g,r---Sigma_g.hpp"
	}
	if ( this->Type >= 2 ){
		Tau.resize(ngrids); Tau.setZero();
		#include "DensityEinSum/D_mu,nu...AO1_g,mu,r...AO1_g,nu,r---Tau_g.hpp"
		ScaleTensor(Tau, 0.5);
		Lapl = 4. * Tau;
		#include "DensityEinSum/D_mu,nu...AO_g,mu...AO2L_g,nu---Lapl_g.hpp"
		#include "DensityEinSum/D_mu,nu...AO2L_g,mu...AO_g,nu---Lapl_g.hpp"
	}
}

void SubGrid::getNumElectrons(double& n){
	assert( this->NumGrids == W.dimension(0) );
	assert( this->NumGrids == Rho.dimension(0) );
	const EigenTensor<0> n_ = (W * Rho).sum();
	n += n_();
}

void SubGrid::getDensityU(std::vector<EigenMatrix>& D_s){
	const int ngrids = this->NumGrids;
	const int nmats = (int)D_s.size();
	const int nbasis = this->getNumBasis();
	EigenTensor<3> D(nbasis, nbasis, nmats);
	for ( int imat = 0; imat < nmats; imat++ ){
		for ( int jbasis = 0; jbasis < nbasis; jbasis++ ){
			for ( int kbasis = jbasis; kbasis < nbasis; kbasis++ ){
				D(jbasis, kbasis, imat) = D(kbasis, jbasis, imat) = D_s[imat](this->BasisList[jbasis], this->BasisList[kbasis]);
			}
		}
	}
	if ( this->Type >= 0 ){
		RhoU.resize(ngrids, nmats); RhoU.setZero();
		#include "DensityEinSum/D_mu,nu,mat...AO_g,mu...AO_g,nu---RhoU_g,mat.hpp"
	}
	if ( this->Type >= 1 ){
		Rho1U.resize(ngrids, 3, nmats); Rho1U.setZero();
		#include "DensityEinSum/D_mu,nu,mat...AO1_g,mu,r...AO_g,nu---Rho1U_g,r,mat.hpp"
		ScaleTensor(Rho1U, 2);
		SigmaU.resize(ngrids, nmats); SigmaU.setZero();
		#include "DensityEinSum/Rho1U_g,r,mat...Rho1_g,r---SigmaU_g,mat.hpp"
		ScaleTensor(SigmaU, 2);
	}
}

void SubGrid::getDensitySkeleton(EigenMatrix& D_){
	const int nbasis = this->getNumBasis();
	const int ngrids = this->NumGrids;
	const int natoms = this->getNumAtoms();
	const EigenMatrix Dblock = D_(this->BasisList, this->BasisList);
	const Eigen::TensorMap<const EigenTensor<2>> D(Dblock.data(), nbasis, nbasis);
	if ( this->Type >= 0 ){
		RhoGrad.resize(ngrids, 3, natoms); RhoGrad.setZero();
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int head = this->AtomHeads[iatom];
			const int length = this->AtomLengths[iatom];
			const EigenTensor<2> Da = SliceTensor(D, {head, 0}, {length, nbasis});
			const EigenTensor<3> AO1a = SliceTensor(AO1, {0, head, 0}, {ngrids, length, 3});
			EigenTensor<2> RhoGrada(ngrids, 3);
			RhoGrada.setZero();
			#include "DensityEinSum/Da_mu,nu...AO1a_g,mu,t...AO_g,nu---RhoGrada_g,t.hpp"
			RhoGrad.chip(iatom, 2) = RhoGrada;
		}
		ScaleTensor(RhoGrad, -2);
	}
	if ( this->Type >= 1 ){
		Rho1Grad.resize(ngrids, 3, 3, natoms); Rho1Grad.setZero();
		SigmaGrad.resize(ngrids, 3, natoms); SigmaGrad.setZero();
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int head = this->AtomHeads[iatom];
			const int length = this->AtomLengths[iatom];
			const EigenTensor<2> Da = SliceTensor(D, {head, 0}, {length, nbasis});
			const EigenTensor<3> AO2a = SliceTensor(AO2, {0, head, 0}, {ngrids, length, 6});
			EigenTensor<3> Rho1Grada(ngrids, 3, 3);
			Rho1Grada.setZero();
			#include "DensityEinSum/Da_mu,nu...AO2a_g,mu,r,t...AO_g,nu---Rho1Grada_g,r,t.hpp"
			const EigenTensor<3> AO1a = SliceTensor(AO1, {0, head, 0}, {ngrids, length, 3});
			#include "DensityEinSum/Da_mu,nu...AO1a_g,mu,t...AO1_g,nu,r---Rho1Grada_g,r,t.hpp"
			ScaleTensor(Rho1Grada, -2);
			Rho1Grad.chip(iatom, 3) = Rho1Grada;
			EigenTensor<2> SigmaGrada(ngrids, 3);
			SigmaGrada.setZero();
			#include "DensityEinSum/Rho1_g,r...Rho1Grada_g,r,t---SigmaGrada_g,t.hpp"
			SigmaGrad.chip(iatom, 2) = 2 * SigmaGrada;
		}
	}
}

void SubGrid::getDensitySkeleton2(EigenMatrix& D_){
	const int nbasis = this->getNumBasis();
	const int ngrids = this->NumGrids;
	const int natoms = this->getNumAtoms();
	const EigenMatrix Dblock = D_(this->BasisList, this->BasisList);
	const Eigen::TensorMap<const EigenTensor<2>> D(Dblock.data(), nbasis, nbasis);
	if ( this->Type >= 0 ){
		RhoHess.resize(ngrids, 3, natoms, 3, natoms); RhoHess.setZero();
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = this->AtomHeads[iatom];
			const int ilength = this->AtomLengths[iatom];
			const EigenTensor<2> Da = SliceTensor(D, {ihead, 0}, {ilength, nbasis});
			const EigenTensor<3> AO2a = SliceTensor(AO2, {0, ihead, 0}, {ngrids, ilength, 6});
			EigenTensor<3> RhoHessaa(ngrids, 3, 3);
			RhoHessaa.setZero();
			#include "DensityEinSum/Da_mu,nu...AO2a_g,mu,t,s...AO_g,nu---RhoHessaa_g,t,s.hpp"
			const EigenTensor<2> Daa = SliceTensor(D, {ihead, ihead}, {ilength, ilength});
			const EigenTensor<3> AO1a = SliceTensor(AO1, {0, ihead, 0}, {ngrids, ilength, 3});
			#include "DensityEinSum/Daa_mu,nu...AO1a_g,mu,t...AO1a_g,nu,s---RhoHessaa_g,t,s.hpp"
			RhoHess.chip(iatom, 4).chip(iatom, 2) = RhoHessaa;
			RhoHessaa.resize(0, 0, 0);
			for ( int jatom = 0; jatom < iatom; jatom++ ){
				const int jhead = this->AtomHeads[jatom];
				const int jlength = this->AtomLengths[jatom];
				const EigenTensor<2> Dab = SliceTensor(D, {ihead, jhead}, {ilength, jlength});
				const EigenTensor<3> AO1b = SliceTensor(AO1, {0, jhead, 0}, {ngrids, jlength, 3});
				EigenTensor<3> RhoHessab(ngrids, 3, 3);
				RhoHessab.setZero();
				#include "DensityEinSum/Dab_mu,nu...AO1a_g,mu,t...AO1b_g,nu,s---RhoHessab_g,t,s.hpp"
				RhoHess.chip(jatom, 4).chip(iatom, 2) = RhoHessab;
				RhoHess.chip(iatom, 4).chip(jatom, 2) = RhoHessab.shuffle(Eigen::array<int, 3>{0, 2, 1});
			}
		}
		ScaleTensor(RhoHess, 2);
	}
	if ( this->Type >= 1 ){
		Rho1Hess.resize(ngrids, 3, 3, natoms, 3, natoms); Rho1Hess.setZero();
		for ( int iatom = 0; iatom < natoms; iatom++ ){
			const int ihead = this->AtomHeads[iatom];
			const int ilength = this->AtomLengths[iatom];
			EigenTensor<4> Rho1Hessaa(ngrids, 3, 3, 3);
			Rho1Hessaa.setZero();
			const EigenTensor<2> Daa = SliceTensor(D, {ihead, ihead}, {ilength, ilength});
			const EigenTensor<3> AO1a = SliceTensor(AO1, {0, ihead, 0}, {ngrids, ilength, 3});
			const EigenTensor<3> AO2a = SliceTensor(AO2, {0, ihead, 0}, {ngrids, ilength, 6});
			#include "DensityEinSum/Daa_mu,nu...AO2a_g,mu,r,t...AO1a_g,nu,s---Rho1Hessaa_g,r,t,s.hpp" // 2, 4
			Rho1Hess.chip(iatom, 5).chip(iatom, 3) = Rho1Hessaa + Rho1Hessaa.shuffle(Eigen::array<int, 4>{0, 1, 3, 2});
			Rho1Hessaa.setZero();
			const EigenTensor<2> Da = SliceTensor(D, {ihead, 0}, {ilength, nbasis});
			const EigenTensor<3> AO3a = SliceTensor(AO3, {0, ihead, 0}, {ngrids, ilength, 10});
			#include "DensityEinSum/Da_mu,nu...AO3a_g,mu,r,t,s...AO_g,nu---Rho1Hessaa_g,r,t,s.hpp" // 1
			#include "DensityEinSum/Da_mu,nu...AO2a_g,mu,t,s...AO1_g,nu,r---Rho1Hessaa_g,r,t,s.hpp" // 3
			Rho1Hess.chip(iatom, 5).chip(iatom, 3) += Rho1Hessaa;
			Rho1Hessaa.resize(0, 0, 0, 0);
			for ( int jatom = 0; jatom < iatom; jatom++ ){
				const int jhead = this->AtomHeads[jatom];
				const int jlength = this->AtomLengths[jatom];
				EigenTensor<4> Rho1Hessab(ngrids, 3, 3, 3);
				Rho1Hessab.setZero();
				const EigenTensor<2> Dab = SliceTensor(D, {ihead, jhead}, {ilength, jlength});
				const EigenTensor<3> AO1b = SliceTensor(AO1, {0, jhead, 0}, {ngrids, jlength, 3});
				#include "DensityEinSum/Dab_mu,nu...AO2a_g,mu,r,t...AO1b_g,nu,s---Rho1Hessab_g,r,t,s.hpp" // 2
				const EigenTensor<3> AO2b = SliceTensor(AO2, {0, jhead, 0}, {ngrids, jlength, 6});
				#include "DensityEinSum/Dab_mu,nu...AO1a_g,mu,t...AO2b_g,nu,r,s---Rho1Hessab_g,r,t,s.hpp" // 4
				Rho1Hess.chip(jatom, 5).chip(iatom, 3) = Rho1Hessab;
				Rho1Hess.chip(iatom, 5).chip(jatom, 3) = Rho1Hessab.shuffle(Eigen::array<int, 4>{0, 1, 3, 2});
			}
		}
		ScaleTensor(Rho1Hess, 2);
		SigmaHess.resize(ngrids, 3, natoms, 3, natoms); SigmaHess.setZero();
		#include "DensityEinSum/Rho1Hess_g,r,t,a,s,b...Rho1_g,r---SigmaHess_g,t,a,s,b.hpp"
		#include "DensityEinSum/Rho1Grad_g,r,t,a...Rho1Grad_g,r,s,b---SigmaHess_g,t,a,s,b.hpp"
		ScaleTensor(SigmaHess, 2);
	}
}
