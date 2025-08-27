#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cmath> // std::abs in "../Macro.h"
#include <experimental/array> // std::array, std::experimental::make_array
#include <utility> // std::pair, std::make_pair
#include <chrono>
#include <libmwfn.h>

#include "../Macro.h"
#include "Int2C1E.h"
#include "Macro.h"

std::vector<EigenMatrix> getTwoCenter0(
		libint2::BasisSet& obs,
		std::vector<std::pair<double, std::array<double, 3>>>& libint2charges,
		libint2::Operator Operator){
	const int nbasis = libint2::nbf(obs);
	std::vector<EigenMatrix> matrices = {};
	libint2::initialize();
	libint2::Engine engine(Operator, obs.max_nprim(), obs.max_l());
	if ( Operator == libint2::Operator::nuclear )
		engine.set_params(libint2charges); // Including nuclear information.
	if ( Operator == libint2::Operator::overlap || Operator == libint2::Operator::kinetic || Operator == libint2::Operator::nuclear )
		matrices.resize(1);
	else if ( Operator == libint2::Operator::emultipole1 )
		matrices.resize(4);
	else if ( Operator == libint2::Operator::emultipole2 )
		matrices.resize(10);
	else if ( Operator == libint2::Operator::emultipole3 )
		matrices.resize(20);
	for ( EigenMatrix& matrix : matrices )
		matrix = EigenZero(nbasis, nbasis);

	const auto& buf_vec = engine.results(); // Preallocating memory for integrals.
	const auto shell2bf = obs.shell2bf(); // Feeding the index of a shell to 'shell2bf' returns the index of the first basis function of that shell among all basis functions.
	for ( short int s1 = 0; s1 < (short int)obs.size(); s1++ ){ // Looping over all shells.
		const short int bf1_first = shell2bf[s1];
		const short int n1 = obs[s1].size(); // Number of basis functions in a shell.
		for ( short int s2 = 0; s2 <= s1; s2++ ){ // Looping over all shells, avoiding replication.
			engine.compute(obs[s1], obs[s2]); // Calculating integrals of basis functions in those two shells.
			//const auto ints_shellset = buf_vec[0]; // Integrals of this shell pair are stored in memory location buf_vec.
			//if (ints_shellset == nullptr) continue; // If there are no integrals at all, continue. Will this ever happen?
			const short int bf2_first = shell2bf[s2];
			const short int n2 = obs[s2].size();
			for ( short int f1 = 0; f1 < n1; f1++ ){ // Looping over all basis functions in shell 1.
				const short int bf1 = bf1_first + f1; // Index of the basis function among all basis functions.
				for ( short int f2 = 0; f2 < n2; f2++ ){ // Looping over all basis functions in shell 2.
					const short int bf2 = bf2_first + f2;
					if ( bf2 <= bf1 ){ // Considering only unique pairs of basis functions.
						for ( short int imat = 0; imat < (short int)matrices.size(); imat++ ){
							if (buf_vec[imat] == nullptr) continue; // If there are no integrals at all, continue. Will this ever happen?
							matrices[imat](bf1, bf2) = buf_vec[imat][f1 * n2 + f2];
							matrices[imat](bf2, bf1) = buf_vec[imat][f1 * n2 + f2]; // One-electron integral matrix is symmetric.
						}
					}
				}
			}
		}
	}
	libint2::finalize();
	return matrices;
}

#define __Loop_Over_XYZ__(iatom, position){\
	for ( short int f1 = 0, f12 = 0; f1 != n1; f1++ ){\
		const short int bf1 = bf1_first + f1;\
		for ( short int f2 = 0; f2 != n2; f2++, f12++ ){\
			const short int bf2 = bf2_first + f2;\
			if (bf2 <= bf1){\
				gs[iatom][0](bf1,bf2)+=buf_vec[3*(position)+0][f12];\
				gs[iatom][1](bf1,bf2)+=buf_vec[3*(position)+1][f12];\
				gs[iatom][2](bf1,bf2)+=buf_vec[3*(position)+2][f12];\
			}\
		}\
	}\
}

std::vector<std::vector<EigenMatrix>> getTwoCenter1(
		libint2::BasisSet& obs,
		std::vector<std::pair<double, std::array<double, 3>>>& libint2charges,
		std::vector<int>& shell2atom,
		libint2::Operator Operator){
	const int nbasis = libint2::nbf(obs);
	const int natoms = *std::max_element(shell2atom.begin(), shell2atom.end()) + 1;
	std::vector<std::vector<EigenMatrix>> gs(natoms);
	for ( std::vector<EigenMatrix>& g : gs )
		g = {EigenZero(nbasis,nbasis), EigenZero(nbasis,nbasis), EigenZero(nbasis,nbasis)};
	libint2::initialize();
	libint2::Engine engine(Operator, obs.max_nprim(), obs.max_l(), 1);
	if ( Operator == libint2::Operator::nuclear )
		engine.set_params(libint2charges); // Including nuclear information.
	const auto& buf_vec = engine.results();
	const auto shell2bf = obs.shell2bf();
	for ( short int s1 = 0; s1 != (short int)obs.size(); s1++ ){
		const short int atom1 = shell2atom[s1];
		const short int bf1_first = shell2bf[s1];
		const short int n1 = obs[s1].size();
		for ( short int s2 = 0; s2 <= s1; s2++ ){
			const short int atom2 = shell2atom[s2];
			if ( Operator != libint2::Operator::nuclear && atom1 == atom2 ) continue;
			const short int bf2_first = shell2bf[s2];
			const short int n2 = obs[s2].size();
			engine.compute(obs[s1], obs[s2]);
			__Loop_Over_XYZ__(atom1, 0)
			__Loop_Over_XYZ__(atom2, 1)
			if ( Operator == libint2::Operator::nuclear )
				for (short int iatom = 0; iatom < natoms; iatom++ )
					__Loop_Over_XYZ__(iatom, iatom + 2)
		}
	}
	for ( int iatom = 0; iatom < natoms; iatom++ ) for ( int xyz = 0; xyz < 3; xyz++ ){
		const EigenMatrix transpose = gs[iatom][xyz].transpose();
		const EigenMatrix diagonal(gs[iatom][xyz].diagonal().asDiagonal());
		gs[iatom][xyz] += transpose - diagonal;
	}
	libint2::finalize();
	return gs;
}

EigenMatrix getTwoCenter2(
		libint2::BasisSet& obs,
		std::vector<std::pair<double, std::array<double, 3>>>& libint2charges,
		std::vector<int>& shell2atom,
		libint2::Operator Operator, EigenMatrix D){
	const int natoms = *std::max_element(shell2atom.begin(), shell2atom.end()) + 1;
	EigenMatrix hessian = EigenMatrix(3 * natoms, 3 * natoms);
	int atomlist[] = {114, 514};
	libint2::initialize();
	libint2::Engine engine(Operator, obs.max_nprim(), obs.max_l(), 2);
	if ( Operator == libint2::Operator::nuclear )
		engine.set_params(libint2charges); // Including nuclear information.
	const auto& buf_vec = engine.results();
	const auto shell2bf = obs.shell2bf();
	for ( short int s1 = 0; s1 != (short int)obs.size(); s1++ ){
		atomlist[0] = shell2atom[s1];
		const short int bf1_first = shell2bf[s1];
		const short int n1 = obs[s1].size();
		for ( short int s2 = 0; s2 <= s1; s2++ ){
			atomlist[1] = shell2atom[s2];
			if ( Operator != libint2::Operator::nuclear && atomlist[0] == atomlist[1] ) continue;
			const short int bf2_first = shell2bf[s2];
			const short int n2 = obs[s2].size();
			engine.compute(obs[s1], obs[s2]);
			if ( !buf_vec[0] ) continue;
			for ( short int f1 = 0, f12 = 0; f1 < n1; f1++ ){
				const short int bf1 = bf1_first + f1;
				for ( short int f2 = 0; f2 < n2; f2++, f12++ ){
					const short int bf2 = bf2_first + f2;
					if ( bf2 <= bf1 ){
						const double tmp = ((bf1==bf2)?1:2) * D(bf1, bf2);
						int xpert = 114514;
						int ypert = 1919810;
						if ( Operator == libint2::Operator::nuclear ) for ( int p = 0, ptqs = 0; p < 2 + natoms; p++ ) for ( int t = 0; t < 3; t++ ){
							xpert = 3 * ( p < 2 ? atomlist[p] : p - 2 ) + t;
							for ( int q = p; q < 2 + natoms; q++ ) for ( int s = ((q==p)?t:0); s < 3; s++, ptqs++ ){
								ypert = 3 * ( q < 2 ? atomlist[q] : q - 2 ) + s;
								hessian(xpert, ypert) += ( (xpert==ypert && p!=q && (p<2 || q<2)) ? 2 : 1 ) * tmp * buf_vec[ptqs][f12];
							}
						}
						else for ( int p = 0, ptqs = 0; p < 2; p++ ) for ( int t = 0; t < 3; t++ ){
							xpert = 3 * atomlist[p] + t;
							for ( int q = p; q < 2; q++ ) for ( int s = ((q==p)?t:0); s < 3; s++, ptqs++ ){
								ypert = 3 * atomlist[q] + s;
								hessian(xpert, ypert) += tmp * buf_vec[ptqs][f12];
							}
						}
					}
				}
			}
		}
	}
	return hessian + hessian.transpose() - (EigenMatrix)hessian.diagonal().asDiagonal();
}

Int2C1E::Int2C1E(Mwfn& mwfn){
	this->MWFN = &mwfn;
	const int natoms = mwfn.getNumCenters();
	this->OverlapGrads.resize(3 * natoms);
	this->KineticGrads.resize(3 * natoms);
	this->NuclearGrads.resize(3 * natoms);
	this->OverlapHesss.resize(3 * natoms);
	this->KineticHesss.resize(3 * natoms);
	this->NuclearHesss.resize(3 * natoms);
	for ( int i = 0; i < 3 * natoms; i++ ){
		this->OverlapHesss[i].resize(3 * natoms);
		this->KineticHesss[i].resize(3 * natoms);
		this->NuclearHesss[i].resize(3 * natoms);
	}
}

#define __Make_Point_Charges__\
	std::vector<std::pair<double, std::array<double, 3>>> libint2charges = {};\
	for ( MwfnCenter& center : this->MWFN->Centers )\
		libint2charges.push_back(std::make_pair(\
			center.Nuclear_charge,\
			std::experimental::make_array(\
				center.Coordinates[0],\
				center.Coordinates[1],\
				center.Coordinates[2])));

void Int2C1E::CalculateIntegrals(int order, int output){
	__Make_Basis_Set__(this->MWFN)
	__Make_Point_Charges__
	const int natoms = this->MWFN->getNumCenters();
	if ( order == 0 ){
		if (output>0) std::printf("Calculating 2c-1e integrals ... ");
		const auto start = __now__;
		std::vector<EigenMatrix> matrices = getTwoCenter0( obs, libint2charges, libint2::Operator::emultipole2 );
		this->Overlap = this->MWFN->Overlap = matrices[0];
		this->DipoleX = matrices[1];
		this->DipoleY = matrices[2];
		this->DipoleZ = matrices[3];
		this->QuadrapoleXX = matrices[4];
		this->QuadrapoleXY = matrices[5];
		this->QuadrapoleXZ = matrices[6];
		this->QuadrapoleYY = matrices[7];
		this->QuadrapoleYZ = matrices[8];
		this->QuadrapoleZZ = matrices[9];
		this->Kinetic = getTwoCenter0( obs, libint2charges, libint2::Operator::kinetic )[0];
		this->Nuclear = getTwoCenter0( obs, libint2charges, libint2::Operator::nuclear )[0];
		if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
	}else if ( order == 1 ){
		if (output>0) std::printf("Calculating 2c-1e integral nuclear gradient ... ");
		const auto start = __now__;
		const std::vector<std::vector<EigenMatrix>> overlapgrads = getTwoCenter1(obs, libint2charges, shell2atom, libint2::Operator::overlap);
		const std::vector<std::vector<EigenMatrix>> kineticgrads = getTwoCenter1(obs, libint2charges, shell2atom, libint2::Operator::kinetic);
		const std::vector<std::vector<EigenMatrix>> nucleargrads = getTwoCenter1(obs, libint2charges, shell2atom, libint2::Operator::nuclear);
		for ( int iatom = 0, tot = 0; iatom < natoms; iatom++ ) for ( int xyz = 0; xyz < 3; xyz++, tot++ ){
			this->OverlapGrads[tot] = overlapgrads[iatom][xyz];
			this->KineticGrads[tot] = kineticgrads[iatom][xyz];
			this->NuclearGrads[tot] = nucleargrads[iatom][xyz];
		}
		if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
	}
}

std::tuple<
	std::vector<double>,
	std::vector<double>,
	std::vector<double>
> Int2C1E::ContractGrads(EigenMatrix D, EigenMatrix W, int output){
	std::vector<EigenMatrix> Ds = {D};
	std::vector<EigenMatrix> Ws = {W};
	auto [SW, KD, VD] = this->ContractGrads(Ds, Ws, output);
	return std::make_tuple(SW[0], KD[0], VD[0]);
}

std::tuple<
	std::vector<std::vector<double>>,
	std::vector<std::vector<double>>,
	std::vector<std::vector<double>>
> Int2C1E::ContractGrads(std::vector<EigenMatrix>& Ds, std::vector<EigenMatrix>& Ws, int output){
	const int natoms = this->MWFN->getNumCenters();
	const int nDs = (int)Ds.size();
	const int nWs = (int)Ws.size();
	if ( this->OverlapGrads[0].size() == 0 || this->KineticGrads[0].size() == 0 || this->NuclearGrads[0].size() == 0 )
		this->CalculateIntegrals(1, output);
	if (output>0) std::printf("Contracting 2c-1e integral gradients with %d Ds and %d Ws ... ", nDs, nWs);
	const auto start = __now__;
	std::vector<std::vector<double>> SW(nWs, std::vector<double>(3 * natoms));
	std::vector<std::vector<double>> KD(nDs, std::vector<double>(3 * natoms));
	std::vector<std::vector<double>> VD(nDs, std::vector<double>(3 * natoms));
	for ( int i = 0; i < nWs; i++ ) for ( int j = 0; j < 3 * natoms; j++ ){
		SW[i][j] = Ws[i].cwiseProduct(this->OverlapGrads[j]).sum();
	}
	for ( int i = 0; i < nDs; i++ ) for ( int j = 0; j < 3 * natoms; j++ ){
		KD[i][j] = Ds[i].cwiseProduct(this->KineticGrads[j]).sum();
		VD[i][j] = Ds[i].cwiseProduct(this->NuclearGrads[j]).sum();
	}
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
	return std::make_tuple(SW, KD, VD);
}

std::tuple<
	std::vector<std::vector<double>>,
	std::vector<std::vector<double>>,
	std::vector<std::vector<double>>
> Int2C1E::ContractHesss(EigenMatrix D, EigenMatrix W, int output){
	if (output>0) std::printf("Contracting 2c-1e integral nuclear hessians with D and W ... ");
	const auto start = __now__;
	__Make_Basis_Set__(this->MWFN)
	__Make_Point_Charges__
	const int natoms = this->MWFN->getNumCenters();
	EigenMatrix SW_ = getTwoCenter2(obs, libint2charges, shell2atom, libint2::Operator::overlap, W);
	EigenMatrix KD_ = getTwoCenter2(obs, libint2charges, shell2atom, libint2::Operator::kinetic, D);
	EigenMatrix VD_ = getTwoCenter2(obs, libint2charges, shell2atom, libint2::Operator::nuclear, D);
	std::vector<std::vector<double>> SW(3 * natoms, std::vector<double>(3 * natoms));
	std::vector<std::vector<double>> KD(3 * natoms, std::vector<double>(3 * natoms));
	std::vector<std::vector<double>> VD(3 * natoms, std::vector<double>(3 * natoms));
	for ( int i = 0; i < 3 * natoms; i++ )for ( int j = 0; j < 3 * natoms; j++ ){
			SW[i][j] = SW_(i, j);
			KD[i][j] = KD_(i, j);
			VD[i][j] = VD_(i, j);
	}
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
	return std::make_tuple(SW, KD, VD);
}
