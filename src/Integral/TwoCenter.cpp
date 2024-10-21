#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cmath> // std::abs in "../Macro.h"
#include <experimental/array> // std::array, std::experimental::make_array
#include <utility> // std::pair, std::make_pair

#include "../Macro.h"
#include "../Multiwfn.h" // Requires <Eigen/Dense>, <vector>, <string>, "Macro.h".
#include "Macro.h"

#include <iostream>

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

void Multiwfn::getTwoCenter(int order, const bool output){
	__Make_Basis_Set__
	std::vector<std::pair<double, std::array<double, 3>>> libint2charges = {}; // Making point charges.
	for ( MwfnCenter& center : this->Centers )
		libint2charges.push_back(std::make_pair(
			center.Nuclear_charge,
			std::experimental::make_array(
				center.Coordinates[0],
				center.Coordinates[1],
				center.Coordinates[2])));

	if ( order >= 0 ){
		if (output) std::printf("Calculating 2c-1e integrals ... ");
		const auto start = __now__;
		std::vector<EigenMatrix> matrices = getTwoCenter0( obs, libint2charges, libint2::Operator::emultipole2 );
		this->Overlap = matrices[0];
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
		if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
	}
}


