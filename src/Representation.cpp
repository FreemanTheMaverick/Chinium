#include <Eigen/Core>
#include <vector>
#include <tuple>
#include <string>
#include <typeinfo>
#include <stdexcept>
#include <libmwfn.h>
#include <iostream>

#include "Macro.h"
#include "Gateway.h"
#include "Integral.h"
#include "Representation.h"

Representation::Representation(std::string inp){
	// File names
	const size_t suffix_pos = inp.find_last_of('.');
	const std::string mwfnname = inp.substr(0, suffix_pos) + ".mwfn";

	// Mwfn: atomic coordinates, basis and pseudopotentials
	const std::vector<std::vector<double>> atoms = ReadXYZ(inp);
	const auto [basis, pseudo] = ReadBasisSet(inp);
	const std::string guess = ReadGuess(inp);
	if ( atoms.empty() || basis.empty() || guess == "READ" ){
		std::printf("Reading existent Mwfn file %s ...\n", mwfnname.c_str());
		mwfn = Mwfn(mwfnname);
		const int wfntype = ReadWfnType(inp);
		if ( wfntype != mwfn.Wfntype ) throw std::runtime_error("Inconsistent wavefunction type between the mwfn file and the input file!");
	}
	const std::string basis_file_path_name = (std::string)std::getenv("CHINIUM_PATH") + "/BasisSets/" + basis + ".gbs";
	const std::string pseudo_file_path_name = pseudo.empty() ? "" : (std::string)std::getenv("CHINIUM_PATH") + "/BasisSets/" + pseudo + ".ecp";
	if ( !atoms.empty() ){
		if ( !basis.empty() ){
			std::printf("Setting atomic coordinates given by input file ...\n");
			mwfn.setCenters(atoms);
			std::printf("Reading basis set file %s and %s ...\n", basis_file_path_name.c_str(), pseudo_file_path_name.c_str());
			mwfn.setBasis(basis_file_path_name, pseudo_file_path_name);
		}else{
			assert( mwfn.getNumCenters() == (int)atoms.size() && "Different numbers of atoms between the input file and the read mwfn file!" );
			for ( int iatom = 0; iatom < mwfn.getNumCenters(); iatom++ ){
				mwfn.Centers[iatom].Index = (int)atoms[iatom][0];
				mwfn.Centers[iatom].Nuclear_charge = atoms[iatom][1];
				mwfn.Centers[iatom].Coordinates[0] = atoms[iatom][2];
				mwfn.Centers[iatom].Coordinates[1] = atoms[iatom][3];
				mwfn.Centers[iatom].Coordinates[2] = atoms[iatom][4];
			}
		}
	}else{ // If atoms are not specified in the input file
		if ( !basis.empty() ){
			mwfn.setBasis(basis_file_path_name, pseudo_file_path_name);
		} // Else do nothing (just keep the existent atoms and basis in the mwfn)
	}
	Normalize(&mwfn);
	mwfn.PrintCenters();

	// Numbers of electrons
	int total_nuclear_charges = 0;
	for ( MwfnCenter& center : mwfn.Centers ) total_nuclear_charges += center.Nuclear_charge;
	std::tie(Na, Nb, Np) = ReadNumElectrons(inp, total_nuclear_charges);

	// One-electron integrals
	int2c1e = Int2C1E(mwfn);
	int2c1e.CalculateIntegrals(0, 1);

	// Orthogonalize read orbitals
	mwfn.Overlap = int2c1e.Overlap;
	if ( guess == "READ" ) mwfn.Orthogonalize("Lowdin");
}

// Constructors of all wavefunction types
// Set occupation numbers

#define Round(x) (int)( isInt(x) ? std::lround(x) : std::floor(x) )
RepR::RepR(std::string inp): Representation(inp){
	mwfn.Wfntype = 0;
	if ( typeid(*this) == typeid(RepR) )
		for ( int occ : mwfn.getOccupation(1) )
			if ( occ != 0 || occ != 2 ) throw std::runtime_error("Bad occupation number!");
	if ( ReadGuess(inp) != "READ" ){
		mwfn.Orbitals.resize(mwfn.getNumBasis());
		EigenVector occ = EigenZero(mwfn.getNumBasis(), 1);
		for ( int i = 0; i < Round(Na); i++ ) occ(i) = 1;
		occ( Round(Na) ) = Na - Round(Na);
		mwfn.setOccupation(occ, 1);
	}
}

RepU::RepU(std::string inp): Representation(inp){
	mwfn.Wfntype = 1;
	if ( typeid(*this) == typeid(RepR) )
		for ( int occ : mwfn.getOccupation(1) )
			if ( occ != 0 || occ != 1 ) throw std::runtime_error("Bad occupation number!");
	if ( ReadGuess(inp) != "READ" ){
		mwfn.Orbitals.resize(mwfn.getNumBasis() * 2);
		EigenVector occ = EigenZero(mwfn.getNumBasis(), 1);
		for ( int i = 0; i < Round(Na); i++ ) occ(i) = 1;
		mwfn.setOccupation(occ, 1);
		occ.setZero();
		for ( int i = 0; i < Nb; i++ ) occ(i) = 1;
		mwfn.setOccupation(occ, 2);
	}
}

RepRO::RepRO(std::string inp): Representation(inp){
	mwfn.Wfntype = 2;
	if ( Np == 0 ) Np = Nb;
	if ( ReadGuess(inp) != "READ" ){
		mwfn.Orbitals.resize(mwfn.getNumBasis());
		const int np_int = std::lround(Np);
		const int na_int = std::lround(Na);
		const int nb_int = std::lround(Nb);
		for ( int i = 0; i < na_int + nb_int - np_int; i++ ){
			auto& orbital_i = mwfn.Orbitals[i];
			if ( i < np_int ){
				orbital_i.Occ = 2;
				orbital_i.Type = 0;
			}else{
				orbital_i.Occ = 1;
				if ( i < na_int ) orbital_i.Type = 1;
				else orbital_i.Type = 2;
			}
		}
	}
}
