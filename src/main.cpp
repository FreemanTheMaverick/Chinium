#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <cstddef>
#include <libmwfn.h>

#include "Macro.h"
#include "Gateway.h"
#include "Integral/Normalization.h"
#include "Integral/Int2C1E.h"
#include "Integral/Int4C2E.h"
#include "Grid/Grid.h"
#include "ExchangeCorrelation.h"
#include "HartreeFockKohnSham/HartreeFockKohnSham.h"
#include "Localization/Localize.h"

int main(int /*argc*/, char* argv[]){
	std::printf("*** Chinium started ***\n");

	// File names
	std::string inp = argv[1];
	std::string job_name = inp;
	size_t suffix_pos = inp.find_last_of('.');
	if ( suffix_pos != std::string::npos ) job_name = job_name.substr(0, suffix_pos);
	std::string mwfn_name = job_name + ".mwfn";

	// Reading input file
	const std::vector<std::vector<double>> atoms = ReadXYZ(inp);
	const std::string basis = ReadBasisSet(inp);
	auto [na, nb] = ReadNumElectrons(inp);
	const int wfntype = ReadWfnType(inp);
	const int nthreads = ReadNumThreads(inp);
	const std::string jobtype = ReadJobType(inp);
	const std::string scf = ReadSCF(inp);
	const std::string guess = ReadGuess(inp);
	const std::string grid_str = ReadGrid(inp);
	const std::string method = ReadMethod(inp);
	const int derivative = ReadDerivative(inp);
	const double temperature = ReadTemperature(inp);
	const double chemicalpotential = ReadChemicalPotential(inp);
	
	// Deciding whether to read existing mwfn file.
	Mwfn mwfn;
	if ( atoms.empty() || basis.empty() || guess == "READ" ){
		std::printf("Reading existent Mwfn file %s ...\n", mwfn_name.c_str());
		mwfn = Mwfn(mwfn_name);
	}

	// Atoms and basis
	std::string basis_file_path_name = (std::string)std::getenv("CHINIUM_PATH") + "/BasisSets/" + basis + ".gbs";
	if ( !atoms.empty() ){
		if ( !basis.empty() ){
			std::printf("Setting atomic coordinates given by input file ...\n");
			mwfn.setCenters(atoms);
			std::printf("Reading basis set file %s ...\n", basis_file_path_name.c_str());
			mwfn.setBasis(basis_file_path_name);
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
			mwfn.setBasis(basis_file_path_name);
		} // Else do nothing (just keep the existent atoms and basis in the mwfn)
	}
	Normalize(&mwfn);
	mwfn.PrintCenters();

	// FT-DFT related
	Environment env(temperature, chemicalpotential);

	// Wavefunction type designation
	if ( guess != "READ" ){
		mwfn.Wfntype = wfntype;
		if ( wfntype == -1 ){
			if ( na == nb ) mwfn.Wfntype = 0;
			else mwfn.Wfntype = 1;
		}
	}
	if ( env.Temperature > 0 && mwfn.Wfntype == 2 ) throw std::runtime_error("FT-DFT cannot be used with spin-restricted open-shell wavefunctions!");

	// Initializing E and its derivatives
	double E_tot = 0;
	EigenMatrix Gradient = EigenZero(mwfn.getNumCenters(), 3);
	EigenMatrix Hessian = EigenZero(3 * mwfn.getNumCenters(), 3 * mwfn.getNumCenters());

	// Nuclear repulsion
	std::printf("Calculating nuclear repulsion energy ...\n");
	const auto [E_nuc, G_nuc, H_nuc] = mwfn.NuclearRepulsion();
	E_tot += E_nuc;
	Gradient += G_nuc;
	Hessian += H_nuc;

	// Electron integrals, density functional and grid
	Int2C1E int2c1e = Int2C1E(mwfn);
	Int4C2E int4c2e = Int4C2E(mwfn, 1, -1); // EXX is unknown by now.
	ExchangeCorrelation xc;
	Grid grid(&mwfn, grid_str, 1);
	int2c1e.CalculateIntegrals(0, 1);
	mwfn.Overlap = int2c1e.Overlap;
	if ( jobtype == "SCF" ){
		int4c2e.getRepulsionDiag(1);
		int4c2e.getRepulsionLength(1);
		int4c2e.getRepulsionIndices(1);
		int4c2e.getThreadPointers(nthreads, 1);
		int4c2e.CalculateIntegrals(0, 1);
		if ( method != "HF" || guess == "SAP" ){
			if ( method != "HF" ){
				xc.Read(method, 1);
				grid.Type = xc.Family == "LDA" ? 0 : xc.Family == "GGA" ? 1 : 2;
				grid.getAO(0, 1); // Prepared for SCF and CPSCF only, which use the AO values more than once. Higher-order derivatives that will be used only once will be computed in batches in HartreeFockKohnSham/HFKSDerivative.cpp to save memory.
				int4c2e.EXX = xc.EXX;
			}else grid.getAO(0, 1); // For SAP initial guess.
		}
		if ( guess == "READ" ){ // Orthogonalizing the orbitals read from mwfn.
			mwfn.Orthogonalize("Lowdin");
		}else{
			if ( mwfn.Wfntype == 0 ){
				mwfn.Orbitals.resize(mwfn.getNumBasis());
				EigenVector occ = EigenZero(mwfn.getNumBasis(), 1);
				for ( int i = 0; i < na; i++ ) occ(i) = 2;
				mwfn.setOccupation(occ);
			}else if ( mwfn.Wfntype == 1 ){
				mwfn.Orbitals.resize(mwfn.getNumBasis() * 2);
				EigenVector occ = EigenZero(mwfn.getNumBasis(), 1);
				for ( int i = 0; i < na; i++ ) occ(i) = 1;
				mwfn.setOccupation(occ, 1);
				occ.setZero();
				for ( int i = 0; i < nb; i++ ) occ(i) = 1;
				mwfn.setOccupation(occ, 2);
			}else if ( mwfn.Wfntype == 2 ){
				mwfn.Orbitals.resize(mwfn.getNumBasis());
				EigenVector occ = EigenZero(mwfn.getNumBasis(), 1);
				for ( int i = 0; i < ( ( na > nb ) ? na : nb ); i++ ){
					if ( i >= na || i >= nb ) occ(i) = 1;
					else occ(i) = 2;
				}
				mwfn.setOccupation(occ);
			}
			GuessSCF(mwfn, env, int2c1e, grid, guess, 1);
		}
		const double E_scf = HartreeFockKohnSham(mwfn, env, int2c1e, int4c2e, xc, grid, scf, 4, nthreads);
		E_tot += E_scf;
		std::printf("Total energy: %17.10f\n", E_tot);
		mwfn.PrintOrbitals();
		std::printf("Exporting wavefunction information to %s ...\n", mwfn_name.c_str());
		mwfn.Export(mwfn_name);

		if ( derivative > 0 ){
			int2c1e.CalculateIntegrals(1, 1);
			auto [grad, hess] = HFKSDerivative(mwfn, env, int2c1e, int4c2e, xc, grid, derivative, 2, nthreads);
			Gradient += grad;
			Hessian += hess;
		}

		if ( derivative > 0 ){
			std::printf("Total nuclear gradient:\n");
			for ( int iatom = 0; iatom < mwfn.getNumCenters(); iatom++ )
				std::printf("| %3d  %2s  % 10.17f  % 10.17f  % 10.17f\n", iatom, mwfn.Centers[iatom].getSymbol().c_str(), Gradient(iatom, 0), Gradient(iatom, 1), Gradient(iatom, 2));
		}
		if ( derivative > 1 ){
			std::printf("Total nuclear hessian:\n");
			for ( int xpert = 0; xpert < mwfn.getNumCenters() * 3; xpert++ ){
				std::printf("|");
				for (int ypert = 0; ypert <= xpert; ypert++ )
					std::printf(" % f", Hessian(xpert, ypert));
				std::printf("\n");
			}
		}
	} // if ( jobtype == "SCF" )
	else if ( jobtype == "LOCALIZATION" ){
		Localize(mwfn, int2c1e, method, "occ", 2);
		Localize(mwfn, int2c1e, method, "vir", 2);
		std::printf("Exporting wavefunction information to %s ...\n", mwfn_name.c_str());
		mwfn.Export(mwfn_name);
	}

	std::printf("*** Chinium terminated normally ***\n");
	return 0;
}

