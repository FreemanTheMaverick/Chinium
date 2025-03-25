#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>
#include <cstdio>
#include <cstddef>

#include "Macro.h"
#include "Multiwfn.h" // Requires <Eigen/Dense>, <vector>, <string>, "Macro.h".
#include "Gateway.h"

int main(int argc, char* argv[]){ (void)argc;
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
	const std::string grid = ReadGrid(inp);
	const std::string method = ReadMethod(inp);
	const int derivative = ReadDerivative(inp);
	const double temperature = ReadTemperature(inp);
	const double chemicalpotential = ReadChemicalPotential(inp);
	
	// Deciding whether to read existing mwfn file.
	Multiwfn mwfn;
	if ( atoms.empty() || basis.empty() || guess.compare("READ") == 0 )
		mwfn = Multiwfn(mwfn_name, 1);

	// Atoms and basis
	if ( !atoms.empty() ){
		if ( !basis.empty() ){
			mwfn.setCenters(atoms, 1);
			mwfn.setBasis(basis, 1);
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
		if ( !basis.empty() ) mwfn.setBasis(basis, 1);
		// Else do nothing (just keep the existent atoms and basis in the mwfn)
	}
	mwfn.PrintCenters();

	if ( !grid.empty() ) mwfn.GenerateGrid(grid, 1);

	// FT-DFT related
	mwfn.Temperature = temperature;
	mwfn.ChemicalPotential = chemicalpotential;

	// Initializing E and its derivatives
	if ( guess.compare("READ") != 0 ){
		mwfn.Wfntype = wfntype;
		if ( wfntype == -1 ){
			if ( na == nb ) mwfn.Wfntype = 0;
			else mwfn.Wfntype = 1;
		}
	}
	mwfn.E_tot = 0;
	mwfn.Gradient = EigenZero(mwfn.getNumCenters(), 3);
	mwfn.Hessian = EigenZero(3 * mwfn.getNumCenters(), 3 * mwfn.getNumCenters());

	// Nuclear repulsion
	mwfn.NuclearRepulsion({0, 1, 2}, 1);

	// Electron integrals
	if ( jobtype.compare("SCF") == 0 ){
		switch(derivative){
			case 0: mwfn.getTwoCenter({0}, 1); break;
			case 1: case 2: mwfn.getTwoCenter({0, 1}, 1); break;
		}
		mwfn.getRepulsion({0}, -1., nthreads, 1);

		if ( method.compare("HF") != 0 || guess.compare("SAP") == 0 ){
			if ( method.compare("HF") != 0 ){
				mwfn.XC.Read(method, 1);
				mwfn.PrepareXC("ev",1);
				if ( derivative > 1 ) mwfn.PrepareXC("f",1);
				if ( mwfn.XC.XCfamily.compare("LDA") == 0 ) switch(derivative){ // Prepared for SCF and CPSCF only, which use the AO values more than once. Higher-order derivatives that will be used only once will be computed in batches in HartreeFockKohnSham/HFKSDerivative.cpp to save memory.
					case 0:
						mwfn.getGridAO(0, 1); break;
					case 1:
					case 2:
						mwfn.getGridAO(1, 1); break;
				}else if ( mwfn.XC.XCfamily.compare("GGA") == 0 ) switch(derivative){
					case 0:
						mwfn.getGridAO(1, 1); break;
					case 1:
					case 2:
						mwfn.getGridAO(2, 1); break;
				}
			}else mwfn.getGridAO(0, 1); // For SAP initial guess.
		} // if ( method.compare("HF") != 0 || guess.compare("SAP") == 0 )
		if ( guess.compare("READ") == 0 ){ // Orthogonalizing the orbitals read from mwfn.
			Eigen::SelfAdjointEigenSolver<EigenMatrix> solver;
			const EigenMatrix S = mwfn.Overlap;
			if ( mwfn.Wfntype == 0 ){
				const EigenMatrix C = mwfn.getCoefficientMatrix(0);
				solver.compute(C.transpose() * S * C);
				const EigenMatrix X = solver.operatorInverseSqrt();
				mwfn.setCoefficientMatrix(C * X, 0);
			}else if ( mwfn.Wfntype == 1 ){
				const EigenMatrix Ca = mwfn.getCoefficientMatrix(1);
				solver.compute(Ca.transpose() * S * Ca);
				const EigenMatrix Xa = solver.operatorInverseSqrt();
				mwfn.setCoefficientMatrix(Ca * Xa, 1);
				const EigenMatrix Cb = mwfn.getCoefficientMatrix(2);
				solver.compute(Cb.transpose() * S * Cb);
				const EigenMatrix Xb = solver.operatorInverseSqrt();
				mwfn.setCoefficientMatrix(Cb * Xb, 2);
			}
		}else{
			if ( mwfn.Wfntype == 0 ){
				mwfn.Orbitals.resize(mwfn.getNumBasis());
				mwfn.setOccupation((EigenVector)(EigenZero(na, 1).array() + 2).matrix());
			}else if ( mwfn.Wfntype == 1 ){
				mwfn.Orbitals.resize(mwfn.getNumBasis() * 2);
				mwfn.setOccupation((EigenVector)(EigenZero(na, 1).array() + 1).matrix(), 1);
				mwfn.setOccupation((EigenVector)(EigenZero(nb, 1).array() + 1).matrix(), 2);
			}
			mwfn.GuessSCF(guess, 1);
		}
		mwfn.HartreeFockKohnSham(scf, 4, nthreads);
		std::printf("Total energy: %17.10f\n", mwfn.E_tot);
		mwfn.PrintOrbitals();
		mwfn.Export(mwfn_name, 1);

		if ( derivative > 0 ){
			if ( derivative > 1 ) mwfn.getTwoCenter({2}, 1);
			switch(derivative){
				case 1: mwfn.getRepulsion({1}, -1., nthreads, 1); break;
				case 2: mwfn.getRepulsion({1, 2}, -1., nthreads, 1); break;
			}
			mwfn.HFKSDerivative(derivative, 2, nthreads);
			if ( derivative > 0 ){
				std::printf("Total nuclear gradient:\n");
				for ( int iatom = 0; iatom < mwfn.getNumCenters(); iatom++ )
					std::printf("| %3d  %2s  % 10.17f  % 10.17f  % 10.17f\n", iatom, mwfn.Centers[iatom].getSymbol().c_str(), mwfn.Gradient(iatom, 0), mwfn.Gradient(iatom, 1), mwfn.Gradient(iatom, 2));
			}
			if ( derivative > 1 ){
				std::printf("Total nuclear hessian:\n");
				for ( int xpert = 0; xpert < mwfn.getNumCenters() * 3; xpert++ ){
					std::printf("|");
					for (int ypert = 0; ypert <= xpert; ypert++ )
						std::printf(" % f", mwfn.Hessian(xpert, ypert));
					std::printf("\n");
				}
			}
		}
	} // if ( jobtype.compare("SCF") == 0 )

	else if ( jobtype.compare("LOCALIZATION") == 0 ){
		mwfn.Localize(method, "both", 2);
		mwfn.Export(mwfn_name, 1);
	}

	std::printf("*** Chinium terminated normally ***\n");
	return 0;
}

