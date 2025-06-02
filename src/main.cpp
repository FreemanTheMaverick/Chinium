#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <cstddef>

#include "Macro.h"
#include "Gateway.h"
#include "Multiwfn/Multiwfn.h" // Requires <Eigen/Dense>, <vector>, <string>, "Macro.h".
#include "Integral/Int2C1E.h"
#include "Integral/Int4C2E.h"
#include "Grid/Grid.h"
#include "ExchangeCorrelation.h"
#include "HartreeFockKohnSham/HartreeFockKohnSham.h"
#include "Localization/Localize.h"

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
	const std::string grid_str = ReadGrid(inp);
	const std::string method = ReadMethod(inp);
	const int derivative = ReadDerivative(inp);
	const double temperature = ReadTemperature(inp);
	const double chemicalpotential = ReadChemicalPotential(inp);
	
	// Deciding whether to read existing mwfn file.
	Multiwfn mwfn;
	if ( atoms.empty() || basis.empty() || guess == "READ" )
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

	// FT-DFT related
	mwfn.Temperature = temperature;
	mwfn.ChemicalPotential = chemicalpotential;

	// Initializing E and its derivatives
	if ( guess != "READ" ){
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

	// Electron integrals, density functional and grid
	Int2C1E int2c1e = Int2C1E(mwfn, 1);
	Int4C2E int4c2e = Int4C2E(mwfn, 1, -1, 1); // EXX is unknown by now.
	ExchangeCorrelation xc;
	Grid grid(&mwfn, grid_str, 1);
	if ( jobtype == "SCF" ){
		int2c1e.CalculateIntegrals(0);
		int4c2e.getRepulsionDiag();
		int4c2e.getRepulsionLength();
		int4c2e.getRepulsionIndices();
		int4c2e.getThreadPointers(nthreads);
		int4c2e.CalculateIntegrals(0);
		if ( method != "HF" || guess == "SAP" ){
			if ( method != "HF" ){
				xc.Read(method, 1);
				grid.Type = xc.Family == "LDA" ? 0 : xc.Family == "GGA" ? 1 : 2;
				grid.getAO(0, 1); // Prepared for SCF and CPSCF only, which use the AO values more than once. Higher-order derivatives that will be used only once will be computed in batches in HartreeFockKohnSham/HFKSDerivative.cpp to save memory.
				int4c2e.EXX = xc.EXX;
			}else grid.getAO(0, 1); // For SAP initial guess.
		}
		if ( guess == "READ" ){ // Orthogonalizing the orbitals read from mwfn.
			Eigen::SelfAdjointEigenSolver<EigenMatrix> solver;
			const EigenMatrix S = int2c1e.Overlap;
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
			GuessSCF(mwfn, int2c1e, grid, guess, 1);
		}
		HartreeFockKohnSham(mwfn, int2c1e, int4c2e, xc, grid, scf, 4, nthreads);
		if (xc) grid.SaveDensity();
		std::printf("Total energy: %17.10f\n", mwfn.E_tot);
		mwfn.PrintOrbitals();
		mwfn.Export(mwfn_name, 1);

		if ( derivative > 0 ){
			int2c1e.CalculateIntegrals(1);
			auto [grad, hess] = HFKSDerivative(mwfn, int2c1e, int4c2e, xc, grid, derivative, 2, nthreads);
			mwfn.Gradient += grad;
			mwfn.Hessian += hess;
		}

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
	} // if ( jobtype == "SCF" )
	else if ( jobtype == "LOCALIZATION" ){
		Localize(mwfn, int2c1e, method, "occ", 2);
		Localize(mwfn, int2c1e, method, "vir", 2);
		mwfn.Export(mwfn_name, 1);
	}

	std::printf("*** Chinium terminated normally ***\n");
	return 0;
}

