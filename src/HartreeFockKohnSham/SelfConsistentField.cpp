#include <string>
#include <tuple>
#include <cstdio>
#include <libmwfn.h>

#include "../Macro.h"
#include "../Gateway.h"
#include "../Integral.h"
#include "../Grid.h"
#include "../ExchangeCorrelation.h"

#include "SelfConsistentField.h"
#include "GuessSCF.h"

SCF::SCF(std::string inp, Mwfn& mwfn, Int2C1E& int2c1e){
	nthreads = ReadNumThreads(inp);

	// Nuclear repulsion
	std::tie(Energy, Gradient, Hessian) = mwfn.NuclearRepulsion();
	std::printf("Nuclear repulsion energy ... %10.17f\n", Energy);

	const std::string method = ReadMethod(inp);
	const std::string guess = ReadGuess(inp);

	// Grid, exchange-correlation functional and initial guess
	const std::string grid_str = ReadGrid(inp);
	grid = Grid(&mwfn, grid_str, nthreads, 1);
	if ( method != "HF" || guess == "SAP" ){
		if ( grid.SubGridBatches[0][0]->NumGrids == 0 ) throw std::runtime_error("For DFT and SAP, you must specify the grid!");
		if ( method != "HF" ){
			xc.Read(method, 1);
			grid.setType( xc.Family == "LDA" ? 0 : xc.Family == "GGA" ? 1 : 2 );
			grid.getAO(0, 1);
		}else{
			grid.setType(0);
			grid.getAO(0, 1); // For SAP initial guess.
			grid.setType(-1);
		}
		if ( guess != "READ" ) GuessSCF(mwfn, int2c1e, grid, guess, 1);
	}

	// SCF type
	scftype = ReadSCF(inp);

	// Two-electron integrals
	if ( scftype != "DRY" ){
		int4c2e = Int4C2E(mwfn, 1, -1);
		if (xc) int4c2e.EXX = xc.EXX;
		int4c2e.getRepulsionDiag(1);
		int4c2e.getRepulsionLength(1);
		int4c2e.getRepulsionIndices(1);
		int4c2e.getThreadPointers(nthreads, 1);
		int4c2e.CalculateIntegrals(0, 1);
	}
}
