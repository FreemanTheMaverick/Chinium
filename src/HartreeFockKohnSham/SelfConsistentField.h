#pragma once

#include <Eigen/Core>
#include <string>
#include <libmwfn.h>

#include "../Macro.h"
#include "../Integral.h"
#include "../Grid.h"
#include "../ExchangeCorrelation.h"

class SCF{ public:
	double Energy = 0;
	EigenMatrix Gradient = EigenZero(0, 0);
	EigenMatrix Hessian = EigenZero(0, 0);

	int nthreads = 1;
	Int4C2E int4c2e;
	ExchangeCorrelation xc;
	Grid grid;
	std::string scftype = "DIIS";
	SCF(std::string inp, Mwfn& mwfn, Int2C1E& int2c1e);
};

#define __PostProcess0__(quantity)\
	std::printf("Total SCF %s: %17.10f\n", #quantity, Energy);\
	mwfn.PrintOrbitals();\
	const std::string mwfnname = basename + ".mwfn";\
	std::printf("Exporting wavefunction information to %s ...\n", mwfnname.c_str());\
	mwfn.Export(mwfnname);

#define __PostProcess1__\
	std::printf("Total nuclear gradient:\n");\
	for ( int iatom = 0; iatom < mwfn.getNumCenters(); iatom++ )\
		std::printf("| %3d  %2s  % 10.17f  % 10.17f  % 10.17f\n", iatom, mwfn.Centers[iatom].getSymbol().c_str(), Gradient(iatom, 0), Gradient(iatom, 1), Gradient(iatom, 2));

#define __PostProcess2__\
	std::printf("Total nuclear hessian:\n");\
	for ( int xpert = 0; xpert < mwfn.getNumCenters() * 3; xpert++ ){\
		std::printf("|");\
		for (int ypert = 0; ypert <= xpert; ypert++ )\
			std::printf(" % f", Hessian(xpert, ypert));\
		std::printf("\n");\
	}
