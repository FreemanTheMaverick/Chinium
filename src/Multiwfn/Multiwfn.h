#pragma once
#include "MwfnShell.h"
#include "MwfnCenter.h"
#include "MwfnOrbital.h"
#include "../ExchangeCorrelation/MwfnXC1.h"

class Multiwfn{ public:
	// Field 1
	int Wfntype = -114;
	double E_tot = -114;
	double VT_ratio = -114;

	// Fields 2 & 3
	std::vector<MwfnCenter> Centers = {};

	// Field 4
	std::vector<MwfnOrbital> Orbitals = {};

	// Field 5
	EigenMatrix Overlap;
	/*
	EigenMatrix Kinetic;
	EigenMatrix Nuclear;
	*/

	EigenMatrix Gradient;
	EigenMatrix Hessian;
	double Temperature = 0;
	double ChemicalPotential = 0;

	#include "MwfnIO.h"
	#include "../Grid/MwfnGrid.h"
	#include "../ExchangeCorrelation/MwfnXC2.h"
};
