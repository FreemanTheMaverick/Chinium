#include "Multiwfn/MwfnShell.h"
#include "Multiwfn/MwfnCenter.h"
#include "Multiwfn/MwfnOrbital.h"

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
	EigenMatrix Kinetic;
	EigenMatrix Nuclear;

	#include "Multiwfn/MwfnIO.h"
	#include "Integral/MwfnIntegral.h"
	#include "Grid/MwfnGrid.h"
	#include "HartreeFockKohnSham/MwfnHFKS.h"
	#include "Localization/MwfnLocalize.h"
};
