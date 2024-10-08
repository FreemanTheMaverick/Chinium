#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>
#include <string>
#include <iostream>

#include "../Macro.h"
#include "../Multiwfn.h" // Requires <Eigen/Dense>, <vector>, <string>, "Macro.h".

#include "Macro.h"

void Multiwfn::Normalize(){
	__Make_Basis_Set__
	int ishell = 0;
	for ( MwfnCenter& center : this->Centers ) for ( MwfnShell& shell : center.Shells ){
		std::copy(obs[ishell].contr[0].coeff.begin(), obs[ishell].contr[0].coeff.end(), shell.NormalizedCoefficients.begin());
		ishell++;
	}
}
