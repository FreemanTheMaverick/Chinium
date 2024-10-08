#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cstdio>

#include "../Macro.h"
#include "../Multiwfn.h" // Requires <Eigen/Dense>, <vector>, <string>, "../Macro.h".

#include <iostream>


int MwfnShell::getSize(){
	if ( this->Type >= 0 )
		return ( this->Type + 1 ) * ( this->Type + 2 ) / 2;
	else
		return -2 * this->Type + 1;
}

int MwfnShell::getNumPrims(){
	return this->Exponents.size();
}

void MwfnShell::Print(){
	std::printf("Type: %d\n", this->Type);
	std::printf("Exponents and Coefficients:\n");
	for ( int i = 0; i < this->getNumPrims(); i++ )
		std::printf("%f %f\n", this->Exponents[i], this->Coefficients[i]);
}
