#include <vector>
#include <string>
#include <cstdio>

#include "../Macro.h"
#include "MwfnShell.h"
#include "MwfnCenter.h"


int MwfnCenter::getNumShells(){
	return this->Shells.size();
}

int MwfnCenter::getNumBasis(){
	int nbasis = 0;
	for ( MwfnShell& shell : this->Shells )
		nbasis += shell.getSize();
	return nbasis;
}

std::string MwfnCenter::getSymbol(){
	__Z_2_Name__
	return Z2Name[this->Index];
}

void MwfnCenter::Print(){
	std::printf("Symbol: %s\n", this->getSymbol().c_str());
	std::printf("Index: %d\n", this->Index);
	std::printf("Nuclear charge: %f\n", this->Nuclear_charge);
	std::printf("Coordinates (a.u.): %f %f %f\n", this->Coordinates[0], this->Coordinates[1], this->Coordinates[2]);
	std::printf("Shells:\n");
	for ( MwfnShell& shell : this->Shells ) shell.Print();
}
