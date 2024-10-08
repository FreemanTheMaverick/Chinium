#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cstdio>

#include "../Macro.h"
#include "../Multiwfn.h" // Requires <Eigen/Dense>, <vector>, <string>, "../Macro.h".

#include <iostream>


int MwfnCenter::getNumShells(){
	return this->Shells.size();
}

std::string MwfnCenter::getSymbol(){
	__Z_2_Name__
	return Z2Name[this->Index].c_str();
}

void MwfnCenter::Print(){
	std::printf("Symbol: %s\n", this->getSymbol().c_str());
	std::printf("Index: %d\n", this->Index);
	std::printf("Nuclear charge: %f\n", this->Nuclear_charge);
	std::printf("Coordinates (a.u.): %f %f %f\n", this->Coordinates[0], this->Coordinates[1], this->Coordinates[2]);
	std::printf("Shells:\n");
	for ( MwfnShell& shell : this->Shells ) shell.Print();
}

