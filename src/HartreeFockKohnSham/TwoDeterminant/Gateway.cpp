#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>

#include "../TwoDeterminant.h"

#define __To_Upper__(str)\
	std::transform(str.begin(), str.end(), str.begin(), ::toupper);

namespace{

int ReadType(std::string inp){
	std::ifstream file(inp);
	std::string thisline;
	bool found = 0;
	int tdtype = 1;
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline == "TWODETTYPE" ){
			found = 1;
			std::getline(file, thisline);
			__To_Upper__(thisline);
			if ( thisline.length() == 0 ) throw std::runtime_error("Missing two-determinant ROKS type!");
			std::stringstream ss(thisline);
			ss >> tdtype;
		}
	}
	if ( tdtype != 1 && tdtype != 2 ) throw std::runtime_error("Invalid two-determinant ROKS type!");
	return tdtype;
}

}

TwoDet::TwoDet(std::string inp): RO_SCF(inp){
	if ( Na != 1 && Nb != 1 ) throw std::runtime_error("Two-determinant ROKS requires exactly one unpaired alpha electron and one unpaired beta electron! (Keyword: spin 2 2)");
	TwoDetType = ReadType(inp);
	if ( TwoDetType == 1 ){
		grid2 = grid;
		for ( auto& subgridbatch : grid2.SubGridBatches ) for ( auto& subgrid : subgridbatch )
			subgrid->Spin = 2;
	}else{
		xc.Spin = 1;
		for ( auto& subgridbatch : grid.SubGridBatches ) for ( auto& subgrid : subgridbatch )
			subgrid->Spin = 1;
	}
}
