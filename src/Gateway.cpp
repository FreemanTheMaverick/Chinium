#include <cmath>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>		
#include <string>
#include <algorithm>
#include <map>
#include <cassert>

#include "Macro.h"

#define __To_Upper__(str)\
	std::transform(str.begin(), str.end(), str.begin(), ::toupper);

std::vector<std::vector<double>> ReadXYZ(std::string inp){
	std::vector<std::vector<double>> atoms;
	std::ifstream file(inp);
	__Name_2_Z__
	std::string thisline;
	bool found = 0;
	int natoms = 0;
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("XYZ") == 0 ){
			found = 1;
			std::getline(file, thisline);
			__To_Upper__(thisline);
			if ( thisline.compare("READ") == 0 )
				return {};
			std::stringstream ss(thisline);
			ss >> natoms;
			assert(natoms > 0 && "Invalid number of atoms!");
			atoms.resize(natoms);
			for ( int iatom = 0; iatom < natoms; iatom++ ){
				std::getline(file, thisline);
				__To_Upper__(thisline);
				assert(thisline.length() > 0 && "Missing atom!");
				std::stringstream ss_(thisline);
				std::string symbol; ss_ >> symbol;
				const double Z = Name2Z[symbol];
				double x; ss_ >> x;
				double y; ss_ >> y;
				double z; ss_ >> z;
				atoms[iatom] = {
					Z, Z,
					x * __angstrom2bohr__,
					y * __angstrom2bohr__,
					z * __angstrom2bohr__
				};
			}
		}
	}
	assert(found && "Missing atomic coordinates");
	return atoms;
}

std::string ReadBasisSet(std::string inp){
	std::ifstream file(inp);
	std::string basis;
	std::string thisline;
	bool found = 0;
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("BASIS") == 0 ){
			found = 1;
			std::getline(file, thisline);
			std::string thisline_ = thisline;
			__To_Upper__(thisline_);
			if ( thisline_.compare("READ") == 0 )
				return "";
			std::stringstream ss(thisline);
			ss >> basis;
		}
	}
	assert(basis.length() > 0 && "Missing basis set name!");
	return basis;
}

int ReadNumElectrons(std::string inp){
	std::ifstream file(inp);
	std::vector<std::vector<double>> atoms = ReadXYZ(inp);
	int ne = 0;
	for ( std::vector<double>& atom : atoms )
		ne += std::round(atom[0]);
	int charge = 0;
	std::string thisline;
	bool found = 0;
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("CHARGE") == 0 ){
			found = 1;
			std::getline(file, thisline);
			assert(thisline.length() > 0 && "Missing charge!");
			std::stringstream ss(thisline);
			ss >> charge;
		}
	}
	ne -= charge;
	assert(ne > 0 && "Invalid number of electrons!");
	return ne;
}

int ReadNumThreads(std::string inp){
	std::ifstream file(inp);
	std::string thisline;
	bool found = 0;
	int nthreads = 1;
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("NTHREADS")==0 ){
			found = 1;
			std::getline(file, thisline);
			assert(thisline.length() > 0 && "Missing nthreads!");
			std::stringstream ss(thisline);
			ss >> nthreads;
		}
	}
	assert(nthreads > 0 && "Invalid number of threads!");
	return nthreads;
}

std::string ReadJobType(std::string inp){
	std::ifstream file(inp);
	std::string thisline;
	bool found = 0;
	std::string jobtype = "SCF";
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("JOBTYPE") == 0 ){
			found = 1;
			std::getline(file, thisline);
			__To_Upper__(thisline);
			assert(thisline.length() > 0 && "Missing job type!");
			std::stringstream ss(thisline);
			ss >> jobtype;
		}
	}
	assert(( jobtype.compare("SCF") == 0 || jobtype.compare("LOCALIZATION") == 0 ) && "Invalid job type!");
	return jobtype;
}

std::string ReadGuess(std::string inp){
	std::ifstream file(inp);
	std::string thisline;
	bool found = 0;
	std::string guess = "SAP";
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("GUESS") == 0 ){
			found = 1;
			std::getline(file, thisline);
			__To_Upper__(thisline);
			assert(thisline.length() > 0 && "Missing guess!");
			std::stringstream ss(thisline);
			ss >> guess;
		}
	}
	assert(( guess.compare("SAP") == 0 || guess.compare("READ") == 0 ) && "Invalid guess!");
	return guess;
}

std::string ReadGrid(std::string inp){
	std::ifstream file(inp);
	std::string thisline;
	bool found = 0;
	std::string grid = "";
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("GRID") == 0){
			found = 1;
			std::getline(file, thisline);
			assert(thisline.length() > 0 && "Missing grid!");
			std::stringstream ss(thisline);
			ss >> grid;
		}
	}
	return grid;
}

std::string ReadMethod(std::string inp){
	std::ifstream file(inp);
	std::string thisline;
	bool found = 0;
	std::string method = "RHF";
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("METHOD") == 0 ){
			found = 1;
			std::getline(file, thisline);
			__To_Upper__(thisline);
			assert(thisline.length() > 0 && "Missing method!");
			std::stringstream ss(thisline);
			ss >> method;
		}
	}
	return method;
}

int ReadDerivative(std::string inp){
	std::ifstream file(inp);
	std::string thisline;
	bool found = 0;
	double order = 0;
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("DERIVATIVE") == 0 ){
			found = 1;
			std::getline(file, thisline);
			assert(thisline.length() > 0 && "Missing derivative!");
			std::stringstream ss(thisline);
			ss >> order;
		}
	}
	assert(order >= 0 && "Invalid order of derivative!");
	return order;
}

double ReadTemperature(std::string inp){
	std::ifstream file(inp);
	std::string thisline;
	bool found = 0;
	double temperature = 0;
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if (thisline.compare("TEMPERATURE")==0){
			found = 1;
			std::getline(file, thisline);
			std::stringstream ss(thisline);
			assert(thisline.length() > 0 && "Missing temperature!");
			ss >> temperature;
		}
	}
	assert(temperature >= 0 && "Invalid temperature!");
	return temperature;
}

double ReadChemicalPotential(std::string inp){
	std::ifstream file(inp);
	std::string thisline;
	bool found = 0;
	double chemicalpotential = 0;
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("CHEMICALPOTENTIAL") == 0 ){
			found = 1;
			std::getline(file, thisline);
			std::stringstream ss(thisline);
			assert(thisline.length() > 0 && "Missing chemical potential!");
			ss >> chemicalpotential;
		}
	}
	return chemicalpotential;
}
