#include <cmath>
#include <vector>
#include <tuple>
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

std::tuple<int, int> ReadNumElectrons(std::string inp){
	std::ifstream file(inp);
	std::vector<std::vector<double>> atoms = ReadXYZ(inp);
	int ne = 0;
	for ( std::vector<double>& atom : atoms )
		ne += std::round(atom[0]);
	int charge = 0;
	int spin = 0;
	std::string thisline;
	bool found = 0;
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("CHARGE") == 0 ){
			found = 1;
			std::getline(file, thisline);
			if ( thisline.length() == 0 ) throw std::runtime_error("Missing charge!");
			std::stringstream ss(thisline);
			ss >> charge;
		}
	}
	file.clear(); file.seekg(0); // Going back to head of file.
	found = 0;
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("SPIN") == 0 ){
			found = 1;
			std::getline(file, thisline);
			if ( thisline.length() == 0 ) throw std::runtime_error("Missing spin!");
			std::stringstream ss(thisline);
			ss >> spin;
		}
	}
	ne -= charge;
	if ( spin == 0 && ne % 2 == 0 ) spin = 1;
	else if ( spin == 0 && ne % 2 == 1 ) spin = 2;
	if ( ( ne + spin ) % 2 == 0 ) throw std::runtime_error("Incompatible number of electrons and spin multiplicity!");
	int na = ( ne + ( spin - 1 ) ) / 2;
	int nb = ( ne - ( spin - 1 ) ) / 2;
	if ( na < 0 || nb < 0 ) throw std::runtime_error("Negative number of electrons!");
	return std::make_tuple(na, nb);
}

int ReadWfnType(std::string inp){
	std::ifstream file(inp);
	std::string thisline;
	bool found = 0;
	int wfntype = -1; // 0 for spin-restricted, 1 for spin-unrestricted, -1 for depending on nelec and spin.
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("WFNTYPE")==0 ){
			found = 1;
			std::getline(file, thisline);
			assert(thisline.length() > 0 && "Missing WfnType!");
			std::stringstream ss(thisline);
			ss >> wfntype;
		}
	}
	if ( wfntype != 0 && wfntype != 1 && wfntype != -1 ) std::runtime_error("Invalid type of wavefunction!");
	return wfntype;
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

std::string ReadSCF(std::string inp){
	std::ifstream file(inp);
	std::string thisline;
	bool found = 0;
	std::string scf = "DIIS";
	while ( std::getline(file, thisline) && ! found ){
		__To_Upper__(thisline);
		if ( thisline.compare("SCFTYPE") == 0 ){
			found = 1;
			std::getline(file, thisline);
			__To_Upper__(thisline);
			if ( thisline.length() == 0 ) throw std::runtime_error("Missing SCF TYPE!");
			std::stringstream ss(thisline);
			ss >> scf;
		}
	}
	if ( scf != "DIIS" && scf != "GRASSMANN" ) throw std::runtime_error("Invalid SCF type!");
	return scf;
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
			__To_Upper__(thisline);
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
	std::string method = "HF";
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
