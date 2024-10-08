#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <cassert>
#include <regex>

#include "../Macro.h"
#include "../Multiwfn.h" // Requires <Eigen/Dense>, <vector>, <string>, "Macro.h".

#include <iostream>


std::vector<MwfnCenter> MwfnReadBasis(std::string basis_filename, bool output){
	std::ifstream file(basis_filename.c_str());
	assert("Basis set file is missing!" && file.good());
	if (output) std::printf("Reading basis set file %s ...\n", basis_filename.c_str());
	std::string line, word;
	std::vector<MwfnCenter> centers={};
	MwfnCenter center;
	MwfnShell shell;
	MwfnShell shell2;
	__Name_2_Z__
	std::regex re("D|d");
	while ( std::getline(file, line) ){
		if ( line.size() == 0 ) continue;
		std::stringstream ss(line);
		ss >> word;
		if ( word[0] == '-' ){
			word.erase(0, 1);
			center.Index = Name2Z[word];
		}else if (
				word == "S" || word == "SP" || word == "P" || word == "D" ||
				word == "F" || word == "G"  || word == "H" || word == "I" ){
			if ( word == "S" ){
				shell.Type = 0;
			}else if ( word == "SP" ){
				shell.Type = 0;
				shell2.Type = 1;
			}else if ( word == "P" ){
				shell.Type = 1;
			}else if ( word == "D" ){
				shell.Type = -2;
			}else if ( word == "F" ){
				shell.Type = -3;
			}else if ( word == "G" ){
				shell.Type = -4;
			}else if ( word == "H" ){
				shell.Type = -5;
			}else if ( word == "I" ){
				shell.Type = -6;
			}
			ss >> word;
			int n = std::stoi(word);
			for ( int i = 0; i < n; i++ ){
				std::getline(file, line);
				std::stringstream ss(line);
				ss >> word; word = std::regex_replace(word, re, "E");
				shell.Exponents.push_back(std::stod(word));
				if ( shell2.Type != -114 ){
					shell2.Exponents.push_back(std::stod(word));
				}
				ss >> word; word = std::regex_replace(word, re, "E");
				shell.Coefficients.push_back(std::stod(word));
				if ( shell2.Type != -114 ){
					ss >> word; word = std::regex_replace(word, re, "E");
					shell2.Coefficients.push_back(std::stod(word));
				}
			}
			center.Shells.push_back(shell);
			if ( shell2.Type != -114 ){
				center.Shells.push_back(shell2);
			}
			shell.Type = -114;
			shell.Exponents.resize(0);
			shell.Coefficients.resize(0);
			shell2.Type = -114;
			shell2.Exponents.resize(0);
			shell2.Coefficients.resize(0);
		}else if ( word == "****" ){
			centers.push_back(center);
			center.Shells.resize(0);
		}
	}
	return centers;
}

/*
int main(){
	std::vector<MwfnCenter> centers = MwfnReadBasis("a", 1);
	for (MwfnCenter& center : centers) center.Print();
	return 0;
}
*/
