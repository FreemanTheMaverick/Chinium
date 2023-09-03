#include <sstream>
#include <fstream>
#include <iostream>		
#include <string>
#include <map>
#include <cassert>
#include "Aliases.h"

int ReadXYZ(char * inp,double * atoms,const bool output){
	std::ifstream file(inp);
	__Name_2_Z__
	std::string thisline;
	bool found=0;
	int natoms=0;
	while (getline(file,thisline) && ! found){
		if (thisline.compare("xyz")==0){
			found=1;
			getline(file,thisline);
			std::stringstream ss(thisline);
			ss>>natoms;
			if (output) std::cout<<"Number of atoms ... "<<natoms<<std::endl;
			if (output) std::cout<<"Atom coordinates in bohrs:"<<std::endl;
			for (int iatom=0;iatom<natoms;iatom++){
				std::getline(file,thisline);	
				std::stringstream ss(thisline);
				std::string elementname;ss>>elementname;atoms[4*iatom]=Name2Z[elementname];
				double x;ss>>x;atoms[4*iatom+1]=x*__angstrom2bohr__;
				double y;ss>>y;atoms[4*iatom+2]=y*__angstrom2bohr__;
				double z;ss>>z;atoms[4*iatom+3]=z*__angstrom2bohr__;
				if (output) std::cout<<"| "<<iatom<<" "<<Name2Z[elementname]<<" "<<elementname<<" "<<x*__angstrom2bohr__<<" "<<y*__angstrom2bohr__<<" "<<z*__angstrom2bohr__<<std::endl;
			}
		}
	}
	return natoms;
}

std::string ReadBasisSet(char * inp,const bool output){
	std::ifstream file(inp);
	std::string basis;
	std::string thisline;
	bool found=0;
	while (getline(file,thisline) && ! found){
		if (thisline.compare("basis")==0){
			found=1;
			getline(file,thisline);
			std::stringstream ss(thisline);
			ss>>basis;
		}
	}
	if (output) std::cout<<"Basis set ... "<<basis<<std::endl;
	return basis;
}

int ReadNElectrons(char * inp,const bool output){
	std::ifstream file(inp);
	double atoms[10000];
	const int natoms=ReadXYZ(inp,atoms,0);
	int ne=0;
	for (int iatom=0;iatom<natoms;iatom++)
		ne+=(int)atoms[iatom*4];
	int charge=0;
	std::string thisline;
	bool found=0;
	while (getline(file,thisline) && ! found){
		if (thisline.compare("charge")==0){
			found=1;
			getline(file,thisline);
			std::stringstream ss(thisline);
			ss>>charge;
		}
	}
	ne+=charge;
	if (output) std::cout<<"Charge ... "<<charge<<std::endl;
	if (output) std::cout<<"Number of electrons ... "<<ne<<std::endl;
	return ne;
}

int ReadNProcs(char * inp,const bool output){
	std::ifstream file(inp);
	std::string thisline;
	bool found=0;
	int nprocs=1;
	while (getline(file,thisline) && ! found){
		if (thisline.compare("nprocs")==0){
			found=1;
			getline(file,thisline);
			std::stringstream ss(thisline);
			ss>>nprocs;
		}
	}
	if (output) std::cout<<"Number of threads ... "<<nprocs<<std::endl;
	return nprocs;
}

std::string ReadGuess(char * inp,const bool output){
	std::ifstream file(inp);
	std::string thisline;
	bool found=0;
	std::string guess="sad";
	while (getline(file,thisline) && ! found){
		if (thisline.compare("guess")==0){
			found=1;
			getline(file,thisline);
			std::stringstream ss(thisline);
			ss>>guess;
		}
	}
	if (guess.compare("core")==0 && output)
		std::cout<<"Initial guessed fock matrix ... Core"<<std::endl;
	else if (guess.compare("sad")==0 && output)
		std::cout<<"Initial guessed density matrix ... SAD"<<std::endl;
	else if (guess.compare("sap")==0 && output)
		std::cout<<"Initial guessed fock matrix ... SAP"<<std::endl;
	else if (output)
		std::cout<<"Initial guessed density matrix ... taken from '"<<guess<<"'"<<std::endl;
	return guess;
}

std::string ReadGrid(char * inp,const bool output){
	std::ifstream file(inp);
	std::string thisline;
	bool found=0;
	std::string grid="";
	while (getline(file,thisline) && ! found){
		if (thisline.compare("grid")==0){
			found=1;
			getline(file,thisline);
			std::stringstream ss(thisline);
			ss>>grid;
		}
	}
	if (!grid.empty()){
		std::string filename=std::string(__Grid_library_path__)+grid;
		std::ifstream gridfile(filename);
		assert(("Cannot find the grid folder in " __Grid_library_path__ && gridfile.good()));

		if (output)
			std::cout<<"Grid ... "<<grid<<" found in "<<std::string(__Grid_library_path__)<<std::endl;
	}
	return grid;
}

std::string ReadMethod(char * inp,const bool output){
	std::ifstream file(inp);
	std::string thisline;
	bool found=0;
	std::string method="rhf";
	while (getline(file,thisline) && ! found){
		if (thisline.compare("method")==0){
			found=1;
			getline(file,thisline);
			std::stringstream ss(thisline);
			ss>>method;
		}
	}
	if (method!="rhf"){
		std::string filename=std::string(__DF_library_path__)+method+".df";
		std::ifstream dffile(filename);
		assert("Cannot find the DF file in " __DF_library_path__ && dffile.good());
		if (output) std::cout<<"Computational method ... "<<method<<" found in "<<std::string(__DF_library_path__)<<std::endl;
	}
	else if (output) std::cout<<"Computational method ... RHF"<<std::endl;
	return method;
}

int ReadDerivative(char * inp,const bool output){
	std::ifstream file(inp);
	std::string thisline;
	bool found=0;
	double order=0;
	while (getline(file,thisline) && ! found){
		if (thisline.compare("derivative")==0){
			found=1;
			getline(file,thisline);
			std::stringstream ss(thisline);
			ss>>order;
		}
	}
	if (output) std::cout<<"Order of nuclear derivatives ... "<<order<<std::endl;
	return order;
}


/*
int main(int argc,char *argv[]){
	double * atoms=new double[10000];
	int natoms=ReadXYZ(argv[1],atoms);
	for (int iatom=0;iatom<natoms;iatom++){
		std::cout<<atoms[4*iatom]<<" "<<atoms[4*iatom+1]<<" "<<atoms[4*iatom+2]<<" "<<atoms[4*iatom+3]<<std::endl;
	}
	char * basis=new char[10];
	ReadBasisSet(argv[1],basis);
	std::cout<<basis<<std::endl;
	delete [] atoms;
	delete [] basis;
}
*/
