#include <sstream>
#include <fstream>
#include <iostream>		
#include <map>
#include <string>

#define __angstrom2bohr__ 1.8897259886

int ReadXYZ(char * inp,double * atoms,const bool output){
	std::ifstream file(inp);
	std::map<std::string,double> ElementName2Z={
		{"H",1},{"He",2},{"Li",3},{"Be",4},{"B",5},
		{"C",6},{"N",7},{"O",8},{"F",9},{"Ne",10}
	};
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
				std::string elementname;ss>>elementname;atoms[4*iatom]=ElementName2Z[elementname];
				double x;ss>>x;atoms[4*iatom+1]=x*__angstrom2bohr__;
				double y;ss>>y;atoms[4*iatom+2]=y*__angstrom2bohr__;
				double z;ss>>z;atoms[4*iatom+3]=z*__angstrom2bohr__;
				if (output) std::cout<<" "<<iatom<<" "<<elementname<<" "<<ElementName2Z[elementname]<<" "<<x*__angstrom2bohr__<<" "<<y*__angstrom2bohr__<<" "<<z*__angstrom2bohr__<<std::endl;
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
		std::cout<<"Initial guessed density matrix ... Core"<<std::endl;
	else if (guess.compare("sad")==0 && output)
		std::cout<<"Initial guessed density matrix ... SAD"<<std::endl;
	else if (output)
		std::cout<<"Initial guessed density matrix ... taken from '"<<guess<<"'"<<std::endl;
	return guess;
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
