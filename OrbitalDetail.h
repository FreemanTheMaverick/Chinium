#include <libint2.hpp>
#include <vector>

using namespace libint2;

int nBasis(BasisSet obs){
	int n=0;
	for (const auto& shell:obs){
		n+=shell.size();
	}
	return n;
}

int nElectron(std::vector<Atom> atoms){
	int nelectron=0;
	for (int i=0;i<atoms.size();i++){
		nelectron+=atoms[i].atomic_number;
	}
	return nelectron;
}

int nOcc(std::vector<Atom> atoms){
	auto nocc=nElectron(atoms)/2;
	return nocc;
}

