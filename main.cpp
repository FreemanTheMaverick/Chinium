#include "OneElectronIntegrals.h"
#include "OrbitalDetail.h"
#include <vector>
#include <iostream>
#include <libint2.hpp>

int main(){
	initialize();
	ifstream input_file("f.xyz");
	vector<Atom> atoms=read_dotxyz(input_file);
	BasisSet obs("sto-3g",atoms);
	int nbasis=nBasis(obs);
	Matrix overlap=Overlap(obs,nbasis);
	cout<<overlap<<endl;
	Matrix kinetic=Kinetic(obs,nbasis);
	cout<<kinetic<<endl;
	Matrix nuclear=Nuclear(obs,nbasis,atoms);
	cout<<nuclear<<endl;
	finalize();
}
	

