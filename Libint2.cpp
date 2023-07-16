#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector> // Atom vectors.

std::vector<libint2::Atom> Libint2Atoms(const int natoms,double * atoms){ // Converting atoms array to libint's std::vector<libint2::Atom>.
	std::vector<libint2::Atom> libint2atoms(natoms);
	for (int iatom=0;iatom<natoms;iatom++){
		libint2::Atom atomi;
		atomi.atomic_number=(int)atoms[iatom*4];
		atomi.x=atoms[iatom*4+1];
		atomi.y=atoms[iatom*4+2];
		atomi.z=atoms[iatom*4+3];
		libint2atoms[iatom]=atomi;
	}
	return libint2atoms;
}

int nBasis_from_obs(libint2::BasisSet obs){ // Size of basis set directly derived from libint2::BasisSet.
	int n=0;
	for (const auto& shell:obs)
		n=n+shell.size();
	return n;
}
