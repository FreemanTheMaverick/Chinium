#include <Eigen/Dense>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <map>
#include <regex>

#include "../Macro.h"

#include "MwfnIO.h" // Requires <Eigen/Dense>, <vector>, <string>, "Macro.h".


#define __Construct_Orbitals__\
	this->Orbitals.resize(nindbasis * ( this->Wfntype == 0 ? 1 : 2 ));\
	for ( MwfnOrbital& orbital : this->Orbitals ){\
		orbital.Coeff.resize(nbasis);\
	}\

#define __Read_Array_Head__\
	int k = 0;\
	while ( std::getline(file, line) && line.length() ){\
		std::stringstream ss_(line);\
		while ( ss_ >> word && k < total){

#define __Read_Array_Tail__\
			k++;\
		}\
		if ( k == total ) break;\
	}

#define __Read_Array2_Head__\
	int k = 0;\
	int l = 0;\
	while ( std::getline(file, line) && line.length() ){\
		std::stringstream ss_(line);\
		while ( ss_ >> word && k < total ){

#define __Read_Array2_Tail__\
			l++;\
			if ( l == total2 ){\
				k++;\
				l = 0;\
			}\
		}\
		if ( k == total ) break;\
	}

#define __Load_Matrix__(mat)\
	int irow = 0;\
	int jcol = 0;\
	int total = lower ? ( ( 1 + ncols ) * ncols / 2 ) : ( nrows * ncols );\
	__Read_Array_Head__\
		if (lower){\
			mat(irow, jcol)=mat(jcol, irow) = SafeStoD(word);\
			if ( irow == jcol ){\
				irow++;\
				jcol = 0;\
			}else jcol++;\
		}else{\
			mat(irow, jcol) = SafeStoD(word);\
			if ( jcol + 1 == ncols ){\
				irow++;\
				jcol = 0;\
			}else jcol++;\
		}\
	__Read_Array_Tail__

EigenMatrix Mwfn::MatrixTransform(){ // Chinium orders basis functions in the order like P-1, P0, P+1 and D-2, D-1, D0, D+1, D+2, while .mwfn does like Px, Py, Pz and D0, D+1, D-1, D+2, D-2. This function is used to transform matrices between two forms. 
	std::map<int, EigenMatrix> SPDFGHI; // 0 1 2 3 4 5 6 -6 -5 -4 -3 -2 -1
	SPDFGHI[0] = EigenOne(1, 1);
	SPDFGHI[1] = EigenOne(3, 3);
	SPDFGHI[2] = EigenOne(6, 6);
	SPDFGHI[3] = EigenOne(10, 10);
	SPDFGHI[4] = EigenOne(15, 15);
	SPDFGHI[5] = EigenOne(21, 21);
	SPDFGHI[6] = EigenOne(28, 28);
	SPDFGHI[-1] = EigenZero(3, 3); SPDFGHI[-1] <<
		0, 1, 0,
		0, 0, 1,
		1, 0, 0;
	SPDFGHI[-2] = EigenZero(5, 5); SPDFGHI[-2] <<
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 1, 0, 0, 0,
		0, 0, 0, 0, 1,
		1, 0, 0, 0, 0;
	SPDFGHI[-3] = EigenZero(7, 7); SPDFGHI[-3] <<
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1,
		1, 0, 0, 0, 0, 0, 0;
	SPDFGHI[-4] = EigenZero(9, 9); SPDFGHI[-4] <<
		0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1,
		1, 0, 0, 0, 0, 0, 0, 0, 0;
	SPDFGHI[-5] = EigenZero(11, 11); SPDFGHI[-5] <<
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	SPDFGHI[-6] = EigenZero(13, 13); SPDFGHI[-6] <<
		0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	int nbasis = this->getNumBasis();
	EigenMatrix transform = EigenZero(nbasis, nbasis);
	int jbasis = 0;
	for ( MwfnCenter& center : this->Centers ) for ( MwfnShell& shell : center.Shells ){
		int l = shell.Type;
		int num = shell.getSize();
		transform.block(jbasis, jbasis, num, num) = SPDFGHI[l];
		jbasis += num;
	}
	return transform;
}

static double SafeStoD(std::string word){ // In FT theory, some occupation numbers can be of dimension of 1E-50 or less. These small values cannot be stored as doubles and are likely to cause "out_of_range" error.
	double value = 1919810;
	try{
		value = std::stod(word);
	}catch ( const std::out_of_range& e ){
		value = 0;
	}
	return value;
}

Mwfn::Mwfn(std::string mwfn_filename, const bool output){
	std::ifstream file(mwfn_filename.c_str());
	assert(file.good() && "Mwfn file does not exist!");
	if (output) std::printf("Reading existent Mwfn file %s ...\n", mwfn_filename.c_str());
	std::string line, word;
	int tmp_int = -114;
	int nbasis = -114;
	int nindbasis = -114;
	std::vector<MwfnShell> Shells = {};
	std::vector<int> Shell_centers={};
	EigenMatrix mwfntransform;
	while ( std::getline(file, line) ){
		std::stringstream ss(line);
		ss >> word;

		// Field 1
		if ( word.compare("Wfntype=") == 0 ){
			ss >> word;
			this->Wfntype = std::stoi(word);
		}else if ( word.compare("E_tot=") == 0 ){
			ss >> word;
			this->E_tot = SafeStoD(word);
		}else if ( word.compare("VT_ratio=") == 0 ){
			ss >> word;
			this->VT_ratio = SafeStoD(word);
		}

		// Field 2
		else if ( word.compare("Ncenter=") == 0 ){
			ss >> word;
			const int ncenters = std::stoi(word);
			this->Centers.resize(ncenters);
		}else if ( word.compare("$Centers") == 0 ){
			for ( MwfnCenter& center : this->Centers ){
				std::getline(file, line);
				std::stringstream ss(line);
				ss >> word;
				ss >> word;
				ss >> word;
				center.Index = std::stoi(word);
				ss >> word;
				center.Nuclear_charge = SafeStoD(word);
				ss >> word;
				center.Coordinates[0] = SafeStoD(word) * __angstrom2bohr__;
				ss >> word;
				center.Coordinates[1] = SafeStoD(word) * __angstrom2bohr__;
				ss >> word;
				center.Coordinates[2] = SafeStoD(word) * __angstrom2bohr__;
			}
		}

		// Field 3
		else if ( word.compare("Nbasis=") == 0 ){
			ss >> word;
			nbasis = std::stoi(word);
			if ( nindbasis != -114 ){
				__Construct_Orbitals__
			}
		}else if ( word.compare("Nindbasis=") == 0 ){
			ss >> word;
			nindbasis = std::stoi(word);
			if ( nbasis != -114 ){
				__Construct_Orbitals__
			}
		}else if ( word.compare("Nshell=") == 0 ){
			ss >> word;
			const int nshells = std::stoi(word);
			Shells.resize(nshells);
			Shell_centers.resize(nshells);
		}else if ( word.compare("$Shell") == 0 ){
			ss >> word;
			if ( word.compare("types") == 0 ){
				const int total = Shells.size();
				__Read_Array_Head__
					Shells[k].Type = std::stoi(word);
				__Read_Array_Tail__
			}else if ( word.compare("centers") == 0 ){
				const int total = Shells.size();
				__Read_Array_Head__
					Shell_centers[k] = std::stoi(word);
				__Read_Array_Tail__
			}else if ( word.compare("contraction") == 0 ){
				const int total = Shells.size();
				__Read_Array_Head__
					Shells[k].Exponents.resize(std::stoi(word));
					Shells[k].Coefficients.resize(std::stoi(word));
					Shells[k].NormalizedCoefficients.resize(std::stoi(word));
				__Read_Array_Tail__
			}
		}else if ( word.compare("$Primitive") == 0 ){
			const int total = Shells.size();
			__Read_Array2_Head__
				const int total2 = Shells[k].Exponents.size();
				Shells[k].Exponents[l] = SafeStoD(word);
			__Read_Array2_Tail__
		}else if ( word.compare("$Contraction") == 0 ){
			const int total = Shells.size();
			__Read_Array2_Head__
				const int total2 = Shells[k].Coefficients.size();
				Shells[k].Coefficients[l] = SafeStoD(word);
			__Read_Array2_Tail__
		}

		// Field 4
		else if ( word.compare("Index=") == 0 ){
			if ( mwfntransform.cols() == 0 ){ // This indicates the end of Field 5.
				for ( int ishell = 0; ishell < (int)Shells.size(); ishell++ ){
					const int jcenter = Shell_centers[ishell] - 1;
					this->Centers[jcenter].Shells.push_back(std::move(Shells[ishell]));
				}
				mwfntransform = this->MatrixTransform();
			}
			ss >> word;
			tmp_int = std::stoi(word) - 1;
		}else if ( word.compare("Type=") == 0 ){
			ss >> word;
			this->Orbitals[tmp_int].Type = std::stoi(word);
		}else if ( word.compare("Energy=") == 0 ){
			ss >> word;
			this->Orbitals[tmp_int].Energy = SafeStoD(word);
		}else if ( word.compare("Occ=") == 0 ){
			ss >> word;
			this->Orbitals[tmp_int].Occ = SafeStoD(word);
		}else if ( word.compare("Sym=") == 0 ){
			ss >> word;
			this->Orbitals[tmp_int].Sym = word;
		}else if ( word.compare("$Coeff") == 0){
			const int total = this->Orbitals[tmp_int].Coeff.size();
			__Read_Array_Head__
				this->Orbitals[tmp_int].Coeff(k) = SafeStoD(word);
			__Read_Array_Tail__
		}

		// Field 5
		/*
		else if ( word.compare("$Total") == 0 ){
			ss >> word; // "density"
			ss >> word; // "matrix,"
			ss >> word; // "dim="
			ss >> word; // nrows
			const int nrows = std::stoi(word);
			ss >> word; // ncols
			const int ncols = std::stoi(word);
			ss >> word; // "lower="
			ss >> word; // lower
			const int lower = std::stoi(word);
			this->Total_density_matrix = EigenZero(nrows, ncols);
			__Load_Matrix__(this->Total_density_matrix)
			this->Total_density_matrix = mwfntransform.transpose() * this->Total_density_matrix * mwfntransform;
			continue;
		}else if ( word.compare("$Overlap") == 0 ){
			ss >> word; // "matrix,"
			ss >> word; // "dim="
			ss >> word; // nrows
			const int nrows = std::stoi(word);
			ss >> word; // ncols
			const int ncols = std::stoi(word);
			ss >> word; // "lower="
			ss >> word; // lower
			const int lower = std::stoi(word);
			this->Overlap_matrix = EigenZero(nrows, ncols);
			__Load_Matrix__(this->Overlap_matrix)
			this->Overlap_matrix = mwfntransform.transpose() * this->Overlap_matrix * mwfntransform;
			continue;
		}
		*/
	}
	for ( int spin : ( this->Wfntype == 0 ? std::vector<int>{0} : std::vector<int>{1, 2} ) ){
		const EigenMatrix tmp_mat = mwfntransform.transpose() * this->getCoefficientMatrix(spin);
		this->setCoefficientMatrix(tmp_mat, spin);
	}
}

void PrintMatrix(std::FILE * file, EigenMatrix matrix, bool lower){
	for ( int i = 0; i < matrix.rows(); i++ ){
		for ( int j = 0; j < ( lower ? i+1 : matrix.cols() ); j++ )
			std::fprintf(file, " %E", matrix(i, j));
		std::fprintf(file, "\n");
	}
}

#define __Check_Spin_Type_Shift__\
	if ( this->Wfntype == 0 ) assert( spin == 0 && "Invalid spin type!" );\
	if ( this->Wfntype == 1 ) assert( ( spin == 1 || spin == 2 ) && "Invalid spin type!" );\
	const int shift = ( this->Wfntype == 0 ? 0 : spin - 1 ) * this->getNumIndBasis();

double Mwfn::getNumElec(int spin){ // Total number of electrons if spin == -1 .
	double nelec = 0;
	if ( spin == -1 ){
		for ( int i = 0; i < (int)this->Orbitals.size(); i++ )
			nelec += this->Orbitals[i].Occ;
	}else{
		__Check_Spin_Type_Shift__
		for ( int i = 0; i < this->getNumIndBasis(); i++ )
			nelec += this->Orbitals[i + shift].Occ;
	}
	return nelec;
}

double Mwfn::getCharge(){
	double nuclear_charge = 0;
	for ( MwfnCenter& center : this->Centers )
		nuclear_charge += center.Nuclear_charge;
	double nelec = this->getNumElec(-1);
	return nuclear_charge - nelec;
}

int Mwfn::getNumCenters(){
	return this->Centers.size();
}

int Mwfn::getNumBasis(){
	int nbasis = 0;
	for ( MwfnCenter& center : this->Centers ) for ( MwfnShell& shell : center.Shells )
		nbasis += shell.getSize();
	return nbasis;
}

int Mwfn::getNumIndBasis(){
	return this->Orbitals.size() / ( this->Wfntype == 0 ? 1 : 2);
}

int Mwfn::getNumPrims(){
	int nprims = 0;
	for ( MwfnCenter& center : this->Centers ) for ( MwfnShell& shell : center.Shells ){
		int l = std::abs(shell.Type);
		nprims += ( l + 1 ) * ( l + 2 ) / 2 * shell.getNumPrims();
	}
	return nprims;
}

int Mwfn::getNumShells(){
	int nshells = 0;
	for ( MwfnCenter& center : this->Centers )
		nshells += center.Shells.size();
	return nshells;
}

int Mwfn::getNumPrimShells(){
	int nprimshells = 0;
	for ( MwfnCenter& center : this->Centers ) for ( MwfnShell& shell : center.Shells )
		nprimshells += shell.getNumPrims();
	return nprimshells;
}

EigenMatrix Mwfn::getCoefficientMatrix(int spin){
	__Check_Spin_Type_Shift__
	EigenMatrix matrix = EigenZero(this->getNumBasis(), this->getNumIndBasis());
	for ( int jcol = 0; jcol < this->getNumIndBasis(); jcol++ )
		matrix.col(jcol) = this->Orbitals[jcol + shift].Coeff;
	return matrix;
}

void Mwfn::setCoefficientMatrix(EigenMatrix matrix, int spin){
	__Check_Spin_Type_Shift__
	for ( int jcol = 0; jcol < this->getNumIndBasis(); jcol++ )
		this->Orbitals[jcol + shift].Coeff = matrix.col(jcol);
}

EigenVector Mwfn::getEnergy(int spin){
	__Check_Spin_Type_Shift__
	EigenVector energies(this->getNumIndBasis());
	for ( int iorbital = 0; iorbital < this->getNumIndBasis(); iorbital++ )
		energies(iorbital) = this->Orbitals[iorbital + shift].Energy;
	return energies;
}

void Mwfn::setEnergy(EigenVector energies, int spin){
	__Check_Spin_Type_Shift__
	for ( int iorbital = 0; iorbital < this->getNumIndBasis(); iorbital++ )
		this->Orbitals[iorbital + shift].Energy = energies(iorbital);
}

EigenVector Mwfn::getOccupation(int spin){
	__Check_Spin_Type_Shift__
	EigenVector occupancies(this->getNumIndBasis());
	for ( int iorbital = 0; iorbital < this->getNumIndBasis(); iorbital++ )
		occupancies(iorbital) = this->Orbitals[iorbital + shift].Occ;
	return occupancies;
}

void Mwfn::setOccupation(EigenVector occupancies, int spin){
	__Check_Spin_Type_Shift__
	for ( int iorbital = 0; iorbital < std::min((int)occupancies.size(), this->getNumIndBasis()); iorbital++ )
		this->Orbitals[iorbital + shift].Occ = occupancies(iorbital);
}

EigenMatrix Mwfn::getFock(int spin){
	const EigenMatrix S = this->Overlap;
	const EigenDiagonal E = this->getEnergy(spin).asDiagonal();
	const EigenMatrix C = this->getCoefficientMatrix(spin);
	return S * C * E * C.transpose() * S;
}

EigenMatrix Mwfn::getDensity(int spin){
	const EigenDiagonal N = this->getOccupation(spin).asDiagonal();
	const EigenMatrix C = this->getCoefficientMatrix(spin);
	return C * N * C.transpose();
}

EigenMatrix Mwfn::getEnergyDensity(int spin){
	const EigenDiagonal N = this->getOccupation(spin).asDiagonal();
	const EigenDiagonal E = this->getEnergy(spin).asDiagonal();
	const EigenMatrix C = this->getCoefficientMatrix(spin);
	return C * N * E * C.transpose();
}

std::vector<int> Mwfn::Basis2Atom(){
	std::vector<int> bf2atom = {}; bf2atom.reserve(this->getNumBasis());
	for ( int icenter = 0; icenter < this->getNumCenters(); icenter++ ){
		for ( int jbasis = 0; jbasis < this->Centers[icenter].getNumBasis(); jbasis++ ){
			bf2atom.push_back(icenter);
		}
	}
	return bf2atom;
}

std::vector<int> Mwfn::Atom2Basis(){
	std::vector<int> atom2bf = {}; atom2bf.reserve(this->getNumCenters());
	int ibasis = 0;
	for ( MwfnCenter& center : this->Centers ){
		atom2bf.push_back(ibasis);
		ibasis += center.getNumBasis();
	}
	return atom2bf;
}
void Mwfn::Export(std::string mwfn_filename, const bool output){
	std::FILE* file = std::fopen(mwfn_filename.c_str(), "w");
	if (output) std::printf("Exporting wavefunction information to %s ...\n", mwfn_filename.c_str());
	std::fprintf(file, "# Generated by Chinium\n");

	// Field 1
	std::fprintf(file, "\n\n# Overview\n");
	std::fprintf(file, "Wfntype= %d\n", this->Wfntype);
	std::fprintf(file, "Charge= %f\n", this->getCharge());
	if ( this->Wfntype == 0 ){
		std::fprintf(file, "Naelec= %f\n", this->getNumElec() / 2);
		std::fprintf(file, "Nbelec= %f\n", this->getNumElec() / 2);
	}else{
		std::fprintf(file, "Naelec= %f\n", this->getNumElec(1));
		std::fprintf(file, "Nbelec= %f\n", this->getNumElec(2));
	}
	if ( this->E_tot != -114514 )
		std::fprintf(file, "E_tot= %f\n", this->E_tot);
	if ( this->VT_ratio != -114514 )
		std::fprintf(file, "VT_ratio= %f\n", this->VT_ratio);

	// Field 2
	std::fprintf(file, "\n\n# Atoms\n");
	std::fprintf(file, "Ncenter= %d\n", this->getNumCenters());
	std::fprintf(file, "$Centers\n");
	int icenter = 1;
	for ( MwfnCenter& center : this->Centers ) std::fprintf(
			file, "%d %s %d %f % f % f % f\n",
			icenter++,
			center.getSymbol().c_str(),
			center.Index,
			center.Nuclear_charge,
			center.Coordinates[0] / __angstrom2bohr__,
			center.Coordinates[1] / __angstrom2bohr__,
			center.Coordinates[2] / __angstrom2bohr__);

	// Field 3
	std::fprintf(file, "\n\n# Basis set\n");
	std::fprintf(file, "Nbasis= %d\n", this->getNumBasis());
	std::fprintf(file, "Nindbasis= %d\n", this->getNumIndBasis());
	std::fprintf(file, "Nprims= %d\n", this->getNumPrims());
	std::fprintf(file, "Nshell= %d\n", this->getNumShells());
	std::fprintf(file, "Nprimshell= %d\n", this->getNumPrimShells());
	std::fprintf(file, "$Shell types\n");
	for ( MwfnCenter& center : this->Centers ){
		for ( MwfnShell& shell : center.Shells ){
			std::fprintf(file, " %d", shell.Type); // Note that Chinium uses only pure spherical harmonics.
		}
		std::fprintf(file, "\n");
	}
	std::fprintf(file, "$Shell centers\n");
	for ( int jcenter = 0; jcenter < int(this->Centers.size()); jcenter++ ){
		for ( int kshell = 0; kshell < int(this->Centers[jcenter].Shells.size()); kshell++ )
			std::fprintf(file, " %d", jcenter + 1);
		std::fprintf(file, "\n");
	}
	std::fprintf(file, "$Shell contraction degrees\n");
	for ( MwfnCenter& center : this->Centers ){
		for ( MwfnShell& shell : center.Shells )
			std::fprintf(file, " %d", shell.getNumPrims());
		std::fprintf(file, "\n");
	}
	std::fprintf(file, "$Primitive exponents\n");
	for ( MwfnCenter& center : this->Centers ){
		for ( MwfnShell& shell : center.Shells )
			for ( double exponent : shell.Exponents )
				std::fprintf(file, " %E", exponent);
		std::fprintf(file, "\n");
	}
	std::fprintf(file, "$Contraction coefficients\n");
	for ( MwfnCenter& center : this->Centers ){
		for ( MwfnShell& shell : center.Shells )
			for ( double coefficient : shell.Coefficients )
				std::fprintf(file, " %E", coefficient);
		std::fprintf(file, "\n");
	}

	// Field 4
	std::fprintf(file, "\n\n# Orbitals\n");
	EigenMatrix mwfntransform = this->MatrixTransform();
	int iorbital = 0;
	for ( MwfnOrbital& orbital : this->Orbitals ){
		std::fprintf(file, "Index= %9d\n", iorbital + 1);
		std::fprintf(file, "Type= %d\n", orbital.Type);
		std::fprintf(file, "Energy= %.10E\n", orbital.Energy);
		std::fprintf(file, "Occ= %E\n", orbital.Occ);
		std::fprintf(file, "Sym= %s\n", orbital.Sym.c_str());
		std::fprintf(file, "$Coeff\n");
		for ( int j = 0; j < this->getNumBasis(); j++ ){
			const EigenVector c = mwfntransform * orbital.Coeff;
			std::fprintf(file, " %.10E", c(j));
		}
		std::fprintf(file, "\n\n");
		iorbital++;
	}

	// Field 5
	/*
	std::fprintf(file,"\n\n# Matrices\n");
	if (this->Total_density_matrix.cols()){
		std::fprintf(file,"$Total density matrix, dim= %d %d lower= 1\n",this->getNumBasis(),this->getNumBasis());
		PrintMatrix(file,mwfntransform*this->Total_density_matrix*mwfntransform.transpose(),1);
	}
	if (this->Hamiltonian_matrix.cols()){
		std::fprintf(file,"$1-e Hamiltonian matrix, dim= %d %d lower= 1\n",this->getNumBasis(),this->getNumBasis());
		PrintMatrix(file,mwfntransform*this->Hamiltonian_matrix*mwfntransform.transpose(),1);
	}
	if (this->Overlap_matrix.cols()){
		std::fprintf(file,"$Overlap matrix, dim= %d %d lower= 1\n",this->getNumBasis(),this->getNumBasis());
		PrintMatrix(file,mwfntransform*this->Overlap_matrix*mwfntransform.transpose(),1);
	}
	if (this->Kinetic_energy_matrix.cols()){
		std::fprintf(file,"$Kinetic energy matrix, dim= %d %d lower= 1\n",this->getNumBasis(),this->getNumBasis());
		PrintMatrix(file,mwfntransform*this->Kinetic_energy_matrix*mwfntransform.transpose(),1);
	}
	if (this->Potential_energy_matrix.cols()){
		std::fprintf(file,"$Potential energy matrix, dim= %d %d lower= 1\n",this->getNumBasis(),this->getNumBasis());
		PrintMatrix(file,mwfntransform*this->Potential_energy_matrix*mwfntransform.transpose(),1);
	}
	*/
	std::fclose(file);
}

std::vector<MwfnCenter> MwfnReadBasis(std::string basis_file_path_name){
	std::ifstream file( basis_file_path_name.c_str() );
	assert("Basis set file is missing in $CHINIUM_PATH/BasisSets/ !" && file.good());
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
			std::transform(word.begin(), word.end(), word.begin(), ::toupper);
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
			}else assert(0 && "Unrecognized symbol of angular momentum!");
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
				shell.NormalizedCoefficients.push_back(0);
				if ( shell2.Type != -114 ){
					ss >> word; word = std::regex_replace(word, re, "E");
					shell2.Coefficients.push_back(std::stod(word));
					shell2.NormalizedCoefficients.push_back(0);
				}
			}
			center.Shells.push_back(shell);
			if ( shell2.Type != -114 ){
				center.Shells.push_back(shell2);
			}
			shell.Type = -114;
			shell.Exponents.resize(0);
			shell.Coefficients.resize(0);
			shell.NormalizedCoefficients.resize(0);
			shell2.Type = -114;
			shell2.Exponents.resize(0);
			shell2.Coefficients.resize(0);
			shell2.NormalizedCoefficients.resize(0);
		}else if ( word == "****" ){
			centers.push_back(center);
			center.Shells.resize(0);
		}
	}
	return centers;
}

void Mwfn::setBasis(std::string basis_filename, const bool output){
	std::string basis_file_path_name = (std::string)std::getenv("CHINIUM_PATH") + "/BasisSets/" + basis_filename + ".gbs";
	if (output) std::printf("Reading basis set file %s ...\n", basis_file_path_name.c_str());
	std::vector<MwfnCenter> bare_centers = MwfnReadBasis(basis_file_path_name);
	for ( MwfnCenter& mwfn_center : this->Centers ){
		bool found = 0;
		for ( MwfnCenter& bare_center : bare_centers ){
			if ( mwfn_center.Index == bare_center.Index ){
				mwfn_center.Shells = bare_center.Shells;
				found = 1;
				break;
			}
		}
		assert(found && "Basis set file does not include this element!");
	}
}

void Mwfn::setCenters(std::vector<std::vector<double>> atoms, const bool output){
	if (output) std::printf("Reading atomic coordinates from input file ...\n");
	this->Centers.clear();
	this->Centers.reserve(atoms.size());
	for ( std::vector<double>& atom : atoms ){
		MwfnCenter center;
		center.Index = std::round(atom[0]);
		center.Nuclear_charge = atom[1];
		center.Coordinates = {atom[2], atom[3], atom[4]};
		this->Centers.push_back(center);
	}
}

void Mwfn::PrintCenters(){
	std::printf("Atoms:\n");
	std::printf("| Number | Symbol | Index | Charge |  X (Bohr)  |  Y (Bohr)  |  Z (Bohr)  |\n");
	for ( int icenter = 0; icenter < this->getNumCenters(); icenter++ ){
		MwfnCenter& center = this->Centers[icenter];
		std::printf(
				"| %6d | %6s | %5d | %6.2f | % 10.5f | % 10.5f | % 10.5f |\n",
				icenter,
				center.getSymbol().c_str(),
				center.Index,
				center.Nuclear_charge,
				center.Coordinates[0],
				center.Coordinates[1],
				center.Coordinates[2]
		);
	}
}

void Mwfn::PrintOrbitals(){
	std::printf("Orbitals:\n");
	if ( this->Wfntype == 0 ){
		std::printf("| Number | Energy (eV) | Occupation |\n");
		for ( int iorbital = 0; iorbital < this->getNumIndBasis(); iorbital++){
			MwfnOrbital& orbital = this->Orbitals[iorbital];
			std::printf(
					"| %6d | % 11.4f | %10.8f |\n",
					iorbital,
					orbital.Energy * __hartree2ev__,
					orbital.Occ
			);
		}
	}else if ( this->Wfntype == 1 ){
		std::printf("| Number | Energy (eV) | Occupation | Number | Energy (eV) | Occupation |\n");
		for ( int iorbital = 0; iorbital < this->getNumIndBasis(); iorbital++ ){
			MwfnOrbital& orbitala = this->Orbitals[iorbital];
			MwfnOrbital& orbitalb = this->Orbitals[iorbital + this->getNumIndBasis()];
			std::printf(
					"| %6d | % 11.4f | %10.8f | %6d | % 11.4f | %10.8f |\n",
					iorbital,
					orbitala.Energy * __hartree2ev__,
					orbitala.Occ,
					iorbital + this->getNumIndBasis(),
					orbitalb.Energy * __hartree2ev__,
					orbitalb.Occ
			);
		}
	}
}
