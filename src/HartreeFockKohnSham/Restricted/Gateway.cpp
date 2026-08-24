#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdio>

#include "../Restricted.h"

inline double A(double b, double x, double y){
	return std::sqrt( ( b + x ) / ( b + y ) );
}

inline double F(double b, double d){
	if ( d == 0 || d == 3 ) return 1;
	else if ( d == 1 ) return A(b, 2, 0) * A(b, -1, 1);
	else if ( d == 2 ) return A(b, 0, 2) * A(b, 3, 1);
	else assert(0 && "Why am I here?");
}

double Eijji(std::vector<int> b, std::vector<int> d, int i, int j){
	if ( i == j ) return -1;
	if ( i > j ) std::swap(i, j);

	if ( d[i] == 0 || d[j] == 0 ) return 0;
	else if ( d[i] == 3 ){
		if ( d[j] == 1 || d[j] == 2 ) return -1;
		else return -2;
	}else if ( d[j] == 3 ){
		if ( d[i] == 1 || d[i] == 2 ) return -1;
		else return -2;
	}else{
		double prod = d[i] == 1 ? A( b[i], 2, 0 ) : A( b[i], 0, 2 );
		prod *= d[j] == 1 ? A( b[j], -1, 1 ) : A( b[j], 3, 1 );
		for ( int k = i + 1; k < j; k++ ) prod *= F(b[k], d[k]);
		const double phase = d[i] == d[j] ? 1 : -1;
		return - 0.5 * ( 1 + phase * prod );
	}
}

std::vector<std::vector<double>> getCouplingCoefficient(std::vector<int> shell_sizes){
	const int nshells = (int)shell_sizes.size();
	int sign = 1;
	std::vector<int> bvec;
	for ( int ishell = 0; ishell < nshells; ishell++ ){
		for ( int shell = 0; shell < shell_sizes[ishell]; shell++ ){
			bvec.push_back( sign );
		}
		sign *= -1;
	}
	const int nact = (int)bvec.size();
	std::vector<int> dist;
	for ( int iorb = 0; iorb < nact; iorb++ ){
		dist.push_back( bvec[iorb] == 1 ? 1 : 2 );
	}
	for ( int iorb = 1; iorb < nact; iorb++ ){
		bvec[iorb] += bvec[iorb - 1];
	}
	std::vector<std::vector<double>> b(nshells, std::vector<double>(nshells));
	for ( int ishell = 0, iorb = 0; ishell < nshells; iorb += shell_sizes[ishell++] ) for ( int jshell = 0, jorb = 0; jshell < nshells; jorb += shell_sizes[jshell++] )
		b[ishell][jshell] = Eijji(bvec, dist, iorb, jorb);
	return b;
}

std::vector<std::array<EigenMatrix, 2>> ReadLower(std::string inp){
	std::vector<std::array<EigenMatrix, 2>> lowers;
	std::ifstream file(inp);
	std::string thisline;
	bool found = 0;
	int nlowers = 0;
	while ( std::getline(file, thisline) && ! found ){
		std::transform(thisline.begin(), thisline.end(), thisline.begin(), ::toupper);
		if ( thisline == "LOWER" ){
			found = 1;
			std::getline(file, thisline);
			std::stringstream ss(thisline);
			ss >> nlowers;
			if ( nlowers < 0 ) throw std::runtime_error("Invalid number of lower states!");
			lowers.resize(nlowers);
			for ( int ilower = 0; ilower < nlowers; ilower++ ){
				std::getline(file, thisline);
				std::stringstream ss(thisline);
				std::string mwfn_str;
				ss >> mwfn_str;
				libmwfn::Mwfn mwfn(mwfn_str);
				if ( mwfn.Wfntype != 0 && mwfn.Wfntype != 2 ) throw std::runtime_error("Orthogonality-constrained SCF only supports spin-restricted lower states!");
				const EigenMatrix Cd = mwfn.getCoefficientMatrix({.Set=0, .Type=0, .OccUpper=2, .OccLower=2});
				const EigenMatrix Ca = mwfn.getCoefficientMatrix({.Set=0, .Type=1, .OccUpper=1, .OccLower=1});
				const EigenMatrix Cb = mwfn.getCoefficientMatrix({.Set=0, .Type=2, .OccUpper=1, .OccLower=1});
				EigenMatrix Ca_ = EigenZero(mwfn.getNumBasis(), Cd.cols() + Ca.cols());
				EigenMatrix Cb_ = EigenZero(mwfn.getNumBasis(), Cd.cols() + Cb.cols());
				Ca_ << Cd, Ca;
				Cb_ << Cd, Cb;
				lowers[ilower] = { Ca_, Cb_ };
			}
		}
	}
	return lowers;
}

R_SCF::R_SCF(std::string inp): Job(inp), RepR(inp), SCF(inp, mwfn, int2c1e){
	if ( Na + Nb == 0 ) xc.Spin = 1;
	else xc.Spin = 2;

	if ( Na > 0 && Nb > 0 ) Coupling = getCouplingCoefficient({ Na, Nb })[1][0];
	std::printf("Restricted open-shell spin-coupling factor: %f\n", Coupling);

	if ( scftype == "DIIS" && ( Na > 0 || Nb > 0 ) ) throw std::runtime_error("DIIS for RO-SCF is not implemented yet!");

	lowers = ReadLower(inp);
	lowers_type.resize(lowers.size());
	if ( scftype == "DIIS" && lowers.size() > 0 ) throw std::runtime_error("DIIS cannot be used for orthogonality-constrained SCF!");
	EigenMatrix Z = EigenZero(mwfn.getNumBasis(), mwfn.getNumIndBasis());
	Z <<
		mwfn.getCoefficientMatrix({.Set=0, .OccUpper=2, .OccLower=2}),
		mwfn.getCoefficientMatrix({.Set=0, .Type=1, .OccUpper=1, .OccLower=1}),
		mwfn.getCoefficientMatrix({.Set=0, .Type=2, .OccUpper=1, .OccLower=1}),
		mwfn.getCoefficientMatrix({.Set=0, .OccUpper=0, .OccLower=0})
	;
	const EigenMatrix Zinv = Z.inverse();
	for ( int ilower = 0; ilower < (int)lowers.size(); ilower++ ){
		std::array<EigenMatrix, 2>& lower = lowers[ilower];
		int& type = lowers_type[ilower] = 1;
		lower[0] = Zinv * lower[0];
		lower[1] = Zinv * lower[1];
		if ( lower[0].cols() != lower[1].cols() ) type = 1;
		else if ( std::abs( ( lower[0].transpose() * lower[1] ).determinant() ) > 1. - 1e-10 ) type = 0;
		else type = 2;
	}
}
