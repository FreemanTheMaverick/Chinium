#include <vector>
#include <string>

#include "../RestrictedOpen.h"

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

RO_SCF::RO_SCF(std::string inp): Job(inp), RepRO(inp), SCF(inp, mwfn, int2c1e){
	if ( Np == Na && Np == Nb ) xc.Spin = 1;
	else xc.Spin = 2;
	Na -= Np;
	Nb -= Np;

	if ( Na > 0 && Nb > 0 ) Coupling = getCouplingCoefficient({ Na, Nb })[1][0];
}
