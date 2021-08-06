#include <libint2.hpp>

using namespace libint2;

int nBasis(BasisSet obs){
	int n=0;
	for (const auto& shell:obs){
		n+=shell.size();
	}
	return n;
}
