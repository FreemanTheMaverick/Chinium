#define __Make_Basis_Set__\
	std::vector<libint2::Shell> libint2shells = {};\
	for ( MwfnCenter& center : this->Centers ) for ( MwfnShell& shell : center.Shells ){\
		const int l = std::abs(shell.Type);\
		const bool pure = ( shell.Type < 0 );\
		libint2::svector<double> exponents = {};\
		libint2::svector<double> coefficients = {};\
		for ( int i = 0; i < shell.getNumPrims(); i++ ){\
			exponents.push_back(shell.Exponents[i]);\
			coefficients.push_back(shell.Coefficients[i]);\
		}\
		libint2shells.push_back(libint2::Shell(\
			exponents, {{l, pure, coefficients}}, {\
				center.Coordinates[0],\
				center.Coordinates[1],\
				center.Coordinates[2]\
			}, 1));\
	}\
	libint2::BasisSet obs(libint2shells);
