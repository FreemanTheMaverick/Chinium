#define __Make_Basis_Set__(mwfn)\
	std::vector<libint2::Shell> libint2shells = {};\
	std::vector<int> shell2atom = {};\
	int iatom = 0;\
	for ( MwfnCenter& center : mwfn->Centers ){\
	   	for ( MwfnShell& shell : center.Shells ){\
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
			shell2atom.push_back(iatom);\
		}\
		iatom++;\
	}\
	libint2::BasisSet obs(libint2shells);
