EigenMatrix getRepulsionDiag(libint2::BasisSet& obs); // Computing the diagonal elements of electron repulsion tensor for Cauchy-Schwarz screening.

std::tuple<long int, long int> getRepulsionLength(libint2::BasisSet& obs, EigenMatrix repulsiondiag, double threshold); // Numbers of nonequivalent two-electron integrals and shell quartets after Cauchy-Schwarz screening.

void getRepulsionIndices(
		libint2::BasisSet& obs, EigenMatrix repulsiondiag,
		double threshold,
		short int* shellis, short int* shelljs,
		short int* shellks, short int* shellls);

std::tuple<std::vector<long int>, std::vector<long int>> getThreadPointers(
		libint2::BasisSet& obs, long int nshellquartets, int nthreads,
		short int* shellis, short int* shelljs,
		short int* shellks, short int* shellls);

void getRepulsion0(
		libint2::BasisSet& obs,
		std::vector<long int> sqheads, std::vector<long int> bqheads,
		short int* shellis, short int* shelljs,
		short int* shellks, short int* shellls,
		short int* basisis, short int* basisjs,
		short int* basisks, short int* basisls,
		char* degs, double* repulsions);
