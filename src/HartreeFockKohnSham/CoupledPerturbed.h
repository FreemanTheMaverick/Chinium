std::tuple<std::vector<EigenMatrix>,std::vector<EigenMatrix>,std::vector<EigenMatrix>,std::vector<EigenMatrix>> NonIdempotent(
		EigenMatrix C, EigenVector es, EigenVector ns,
		std::vector<EigenMatrix>& Ss,
		std::vector<EigenMatrix>& Fskeletons,
		short int* is, short int* js, short int* ks, short int* ls,
		char* degs, double* ints, long int length,
		double kscale, int output, int nthreads);
