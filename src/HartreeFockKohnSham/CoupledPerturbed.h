std::tuple<
	std::vector<EigenMatrix>,
	std::vector<EigenMatrix>,
	std::vector<EigenMatrix>,
	std::vector<EigenMatrix>
> NonIdempotent(
		EigenMatrix C, EigenVector es, EigenVector ns,
		std::vector<EigenMatrix>& Ss,
		std::vector<EigenMatrix>& Fskeletons,
		short int* is, short int* js, short int* ks, short int* ls,
		char* degs, double* ints, long int length,
		double kscale,
		std::vector<int> orders,
		double* ws,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2xxs, double* ao2yys, double* ao2zzs,
		double* ao2xys, double* ao2xzs, double* ao2yzs,
		double* d1xs, double* d1ys, double* d1zs,
		double* vrs, double* vss,
		double* vrrs, double* vrss, double* vsss,
		long int ngrids,
		int output, int nthreads);
