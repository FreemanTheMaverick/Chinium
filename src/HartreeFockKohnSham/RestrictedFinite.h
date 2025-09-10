std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix F, EigenVector Occ, EigenMatrix Z,
		int output, int nthreads);

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteLBFGS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenVector occ_guess, EigenMatrix Z,
		int output, int nthreads);

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteNewton(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenVector occ_guess, EigenMatrix Z,
		int output, int nthreads);

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteARH(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenVector occ_guess, EigenMatrix Z,
		int output, int nthreads);
