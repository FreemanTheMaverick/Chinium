std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteLoopDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix F, EigenVector Occ, EigenMatrix Z,
		int output, int nthreads);

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix F, EigenVector Occ, EigenMatrix Z,
		int output, int nthreads);

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteLBFGS(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		double T, double Mu,
		EigenVector all_occ, EigenMatrix Z,
		int nthreads, int output
);

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteNewton(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		double T, double Mu,
		EigenVector all_occ, EigenMatrix Z,
		int nthreads, int output
);

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFiniteARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		double T, double Mu,
		EigenVector all_occ, EigenMatrix Z,
		int nthreads, int output
);
