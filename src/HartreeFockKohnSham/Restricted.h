std::tuple<double, EigenVector, EigenMatrix> RestrictedDIIS(
		int nocc,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix F, EigenMatrix Z,
		int output, int nthreads
);

std::tuple<double, EigenVector, EigenMatrix> RestrictedLBFGS(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc, EigenMatrix Z,
		int nthreads, int output
);

std::tuple<double, EigenVector, EigenMatrix> RestrictedNewton(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc, EigenMatrix Z,
		int nthreads, int output
);

std::tuple<double, EigenVector, EigenMatrix> RestrictedARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nocc, EigenMatrix Z,
		int nthreads, int output
);
