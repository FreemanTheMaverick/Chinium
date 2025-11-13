std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenLBFGS(
		int nd, int na, int nb,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads);

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenNewton(
		int nd, int na, int nb,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads);

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenARH(
		int nd, int na, int nb,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads);
