std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenLBFGS(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nd, int na, int nb, EigenMatrix Z,
		int nthreads, int output);

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenNewton(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nd, int na, int nb, EigenMatrix Z,
		int nthreads, int output);

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		int nd, int na, int nb, EigenMatrix Z,
		int nthreads, int output);
