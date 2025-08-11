std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix F, EigenVector Occ, EigenMatrix Z,
		int output, int nthreads
);

std::tuple<double, EigenVector, EigenMatrix> RestrictedRiemann(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Dprime, EigenMatrix Z,
		int output, int nthreads
);

std::tuple<double, EigenVector, EigenMatrix> RestrictedRiemannARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Dprime, EigenMatrix Z,
		int output, int nthreads
);

std::tuple<double, EigenVector, EigenVector, EigenMatrix> RestrictedFock(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		ExchangeCorrelation& xc, Grid& grid,
		EigenMatrix Fprime, EigenVector Occ, EigenMatrix Z,
		int output, int nthreads
);
