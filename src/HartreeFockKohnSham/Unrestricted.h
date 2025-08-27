std::tuple<double, EigenVector, EigenVector, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedDIIS(
		double T, double Mu,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Fa, EigenMatrix Fb,
		EigenVector Occa, EigenVector Occb,
		EigenMatrix Za, EigenMatrix Zb,
		int output, int nthreads
);

std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedLBFGS(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix D1prime, EigenMatrix D2prime,
		EigenMatrix Z1, EigenMatrix Z2,
		int output, int nthreads
);

std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedNewton(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix D1prime, EigenMatrix D2prime,
		EigenMatrix Z1, EigenMatrix Z2,
		int output, int nthreads
);

std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedARH(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix D1prime, EigenMatrix D2prime,
		EigenMatrix Z1, EigenMatrix Z2,
		int output, int nthreads
);

std::tuple<double, EigenVector, EigenVector, EigenMatrix, EigenMatrix> UnrestrictedRiemannARH_villain(
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix D1prime, EigenMatrix D2prime,
		EigenMatrix Z1, EigenMatrix Z2,
		int output, int nthreads
);
