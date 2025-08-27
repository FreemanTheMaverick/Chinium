std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenLBFGS(
		int nd, int ns,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads);

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenNewton(
		int nd, int ns,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads);

std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenARH(
		int nd, int ns,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads);
