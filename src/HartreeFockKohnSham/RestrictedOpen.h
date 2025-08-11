std::tuple<double, EigenVector, EigenMatrix> RestrictedOpenRiemann(
		int nd, int ns,
		Int2C1E& int2c1e, Int4C2E& int4c2e,
		EigenMatrix Cprime, EigenMatrix Z,
		int output, int nthreads);
