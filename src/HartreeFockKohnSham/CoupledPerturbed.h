std::tuple<
	std::vector<EigenMatrix>,
	std::vector<EigenMatrix>,
	std::vector<EigenVector>,
	std::vector<EigenMatrix>,
	std::vector<EigenMatrix>
> NonIdempotent(
		EigenMatrix C, EigenVector es, EigenVector ns,
		std::vector<EigenMatrix>& Ss,
		std::vector<EigenMatrix>& Fskeletons,
		Int4C2E& int4c2e, Grid& grid,
		int output, int nthreads);

std::vector<EigenVector> OccupationGradient(
		EigenMatrix C, EigenVector es,
		std::map<int, EigenMatrix> Dns,
		EigenVector Nes,
		std::vector<EigenMatrix>& Ss,
		std::vector<EigenMatrix>& Fskeletons,
		Int4C2E& int4c2e, Grid& grid,
		int output, int nthreads);

std::map<int, EigenMatrix> OccupationFluctuation(
		EigenMatrix C, EigenVector es, EigenVector ns,
		std::vector<int> frac_indeces,
		Int4C2E& int4c2e, Grid& grid,
		int output, int nthreads);
