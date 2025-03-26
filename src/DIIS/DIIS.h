#pragma once

class DIIS{ public:
	std::string Name;
	bool Verbose;
	int MaxSize;
	double Tolerance;
	int MaxIter;
	std::vector<std::deque<EigenMatrix>> Updatess;
	std::vector<std::deque<EigenMatrix>> Residualss;
	std::vector<std::deque<EigenMatrix>> Auxiliariess;
	std::function<
		std::tuple<
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
		> (std::vector<EigenMatrix>&, std::vector<bool>&)
	>* UpdateFunc;

	DIIS(
		std::function<std::tuple<
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>,
			std::vector<EigenMatrix>
			> (std::vector<EigenMatrix>&, std::vector<bool>&)
		>* update_func,
		int nmatrices, int max_size, double tolerance,
		int max_iter, bool verbose
	);
	virtual ~DIIS() = default;

	int getNumMatrices();
	int getCurrentSize();
	void Steal(DIIS& another_diis);
	void Append(std::vector<EigenMatrix>& updates, std::vector<EigenMatrix>& residuals, std::vector<EigenMatrix>& auxiliaries);
	bool Run(std::vector<EigenMatrix>& Ms);

	virtual EigenVector Extrapolate(int index) = 0;
};
