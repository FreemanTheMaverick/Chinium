#pragma once

#include <Eigen/Core>
#include <functional>
#include <vector>
#include <tuple>

#include "../Macro.h"

#include "DIIS.h"

class AEDIIS: public DIIS{ public:
	using DIIS::DIIS;
	virtual std::tuple<EigenMatrix, EigenMatrix> MakeAB(int index) = 0;
	EigenVector Extrapolate(int index) override;
};

class EDIIS: public AEDIIS{ public:
	EDIIS(
		std::function<
			std::tuple<
				std::vector<EigenMatrix>,
				std::vector<EigenMatrix>,
				std::vector<EigenMatrix>
			> (std::vector<EigenMatrix>&, std::vector<bool>&)
		>* update_func,
		int nmatrices, int max_size, double tolerance,
		int max_iter, int verbose
	);
	std::tuple<EigenMatrix, EigenMatrix> MakeAB(int index) override;
};

class ADIIS: public AEDIIS{ public:
	ADIIS(
		std::function<
			std::tuple<
				std::vector<EigenMatrix>,
				std::vector<EigenMatrix>,
				std::vector<EigenMatrix>
			> (std::vector<EigenMatrix>&, std::vector<bool>&)
		>* update_func,
		int nmatrices, int max_size, double tolerance,
		int max_iter, int verbose
	);
	std::tuple<EigenMatrix, EigenMatrix> MakeAB(int index) override;
};
