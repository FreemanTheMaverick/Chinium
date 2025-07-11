#include "DIIS.h"

class EDIIS: public DIIS{ public:
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
	EigenVector Extrapolate(int index) override;
};
