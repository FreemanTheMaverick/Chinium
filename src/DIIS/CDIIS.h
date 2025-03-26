#include "DIIS.h"

class CDIIS: public DIIS{ public:
	CDIIS(
		std::function<
			std::tuple<
				std::vector<EigenMatrix>,
				std::vector<EigenMatrix>,
				std::vector<EigenMatrix>
			> (std::vector<EigenMatrix>&)
		>* update_func,
		int nmatrices, int max_size, double tolerance,
		int max_iter, bool verbose
	);
	EigenVector Extrapolate(int index) override;
};
