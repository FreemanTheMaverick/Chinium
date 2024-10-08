bool Loong(
		std::function<
			EigenMatrix
			(
				EigenMatrix,
				std::vector<EigenMatrix>
			)
		>& newton_update, std::vector<EigenMatrix> params,
		double tol, int diis_space, int maxiter, EigenMatrix& D, bool output);
