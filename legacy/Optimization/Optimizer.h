bool Mouse(
		std::function<std::tuple<double, EigenMatrix, EigenMatrix>
		(EigenMatrix, int)>& func,
		std::tuple<double, double, double> tol,
		int diis_space, int max_iter,
		double& Y, EigenMatrix& X, bool output);
