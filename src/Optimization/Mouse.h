bool Mouse(
		std::function<std::tuple<double, EigenMatrix, EigenMatrix, EigenMatrix> (EigenMatrix)>& func,
		std::tuple<double, double, double> adtol,
		std::tuple<double, double, double> tol,
		int diis_space, int max_iter,
		double& L, EigenMatrix& X, int output);
