bool Ox(std::function<std::tuple<double, EigenVector, EigenMatrix> (EigenVector)>& func,
		std::tuple<double, double, double> tol,
		int max_iter,
		double& L, EigenVector& C, bool output);
