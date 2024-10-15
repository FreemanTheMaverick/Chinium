bool Ox(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<EigenMatrix (EigenMatrix)>
			> (EigenMatrix)
		>& func,
		std::tuple<double, double, double> tol,
		int max_iter,
		double& L, EigenMatrix& C, bool output);
