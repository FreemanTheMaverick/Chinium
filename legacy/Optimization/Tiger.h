bool Tiger(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<EigenMatrix (EigenMatrix)>
			> (EigenMatrix)
		>& func,
		std::tuple<double, double, double> tol,
		int max_iter,
		double& L, Manifold& M, Manifold& M_last, int output);
