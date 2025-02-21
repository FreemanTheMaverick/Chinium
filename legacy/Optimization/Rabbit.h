bool Rabbit(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<
					EigenMatrix
					(
						EigenMatrix,
						std::vector<EigenMatrix>
					)
				>
			>
			(
				EigenMatrix
			)
		>& func,
		std::tuple<double, double, double> tol,
		int max_iter,
		double& L, EigenMatrix& U, bool output);
