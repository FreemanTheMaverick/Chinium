EigenMatrix GrassmannExp(EigenMatrix p, EigenMatrix X);

EigenMatrix GrassmannGrad(EigenMatrix X, EigenMatrix Ge);

bool Tiger(
		std::function<std::tuple<double, EigenMatrix> (EigenMatrix)>& func,
		std::tuple<double, double, double> tol,
		int diis_space, int max_iter,
		double& L, EigenMatrix& X, bool output);
