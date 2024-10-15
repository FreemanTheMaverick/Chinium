EigenMatrix Loong(
		std::function<double (EigenMatrix, EigenMatrix)>& Inner,
		std::function<EigenMatrix (EigenMatrix)>& Hess,
		EigenMatrix b, double R, int ndim, bool output);


