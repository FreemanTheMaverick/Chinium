	void GuessSCF(std::string guess);
	EigenMatrix calcFock(int nthreads);
	EigenMatrix calcFock(EigenMatrix D, int nthreads);
	void HartreeFockKohnSham(double temperature, double chemicalpotential, int output, int nthreads);
