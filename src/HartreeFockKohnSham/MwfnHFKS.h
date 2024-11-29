	void GuessSCF(std::string guess);
	std::tuple<EigenMatrix, EigenMatrix, double> calcFock(EigenMatrix D, int nthreads);
	void HartreeFockKohnSham(double temperature, double chemicalpotential, int output, int nthreads);
	void HFKSDerivative(int order, int output, int nthreads);
