	double Temperature = 0;
	double ChemicalPotential = 0;
	void GuessSCF(std::string guess);
	std::tuple<EigenMatrix, EigenMatrix, double> calcFock(EigenMatrix D, int nthreads);
	void HartreeFockKohnSham(int output, int nthreads);
	void HFKSDerivative(int order, int output, int nthreads);
