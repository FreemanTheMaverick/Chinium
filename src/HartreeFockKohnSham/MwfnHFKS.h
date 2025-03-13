	double Temperature = 0;
	double ChemicalPotential = 0;
	void GuessSCF(std::string guess, const bool output);
	void HartreeFockKohnSham(std::string scf, int output, int nthreads);
	void HFKSDerivative(int order, int output, int nthreads);
