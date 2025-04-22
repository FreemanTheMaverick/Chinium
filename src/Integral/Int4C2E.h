class Int4C2E{ public:
	Multiwfn* Mwfn;
	double Threshold;
	double EXX;
	bool Verbose;

	std::tuple<EigenMatrix, EigenMatrix> RepulsionDiags;
	long int ShellQuartetLength;
	long int RepulsionLength;

	std::vector<short int> ShellIs;
	std::vector<short int> ShellJs;
	std::vector<short int> ShellKs;
	std::vector<short int> ShellLs;

	std::vector<long int> ShellQuartetHeads;
	std::vector<long int> BasisQuartetHeads;
	std::vector<short int> BasisIs;
	std::vector<short int> BasisJs;
	std::vector<short int> BasisKs;
	std::vector<short int> BasisLs;

	std::vector<double> RepulsionInts;
	std::vector<std::vector<double>> RepulsionGrads;
	std::vector<std::vector<std::vector<double>>> RepulsionHesss;
	std::vector<std::tuple<EigenMatrix, std::vector<EigenMatrix>>> GradCache;

	Int4C2E(Multiwfn& mwfn, double exx, double threshold, bool verbose);
	void getRepulsionDiag(); // Computing the diagonal elements of electron repulsion tensor for Cauchy-Schwarz screening.
	void getRepulsionLength(); // Numbers of nonequivalent two-electron integrals and shell quartets after Cauchy-Schwarz screening.
	void getRepulsionIndices();
	void getThreadPointers(int nthreads);
	void CalculateIntegrals(int order);
	EigenMatrix ContractInts(EigenMatrix D, int nthreads);
	std::vector<EigenMatrix> ContractInts(std::vector<EigenMatrix>& Ds, int nthreads);
	std::tuple<EigenMatrix, EigenMatrix> ContractInts(EigenMatrix Da, EigenMatrix Db, int nthreads);
	std::vector<EigenMatrix> ContractGrads(EigenMatrix D);
	std::vector<double> ContractGrads(EigenMatrix D1, EigenMatrix D2);
	std::vector<std::vector<double>> ContractGrads(std::vector<EigenMatrix>& D1s, EigenMatrix D2);
	//std::vector<std::vector<EigenMatrix>> ContractHesss(EigenMatrix D);
	std::vector<std::vector<double>> ContractHesss(EigenMatrix D1, EigenMatrix D2);
};
