class Int4C2E{ public:
	Mwfn* MWFN;
	double Threshold;
	double EXX;

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

	Int4C2E(Mwfn& mwfn, double exx, double threshold);
	void getRepulsionDiag(int output); // Computing the diagonal elements of electron repulsion tensor for Cauchy-Schwarz screening.
	void getRepulsionLength(int output); // Numbers of nonequivalent two-electron integrals and shell quartets after Cauchy-Schwarz screening.
	void getRepulsionIndices(int output);
	void getThreadPointers(int nthreads, int output);
	void CalculateIntegrals(int order, int output);
	EigenMatrix ContractInts(EigenMatrix D, int nthreads, int output);
	std::tuple<EigenMatrix, EigenMatrix> ContractInts2(EigenMatrix Da, int nthreads, int output);
	std::vector<EigenMatrix> ContractInts(std::vector<EigenMatrix>& Ds, int nthreads, int output);
	std::tuple<EigenMatrix, EigenMatrix> ContractInts(EigenMatrix Da, EigenMatrix Db, int nthreads, int output);
	std::vector<EigenMatrix> ContractGrads(EigenMatrix D, int output);
	std::vector<double> ContractGrads(EigenMatrix D1, EigenMatrix D2, int output);
	std::vector<std::vector<double>> ContractGrads(std::vector<EigenMatrix>& D1s, EigenMatrix D2, int output);
	//std::vector<std::vector<EigenMatrix>> ContractHesss(EigenMatrix D);
	std::vector<std::vector<double>> ContractHesss(EigenMatrix D1, EigenMatrix D2, int output);
};
