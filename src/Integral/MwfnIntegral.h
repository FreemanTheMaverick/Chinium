	void NuclearRepulsion(std::vector<double> orders, int output);

	void Normalize();

	// Two-center integrals
	EigenMatrix Overlap;
	EigenMatrix Kinetic;
	EigenMatrix Nuclear;
	EigenMatrix DipoleX;
	EigenMatrix DipoleY;
	EigenMatrix DipoleZ;
	EigenMatrix QuadrapoleXX;
	EigenMatrix QuadrapoleXY;
	EigenMatrix QuadrapoleXZ;
	EigenMatrix QuadrapoleYY;
	EigenMatrix QuadrapoleYZ;
	EigenMatrix QuadrapoleZZ;

	std::vector<std::vector<EigenMatrix>> OverlapGrads;
	std::vector<std::vector<EigenMatrix>> KineticGrads;
	std::vector<std::vector<EigenMatrix>> NuclearGrads;

	EigenMatrix OverlapHess;
	EigenMatrix KineticHess;
	EigenMatrix NuclearHess;

	void getTwoCenter(std::vector<int> orders, const bool output);

	// Four-center integrals
	std::tuple<EigenMatrix, EigenMatrix> RepulsionDiags;
	std::vector<short int> ShellIs;
	std::vector<short int> ShellJs;
	std::vector<short int> ShellKs;
	std::vector<short int> ShellLs;
	std::vector<short int> BasisIs;
	std::vector<short int> BasisJs;
	std::vector<short int> BasisKs;
	std::vector<short int> BasisLs;
	std::vector<double> RepulsionInts;
	std::vector<std::vector<EigenMatrix>> GGrads;
	EigenMatrix GHess;
	void getRepulsion(std::vector<int> orders, double threshold, int nthreads, const bool output);
