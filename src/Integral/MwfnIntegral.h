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
	EigenMatrix RepulsionDiag;
	long int ShellQuartetLength = 0;
	short int* ShellIs = nullptr;
	short int* ShellJs = nullptr;
	short int* ShellKs = nullptr;
	short int* ShellLs = nullptr;
	long int RepulsionLength = 0;
	short int* RepulsionIs = nullptr;
	short int* RepulsionJs = nullptr;
	short int* RepulsionKs = nullptr;
	short int* RepulsionLs = nullptr;
	char* RepulsionDegs = nullptr;
	double* Repulsions = nullptr;
	std::vector<std::vector<EigenMatrix>> GGrads;
	EigenMatrix GHess;
	void getRepulsion(std::vector<int> orders, double threshold, int nthreads, const bool output);
