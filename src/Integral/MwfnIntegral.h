//class Multiwfn{ public:

	void Normalize();

	// Two-center integrals
	/*
	EigenMatrix Overlap;
	EigenMatrix Kinetic;
	EigenMatrix Nuclear;
	*/
	EigenMatrix DipoleX;
	EigenMatrix DipoleY;
	EigenMatrix DipoleZ;
	EigenMatrix QuadrapoleXX;
	EigenMatrix QuadrapoleXY;
	EigenMatrix QuadrapoleXZ;
	EigenMatrix QuadrapoleYY;
	EigenMatrix QuadrapoleYZ;
	EigenMatrix QuadrapoleZZ;
	void getTwoCenter(int order, const bool output);

	// Four-center integrals
	EigenMatrix RepulsionDiag;
	long int RepulsionLength = 0;
	short int* RepulsionIs = nullptr;
	short int* RepulsionJs = nullptr;
	short int* RepulsionKs = nullptr;
	short int* RepulsionLs = nullptr;
	char* RepulsionDegs = nullptr;
	double* Repulsions = nullptr;
	void getRepulsion(double threshold, int nthreads, const bool output);

//};
