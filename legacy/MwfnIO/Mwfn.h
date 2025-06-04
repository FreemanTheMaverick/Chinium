class Mwfn{ public:
	// Field 1
	int Wfntype = -114;
	double E_tot = -114;
	double VT_ratio = -114;

	// Fields 2 & 3
	std::vector<MwfnCenter> Centers = {};

	// Field 4
	std::vector<MwfnOrbital> Orbitals = {};

	// Field 5
	EigenMatrix Overlap;

	double getCharge();
	double getNumElec(int spin = -1);

	int getNumCenters();

	int getNumBasis();
	int getNumIndBasis();
	int getNumPrims();
	int getNumShells();
	int getNumPrimShells();

	EigenMatrix MatrixTransform();
	EigenMatrix getCoefficientMatrix(int spin = 0);
	void setCoefficientMatrix(EigenMatrix matrix, int spin = 0);

	EigenVector getEnergy(int spin = 0);
	void setEnergy(EigenVector energies, int spin = 0);
	void setEnergy(std::vector<double> energies, int spin = 0);
	EigenVector getOccupation(int spin = 0);
	void setOccupation(EigenVector occupancies, int spin = 0);
	EigenMatrix getFock(int spin = 0);
	EigenMatrix getDensity(int spin = 0);
	EigenMatrix getEnergyDensity(int spin = 0);
	std::vector<int> Basis2Atom();
	std::vector<int> Atom2Basis();

	Mwfn() = default;
	Mwfn(std::string mwfn_filename, const bool output);
	void Export(std::string mwfn_filename, const bool output);
	void PrintCenters();
	void PrintOrbitals();
	void setBasis(std::string basis_filename, const bool output);
	void setCenters(std::vector<std::vector<double>> atoms, const bool output);
	std::tuple<double, EigenMatrix, EigenMatrix> NuclearRepulsion(int output);
};
