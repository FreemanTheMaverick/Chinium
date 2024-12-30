	double getCharge();
	double getNumElec(int spin);

	int getNumCenters();

	int getNumBasis();
	int getNumIndBasis();
	int getNumPrims();
	int getNumShells();
	int getNumPrimShells();

	EigenMatrix MatrixTransform();
	EigenMatrix getCoefficientMatrix();
	void setCoefficientMatrix(EigenMatrix matrix);

	EigenVector getEnergy();
	void setEnergy(EigenVector energies);
	void setEnergy(std::vector<double> energies);
	EigenVector getOccupation();
	void setOccupation(EigenVector occupancies);
	EigenMatrix getFock();
	EigenMatrix getDensity();
	EigenMatrix getEnergyDensity();

	Multiwfn() = default;
	Multiwfn(std::string mwfn_filename, const bool output);
	void Export(std::string mwfn_filename, const bool output);
	void PrintCenters();
	void PrintOrbitals();
	void setBasis(std::string basis_filename, const bool output);
	void setCenters(std::vector<std::vector<double>> atoms, const bool output);
