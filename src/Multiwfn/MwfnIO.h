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

	void Export(std::string mwfn_filename, const bool output);
	void Print();
	Multiwfn(std::string mwfn_filename, const bool output);
	Multiwfn(std::string mwfn_filename, std::string basis_filename, const bool output);
