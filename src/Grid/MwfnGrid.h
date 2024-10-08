	long int NumGrids = 0;
	double* Xs = nullptr;
	double* Ys = nullptr;
	double* Zs = nullptr;
	double* Ws = nullptr;
	double* AOs = nullptr;
	double* AO1Xs = nullptr;
	double* AO1Ys = nullptr;
	double* AO1Zs = nullptr;
	double* AO2Ls = nullptr;
	double* AO2XXs = nullptr;
	double* AO2YYs = nullptr;
	double* AO2ZZs = nullptr;
	double* AO2XYs = nullptr;
	double* AO2XZs = nullptr;
	double* AO2YZs = nullptr;
	void GenerateGrid(std::string grid, int order, const bool output);

	double* GridDensity = nullptr;
	void getGridDensity(EigenMatrix D, const bool output);


