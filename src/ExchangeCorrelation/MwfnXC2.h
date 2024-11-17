	ExchangeCorrelation XC;
	double* Es = nullptr;

	double* E1Rhos = nullptr;
	double* E1Sigmas = nullptr;
	double* E1Lapls = nullptr;
	double* E1Taus = nullptr;

	double* E2Rho2s = nullptr;
	double* E2RhoSigmas = nullptr;
	double* E2Sigma2s = nullptr;

	double* E3Rho3s = nullptr;
	double* E3Rho2Sigmas = nullptr;
	double* E3RhoSigma2s = nullptr;
	double* E3Sigma3s = nullptr;

	double* E4Rho4s = nullptr;
	double* E4Rho3Sigmas = nullptr;
	double* E4Rho2Sigma2s = nullptr;
	double* E4RhoSigma3s = nullptr;
	double* E4Sigma4s = nullptr;

	void PrepareXC(std::string order, int output);
