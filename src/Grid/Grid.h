class Grid{ public:
	Multiwfn* Mwfn;
	int Type = 0; // 0 - LDA, 1 - GGA, 2 - mGGA
	int NumGrids = 0;
	std::vector<double> Xs;
	std::vector<double> Ys;
	std::vector<double> Zs;
	Eigen::Tensor<double, 1> Weights;
	Grid(Multiwfn* mwfn, std::string grid, int output);
	Grid(Grid& another_grid, int from, int length, int output);

	Eigen::Tensor<double, 2> AOs; // g - grids, mu - basis
	Eigen::Tensor<double, 3> AO1s; // ..., (x, y, z)
	Eigen::Tensor<double, 2> AO2Ls;
	Eigen::Tensor<double, 3> AO2s; // ..., (xx, xy, yy, xz, yz, zz)
	Eigen::Tensor<double, 3> AO3s; // ..., (xxx, xxy, xyy, yyy, xxz, xyz, yyz, xzz, yzz, zzz)
	void getAO(int derivative, int output);

	// Temporary values (before (CP-)SCF converges)
	Eigen::Tensor<double, 1> Rhos_Cache;
	Eigen::Tensor<double, 2> Rho1s_Cache;
	Eigen::Tensor<double, 1> Sigmas_Cache;
	Eigen::Tensor<double, 1> Lapls_Cache;
	Eigen::Tensor<double, 1> Taus_Cache;
	void getDensity(EigenMatrix D);
	void getDensityU(
			std::vector<EigenMatrix> Ds_,
			std::vector<Eigen::Tensor<double, 1>>& Rhoss,
			std::vector<Eigen::Tensor<double, 2>>& Rho1ss,
			std::vector<Eigen::Tensor<double, 1>>& Sigmass
	);

	// True values
	Eigen::Tensor<double, 1> Rhos;
	Eigen::Tensor<double, 2> Rho1s;
	Eigen::Tensor<double, 1> Sigmas;
	Eigen::Tensor<double, 1> Lapls;
	Eigen::Tensor<double, 1> Taus;
	void SaveDensity();
	void RetrieveDensity();
	double getNumElectrons();

	Eigen::Tensor<double, 3> RhoGrads;
	Eigen::Tensor<double, 4> Rho1Grads;
	Eigen::Tensor<double, 3> SigmaGrads;
	void getDensitySkeleton(EigenMatrix D);

	Eigen::Tensor<double, 5> RhoHesss;
	Eigen::Tensor<double, 6> Rho1Hesss;
	Eigen::Tensor<double, 5> SigmaHesss;
	void getDensitySkeleton2(EigenMatrix D);

	Eigen::Tensor<double, 1> Es;
	Eigen::Tensor<double, 1> E1Rhos;
	Eigen::Tensor<double, 1> E1Sigmas;
	Eigen::Tensor<double, 1> E1Lapls;
	Eigen::Tensor<double, 1> E1Taus;
	Eigen::Tensor<double, 1> E2Rho2s;
	Eigen::Tensor<double, 1> E2RhoSigmas;
	Eigen::Tensor<double, 1> E2Sigma2s;
	Eigen::Tensor<double, 1> E3Rho3s;
	Eigen::Tensor<double, 1> E3Rho2Sigmas;
	Eigen::Tensor<double, 1> E3RhoSigma2s;
	Eigen::Tensor<double, 1> E3Sigma3s;
	Eigen::Tensor<double, 1> E4Rho4s;
	Eigen::Tensor<double, 1> E4Rho3Sigmas;
	Eigen::Tensor<double, 1> E4Rho2Sigma2s;
	Eigen::Tensor<double, 1> E4RhoSigma3s;
	Eigen::Tensor<double, 1> E4Sigma4s;

	double getEnergy();
	std::vector<double> getEnergyGrad();
	std::vector<std::vector<double>> getEnergyHess();

	EigenMatrix getFock(int type = -1);
	std::vector<EigenMatrix> getFockSkeleton();
	std::vector<EigenMatrix> getFockU();
	std::vector<EigenMatrix> getFockU(
			std::vector<Eigen::Tensor<double, 1>>& Rhoss,
			std::vector<Eigen::Tensor<double, 2>>& Rho1ss,
			std::vector<Eigen::Tensor<double, 1>>& Sigmass
	);
};
