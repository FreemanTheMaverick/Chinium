enum D_t{ s_t, u_t };

class SubGrid{ public:
	Mwfn* MWFN;
	int NumGrids;
	std::vector<double> X;
	std::vector<double> Y;
	std::vector<double> Z;
	Eigen::Tensor<double, 1> W;
	std::vector<int> BasisList;
	std::vector<int> AtomList;
	std::vector<int> AtomHeads;
	std::vector<int> AtomLengths;
	SubGrid(EigenMatrix points);
	int getNumBasis(){ return this->BasisList.size(); };
	int getNumAtoms(){ return this->AtomList.size(); };

	Eigen::Tensor<double, 2> AO; // g - grids, mu - basis
	Eigen::Tensor<double, 3> AO1; // ..., (x, y, z)
	Eigen::Tensor<double, 2> AO2L;
	Eigen::Tensor<double, 3> AO2; // ..., (xx, xy, yy, xz, yz, zz)
	Eigen::Tensor<double, 3> AO3; // ..., (xxx, xxy, xyy, yyy, xxz, xyz, yyz, xzz, yzz, zzz)
	void getAO(int derivative, int output);

	Eigen::Tensor<double, 1> Rho;
	Eigen::Tensor<double, 2> Rho1;
	Eigen::Tensor<double, 1> Sigma;
	Eigen::Tensor<double, 1> Lapl;
	Eigen::Tensor<double, 1> Tau;
	void getNumElectrons();
	void getDensity(EigenMatrix& D);

	Eigen::Tensor<double, 2> RhoU;
	Eigen::Tensor<double, 3> Rho1U;
	Eigen::Tensor<double, 2> SigmaU;
	void getDensityU(std::vector<EigenMatrix>& Ds);

	Eigen::Tensor<double, 3> RhoGrad;
	Eigen::Tensor<double, 4> Rho1Grad;
	Eigen::Tensor<double, 3> SigmaGrad;
	void getDensitySkeleton(EigenMatrix& D);

	Eigen::Tensor<double, 5> RhoHess;
	Eigen::Tensor<double, 6> Rho1Hess;
	Eigen::Tensor<double, 5> SigmaHess;
	void getDensitySkeleton2(EigenMatrix& D);

	Eigen::Tensor<double, 1> E;
	Eigen::Tensor<double, 1> E1Rho;
	Eigen::Tensor<double, 1> E1Sigma;
	Eigen::Tensor<double, 1> E1Lapl;
	Eigen::Tensor<double, 1> E1Tau;
	Eigen::Tensor<double, 1> E2Rho2;
	Eigen::Tensor<double, 1> E2RhoSigma;
	Eigen::Tensor<double, 1> E2Sigma2;
	Eigen::Tensor<double, 1> E3Rho3;
	Eigen::Tensor<double, 1> E3Rho2Sigma;
	Eigen::Tensor<double, 1> E3RhoSigma2;
	Eigen::Tensor<double, 1> E3Sigma3;
	Eigen::Tensor<double, 1> E4Rho4;
	Eigen::Tensor<double, 1> E4Rho3Sigma;
	Eigen::Tensor<double, 1> E4Rho2Sigma2;
	Eigen::Tensor<double, 1> E4RhoSigma3;
	Eigen::Tensor<double, 1> E4Sigma4;

	void getEnergy(double& e);
	void getEnergyGrad(std::vector<double>& e);
	void getEnergyHess(std::vector<std::vector<double>>& e);

	void getFock(EigenMatrix& F);
	void getFockSkeleton(std::vector<EigenMatrix>& Fs);
	template <D_t d_t> void getFockU(std::vector<EigenMatrix>& Fs);
};

class Grid{ public:
	Mwfn* MWFN;
	int Type = 0; // 0 - LDA, 1 - GGA, 2 - mGGA
	std::vector<std::vector<std::unique_ptr<SubGrid>>> SubGridBatches;
	Grid(Mwfn* mwfn, std::string grid, int output);
	int getNumThreads(){ return (int)this->SubGridBatches.size(); };

	void getAO(int derivative, int output);

	double getNumElectrons();
	void getDensity(EigenMatrix D);
	void getDensityU(std::vector<EigenMatrix> Ds);
	void getDensitySkeleton(EigenMatrix D);
	void getDensitySkeleton2(EigenMatrix D);

	double getEnergy();
	std::vector<double> getEnergyGrad();
	std::vector<std::vector<double>> getEnergyHess();

	EigenMatrix getFock();
	std::vector<EigenMatrix> getFockSkeleton();
	template <D_t d_t> std::vector<EigenMatrix> getFockU();

	void WhatDoWeHave();
};
