#include "Tensor.h"

enum D_t{ s_t, u_t };

class SubGrid{ public:
	Mwfn* MWFN;
	int NumGrids;
	int Type;
	std::vector<double> X;
	std::vector<double> Y;
	std::vector<double> Z;
	EigenTensor<1> W;
	std::vector<int> BasisList;
	std::vector<int> AtomList;
	std::vector<int> AtomHeads;
	std::vector<int> AtomLengths;
	SubGrid(){};
	SubGrid(EigenMatrix points);
	int getNumBasis(){ return this->BasisList.size(); };
	int getNumAtoms(){ return this->AtomList.size(); };

	EigenTensor<2> AO; // g - grids, mu - basis
	EigenTensor<3> AO1; // ..., (x, y, z)
	EigenTensor<2> AO2L;
	EigenTensor<3> AO2; // ..., (xx, xy, yy, xz, yz, zz)
	EigenTensor<3> AO3; // ..., (xxx, xxy, xyy, yyy, xxz, xyz, yyz, xzz, yzz, zzz)
	void getAO(int derivative);

	EigenTensor<1> Rho;
	EigenTensor<2> Rho1;
	EigenTensor<1> Sigma;
	EigenTensor<1> Lapl;
	EigenTensor<1> Tau;
	void getNumElectrons(double& n);
	void getDensity(EigenMatrix& D);

	EigenTensor<2> RhoU;
	EigenTensor<3> Rho1U;
	EigenTensor<2> SigmaU;
	void getDensityU(std::vector<EigenMatrix>& Ds);

	EigenTensor<3> RhoGrad;
	EigenTensor<4> Rho1Grad;
	EigenTensor<3> SigmaGrad;
	void getDensitySkeleton(EigenMatrix& D);

	EigenTensor<5> RhoHess;
	EigenTensor<6> Rho1Hess;
	EigenTensor<5> SigmaHess;
	void getDensitySkeleton2(EigenMatrix& D);

	EigenTensor<1> E;
	EigenTensor<1> E1Rho;
	EigenTensor<1> E1Sigma;
	EigenTensor<1> E1Lapl;
	EigenTensor<1> E1Tau;
	EigenTensor<1> E2Rho2;
	EigenTensor<1> E2RhoSigma;
	EigenTensor<1> E2Sigma2;
	EigenTensor<1> E3Rho3;
	EigenTensor<1> E3Rho2Sigma;
	EigenTensor<1> E3RhoSigma2;
	EigenTensor<1> E3Sigma3;
	EigenTensor<1> E4Rho4;
	EigenTensor<1> E4Rho3Sigma;
	EigenTensor<1> E4Rho2Sigma2;
	EigenTensor<1> E4RhoSigma3;
	EigenTensor<1> E4Sigma4;

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
	Grid(Mwfn* mwfn, std::string grid, int nthreads, int output);
	int getNumThreads(){ return (int)this->SubGridBatches.size(); };
	void setType(int type);

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
