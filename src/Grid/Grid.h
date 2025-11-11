#include "Tensor.h"

enum D_t{ s_t, u_t };

class SubGrid{ public:
	Mwfn* MWFN;
	int NumGrids;
	int Type = 0; // 0 - LDA, 1 - GGA, 2 - mGGA
	int Spin = -1;

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

	EigenTensor<2> Rho;
	EigenTensor<3> Rho1;
	EigenTensor<2> Sigma;
	EigenTensor<2> Lapl;
	EigenTensor<2> Tau;
	void getNumElectrons(EigenTensor<0>& n);
	void getDensity(EigenTensor<3>& D);

	EigenTensor<3> RhoU;
	EigenTensor<4> Rho1U;
	EigenTensor<3> SigmaU;
	void getDensityU(EigenTensor<4>& D);

	EigenTensor<4> RhoGrad;
	EigenTensor<5> Rho1Grad;
	EigenTensor<4> SigmaGrad;
	void getDensitySkeleton(EigenTensor<3>& D);

	EigenTensor<6> RhoHess;
	EigenTensor<7> Rho1Hess;
	EigenTensor<6> SigmaHess;
	void getDensitySkeleton2(EigenTensor<3>& D);

	EigenTensor<1> Eps;
	EigenTensor<2> Eps1Rho;
	EigenTensor<2> Eps1Sigma;
	EigenTensor<2> Eps1Lapl;
	EigenTensor<2> Eps1Tau;
	EigenTensor<3> Eps2Rho2;
	EigenTensor<3> Eps2RhoSigma;
	EigenTensor<3> Eps2Sigma2;
	EigenTensor<4> Eps3Rho3;
	EigenTensor<4> Eps3Rho2Sigma;
	EigenTensor<4> Eps3RhoSigma2;
	EigenTensor<4> Eps3Sigma3;
	EigenTensor<5> Eps4Rho4;
	EigenTensor<5> Eps4Rho3Sigma;
	EigenTensor<5> Eps4Rho2Sigma2;
	EigenTensor<5> Eps4RhoSigma3;
	EigenTensor<5> Eps4Sigma4;

	void getEnergy(EigenTensor<0>& E);
	void getEnergyGrad(EigenTensor<2>& E);
	void getEnergyHess(EigenTensor<4>& E);

	void getFock(EigenTensor<3>& F);
	void getFockSkeleton(EigenTensor<5>& F);
	template <D_t d_t> void getFockU(EigenTensor<4>& F);
};

class Grid{ public:
	std::vector<std::vector<std::unique_ptr<SubGrid>>> SubGridBatches;
	Grid(Mwfn* mwfn, std::string grid, int nthreads, int output);
	int getNumThreads(){ return (int)this->SubGridBatches.size(); };
	void setType(int type);

	void getAO(int derivative, int output);

	double getNumElectrons();
	void getDensity(std::vector<EigenMatrix> D);
	void getDensityU(std::vector<std::vector<EigenMatrix>> D);
	void getDensitySkeleton(std::vector<EigenMatrix> D);
	void getDensitySkeleton2(std::vector<EigenMatrix> D);

	double getEnergy();
	std::vector<double> getEnergyGrad();
	std::vector<std::vector<double>> getEnergyHess();

	std::vector<EigenMatrix> getFock();
	std::vector<std::vector<EigenMatrix>> getFockSkeleton();
	template <D_t d_t> std::vector<std::vector<EigenMatrix>> getFockU();

	void WhatDoWeHave();
};
