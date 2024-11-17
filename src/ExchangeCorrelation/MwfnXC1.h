#pragma once

class ExchangeCorrelation{ public:
	int Xcode = 0; int Ccode = 0; int XCcode = 0;
	int Xkind; int Ckind; int XCkind;
	std::string Xname; std::string Cname; std::string XCname;
	std::string Xfamily; std::string Cfamily; std::string XCfamily;
	double EXX = 1; int Spin = 1;

	void Read(std::string df, bool output);
	void Print();
	void Evaluate(
		std::string order, long int ngrids,
		double* rhos, // Input for LDA
		double* sigmas, // Input for GGA
		double* lapls, double* taus, // Input for mGGA
		double* es, // Output epsilon
		double* erhos, double* esigmas, double* elapls, double* etaus, // First-order derivatives
		double* e2rho2s, double* e2rhosigmas, double* e2sigma2s, // Second-order derivatives
		double* e3rho3s, double* e3rho2sigmas, double* e3rhosigma2s, double* e3sigma3s, // Third-order derivatives
		double* e4rho4s, double* e4rho3sigmas, double* e4rho2sigma2s, double* e4rhosigma3s, double* e4sigma4s // Fourth-order derivatives
		);
};
