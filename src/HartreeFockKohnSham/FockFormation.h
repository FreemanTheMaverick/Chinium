std::tuple<double, EigenMatrix> Gxc(
		ExchangeCorrelation& xc,
		double* ws, long int ngrids, int nbasis,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		std::vector<EigenMatrix> Ds,
		double* rhos,
		double* rho1xs, double* rho1ys, double* rho1zs, double* sigmas,
		double* lapls, double* taus,
		double* es,
		double* erhos, double* esigmas,
		double* elapls, double* etaus,
		int nthreads);
