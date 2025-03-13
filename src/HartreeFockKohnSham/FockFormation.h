std::vector<long int> getThreadPointers(long int nitems, int nthreads);

EigenMatrix Grhf(
		short int* is, short int* js, short int* ks, short int* ls,
		double* ints, long int length,
		EigenMatrix D, double kscale, int nthreads);

std::tuple<EigenMatrix, EigenMatrix, EigenMatrix, EigenMatrix> Guhf(
		short int* is, short int* js, short int* ks, short int* ls,
		double* ints, long int length,
		EigenMatrix Da, EigenMatrix Db, double kscale, int nthreads);
	
std::vector<EigenMatrix> GhfMultiple(
		short int* is, short int* js, short int* ks, short int* ls,
		double* ints, long int length,
		std::vector<EigenMatrix>& Ds, double kscale, int nthreads);

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
