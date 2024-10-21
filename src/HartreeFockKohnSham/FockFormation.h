std::vector<long int> getThreadPointers(long int nitems, int nthreads);

EigenMatrix Ghf(
		short int* is, short int* js, short int* ks, short int* ls,
		char* degs, double* ints, long int length,
		EigenMatrix D, double kscale, int nthreads);