EigenMatrix VMatrix(
		std::vector<int> orders,
		double* ws, long int ngrids, int nbasis,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		double* d1xs, double* d1ys, double* d1zs,
		double* vrs, double* vss,
		double* vls, double* vts);

EigenMatrix PotentialSkeleton(
		std::vector<int> orders,
		double* ws, long int ngrids, int nbasis,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2xxs, double* ao2yys, double* ao2zzs,
		double* ao2xys, double* ao2xzs, double* ao2yzs,
		double* d1xs, double* d1ys, double* d1zs,
		double* vrs, double* vss,
		double* vrrs, double* vrss, double* vsss,
		double* gds,
		double* gd1xs, double* gd1ys, double* gd1zs);
