void GetDensity(
		std::vector<int> orders,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		long int ngrids, EigenMatrix D,
		double* ds,
		double* d1xs, double* d1ys, double* d1zs,
		double* d2s, double* ts,
		int nthreads);

void GetDensitySkeleton(
		std::vector<int> orders,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2xxs, double* ao2yys, double* ao2zzs,
		double* ao2xys, double* ao2xzs, double* ao2yzs,
		long int ngrids_this_batch, EigenMatrix D,
		std::vector<int>& bf2atom,
		std::vector<std::vector<double*>>& ds,
		std::vector<std::vector<double*>>& d1xs,
		std::vector<std::vector<double*>>& d1ys,
		std::vector<std::vector<double*>>& d1zs);

void GetDensitySkeleton2(
		std::vector<int> orders,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2xxs, double* ao2yys, double* ao2zzs,
		double* ao2xys, double* ao2xzs, double* ao2yzs,
		double* ao3xxxs, double* ao3xxys, double* ao3xxzs,
		double* ao3xyys, double* ao3xyzs, double* ao3xzzs,
		double* ao3yyys, double* ao3yyzs, double* ao3yzzs, double* ao3zzzs,
		long int ngrids, EigenMatrix D,
		std::vector<int>& bf2atom,
		std::vector<std::vector<double*>>& hds,
		std::vector<std::vector<double*>>& hd1xs,
		std::vector<std::vector<double*>>& hd1ys,
		std::vector<std::vector<double*>>& hd1zs);
