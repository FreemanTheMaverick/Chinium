void GetDensity(
		std::vector<int> orders,
		double* aos,
		double* ao1xs, double* ao1ys, double* ao1zs,
		double* ao2ls,
		long int ngrids, EigenMatrix D,
		double* ds,
		double* d1xs, double* d1ys, double* d1zs,
		double* d2s, double* ts);

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
		long int ngrids, EigenMatrix D,
		std::vector<int>& bf2atom,
		std::vector<std::vector<double*>>& hds);
