void GetDensity(double * aos,
                double * ao1xs,double * ao1ys,double * ao1zs,
                double * ao2ls,
                int ngrids,EigenMatrix D,
                double * ds,
                double * d1xs,double * d1ys,double * d1zs,double * cgs,
                double * d2s,double * ts);

void GetDensitySkeleton(
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2xxs,double * ao2yys,double * ao2zzs,
		double * ao2xys,double * ao2xzs,double * ao2yzs,
		int ngrids,EigenMatrix D,
		int atom,int * bf2atom,
		double * d1nxs,double * d1nys,double * d1nzs,
		double * d2nxxs,double * d2nxys,double * d2nxzs, // For example, d2nxys is the x component of grid density gradient derivative with respect to a nuclear y coordinate perturbation.
		double * d2nyxs,double * d2nyys,double * d2nyzs,
		double * d2nzxs,double * d2nzys,double * d2nzzs);

void VectorAddition(double * as,double * bs,double * cs,int ngrids);

void VectorMultiplication(double * as,double * bs,double * cs,int ngrids);

double SumUp(double * ds,double * weights,int ngrids);

EigenMatrix FxcMatrix(
		double * ws,int ngrids,int nbasis,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2ls,
		double * ds,
		double * d1xs,double * d1ys,double * d1zs,double * cgs,
		double * vrs,double * vss,
		double * vls,double * vts);

EigenMatrix FxcUMatrix(
		double * ws,int ngrids,int nbasis,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * d1xs,double * d1ys,double * d1zs,
		double * vss,
		double * vrrs,
		double * vrss,double * vsss,
		double * dns,
		double * dxns,double * dyns,double * dzns);

