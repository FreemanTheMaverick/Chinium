void GetDensity(double * aos,
                double * ao1xs,double * ao1ys,double * ao1zs,
                double * ao2ls,
                int ngrids,EigenMatrix D,
                double * ds,
                double * d1xs,double * d1ys,double * d1zs,double * cgs,
                double * d2s,double * ts);

void VectorAddition(double * as,double * bs,int ngrids);

double SumUp(double * ds,double * weights,int ngrids);

EigenMatrix FxcMatrix(
		bool u,double * ws,int ngrids,int nbasis,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2ls,
		double * ds,
		double * d1xs,double * d1ys,double * d1zs,double * cgs,
		double * vrs,double * vss,
		double * vls,double * vts,
		double * vrrs,double * vrss,double * vsss);
