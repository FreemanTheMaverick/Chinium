void GetDensity(double * aos,
                double * ao1xs,double * ao1ys,double * ao1zs,
                double * ao2ls,
                int ngrids,EigenMatrix D,
                double * ds,
                double * d1xs,double * d1ys,double * d1zs,double * cgs,
                double * d2s,double * ts);

void VectorAddition(double * as,double * bs,int ngrids);

double SumUp(double * ds,double * weights,int ngrids);

EigenMatrix FxcMatrix(double * aos,double * vrs,
                      double * d1xs,double * d1ys,double * d1zs,
                      double * ao1xs,double * ao1ys,double * ao1zs,double * vss,
                      double * ao2ls,double * vls,double * vts,
                      double * ws,int ngrids,int nbasis);
