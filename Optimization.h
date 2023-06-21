void PushMatrixQueue(EigenMatrix M,EigenMatrix * Ms,int size);

void PushDoubleQueue(double M,double * Ms,int size);

EigenMatrix DIIS(EigenMatrix * Ds,EigenMatrix * Es,int size,double & error2norm);

EigenMatrix LBFGS(EigenMatrix g,EigenMatrix * Ss,EigenMatrix * Ys,int size,int latest);
