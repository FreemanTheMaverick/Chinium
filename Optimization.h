void PushMatrixQueue(EigenMatrix M,EigenMatrix * Ms,int size);

void PushDoubleQueue(double M,double * Ms,int size);

EigenMatrix DIIS(EigenMatrix * Ds,EigenMatrix * Es,int maxsize,double & error2norm);

EigenMatrix AEDIIS(char diistype,double * Es,EigenMatrix * Ds,EigenMatrix * Fs,int size);

EigenMatrix LBFGS(EigenMatrix g,EigenMatrix * Ss,EigenMatrix * Ys,int size,int latest);
