void PushVectorQueue(EigenVector M,EigenVector * Ms,int size);

void PushMatrixQueue(EigenMatrix M,EigenMatrix * Ms,int size);

void PushDoubleQueue(double M,double * Ms,int size);

EigenMatrix DIIS(EigenMatrix * Ds,EigenMatrix * Es,int maxsize,double & error2norm);

EigenMatrix AEDIIS(char diistype,double * Es,EigenMatrix * Ds,EigenMatrix * Fs,int size);

EigenVector LBFGS(EigenVector * pGs,EigenVector * pXs,int size,EigenVector hessiandiag);

EigenVector FABFGS(EigenVector * pGs,EigenVector * pXs,int size,EigenVector hessiandiag);

EigenMatrix AdaGrad(EigenMatrix * pGs,int size);
