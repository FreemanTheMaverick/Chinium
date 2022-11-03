#include <Eigen/Dense>

#define EigenMatrix Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>

EigenMatrix DIIS(EigenMatrix * Ds,EigenMatrix * Es,int size,double & error2norm);

EigenMatrix LBFGS(EigenMatrix g,EigenMatrix * Ss,EigenMatrix * Ys,int size,int latest);
