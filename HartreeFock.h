#include <Eigen/Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;

void RHF(int nele,EigenMatrix overlap,EigenMatrix Hcore,double *repulsion,short int *indices,int nintegrals,EigenMatrix guesscoefficients,EigenMatrix& coefficients,double& energy);

