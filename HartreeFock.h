#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


typedef Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;
typedef Eigen::Tensor<double,2> EigenTensor2;
typedef Eigen::Tensor<double,3> EigenTensor3;

void RHF(int nele,EigenMatrix overlap,EigenMatrix Hcore,double *repulsion,short int *indices,int nintegrals,EigenMatrix guesscoefficients,EigenMatrix& coefficients,double& energy);

