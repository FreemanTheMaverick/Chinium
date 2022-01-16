#include <libint2.hpp>
#include <Eigen/Dense> // Eigen::Matrix.
#include <unsupported/Eigen/CXX11/Tensor> // Eigen::Tensor.
#include <vector> // Atom vectors.

typedef Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;
typedef Eigen::Tensor<double,2> EigenTensor2;
typedef Eigen::Tensor<double,3> EigenTensor3;

std::vector<libint2::Atom> atoms;
int nBasis(libint2::BasisSet obs);
EigenMatrix Overlap(libint2::BasisSet obs);
EigenMatrix Kinetic(libint2::BasisSet obs);
EigenMatrix Nuclear(libint2::BasisSet obs,std::vector<libint2::Atom> atoms);
void Repulsion(libint2::BasisSet obs,double **repulsion,short int **indices,int& nintegrals);



