#include <libint2.hpp>
#include <Eigen/Dense> // Eigen::Matrix.
#include <vector> // Atom vectors.

typedef Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;

EigenMatrix Overlap(libint2::BasisSet obs);
EigenMatrix Kinetic(libint2::BasisSet obs);
EigenMatrix Nuclear(libint2::BasisSet obs,std::vector<libint2::Atom> atoms);
void Repulsion(libint2::BasisSet obs,double **repulsion,short int **indices,int& nintegrals);



