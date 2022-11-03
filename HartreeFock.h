#include <Eigen/Dense>

#define EigenMatrix Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>

double RHF(int nele,EigenMatrix overlap,EigenMatrix hcore,double * repulsion,short int * indices,long int n2integrals,EigenMatrix & orbitalenergies,EigenMatrix & coefficients,EigenMatrix & densitymatrix,const int nprocs,const bool output);
