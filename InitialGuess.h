#include <Eigen/Dense>

#define EigenMatrix Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>

EigenMatrix CoreHamiltonian(const int natoms,double * atoms,const char * basisset,const bool output);

EigenMatrix SuperpositionAtomicDensity(int nele,const int natoms,double * atoms,const char * basisset,const bool output);
