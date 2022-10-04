#include <Eigen/Dense>

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;

EigenMatrix FormMatrix_eigen(double * matrix,int nrows,int ncols,char storage);

void PrintMatrix_eigen(double * matrix,int nrows,int ncols,char storage);

void FormArray(EigenMatrix eigenmatrix,int nrows,int ncols,char storage,double * matrix);

void GeneralizedSelfAdjointEigenSolver(double * matrix1,double * matrix2,int nrows,char storage,double * eigenvalues_,double * eigenvectors_);

void MultiplyMatrix(double a,double * A,char storageA,bool transposeA,double * B,char storageB,bool transposeB,double b,double * C,char storageC,bool transposeC,int nrowsA,int ncolsA,int nrowsB,int ncolsB,int nrowsC,int ncolsC,int nrows,int ncols,char storage,double * matrix); // a*A(^T)*B(^T)+b*C(^T)

double Trace(double * matrix,int nrows,char storage);

void SliceMatrix(double * initialmatrix,int initialnrows,int initialncols,int up,int down,int left,int right,double * finalmatrix);


