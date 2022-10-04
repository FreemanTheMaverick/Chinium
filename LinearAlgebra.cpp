#include <iostream>
#include <Eigen/Dense>

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;

EigenMatrix FormMatrix_eigen(double * matrix,int nrows,int ncols,char storage){
	EigenMatrix eigenmatrix(nrows,ncols);eigenmatrix=eigenmatrix*0;
	if (storage=='f'){
		for (int irow=0;irow<nrows;irow++){
			for (int icol=0;icol<ncols;icol++){
				eigenmatrix(irow,icol)=matrix[irow*ncols+icol];
			}
		}
	}else if (storage=='l'){
		for (int irow=0;irow<nrows;irow++){
			for (int icol=0;icol<ncols;icol++){
				if (irow>=icol){
					eigenmatrix(irow,icol)=matrix[irow*(irow+1)/2+icol];
					eigenmatrix(icol,irow)=matrix[irow*(irow+1)/2+icol];
				}else{
					eigenmatrix(irow,icol)=matrix[icol*(icol+1)/2+irow];
					eigenmatrix(icol,irow)=matrix[icol*(icol+1)/2+irow];
				}
			}
		}
	}
	return eigenmatrix;
}

void PrintMatrix_eigen(double * matrix,int nrows,int ncols,char storage){
	std::cout<<FormMatrix_eigen(matrix,nrows,ncols,storage)<<std::endl;
}

void FormArray(EigenMatrix eigenmatrix,int nrows,int ncols,char storage,double * matrix){
	double * matrixranger=matrix;
	if (storage=='f'){
		for (int irow=0;irow<nrows;irow++){
			for (int icol=0;icol<ncols;icol++){
				*(matrixranger++)=eigenmatrix(irow,icol);
			}
		}
	}else if (storage=='l'){
		for (int irow=0;irow<nrows;irow++){
			for (int icol=0;icol<=irow;icol++){
				*(matrixranger++)=eigenmatrix(irow,icol);
			}
		}
	}
}

void GeneralizedSelfAdjointEigenSolver(double * matrix1,double * matrix2,int nrows,char storage,double * eigenvalues_,double * eigenvectors_){
	EigenMatrix eigenmatrix1=FormMatrix_eigen(matrix1,nrows,nrows,storage);
	EigenMatrix eigenmatrix2=FormMatrix_eigen(matrix2,nrows,nrows,storage);
	Eigen::GeneralizedSelfAdjointEigenSolver<EigenMatrix> solver(eigenmatrix1,eigenmatrix2);
	FormArray(solver.eigenvalues(),nrows,1,'f',eigenvalues_);
	FormArray(solver.eigenvectors(),nrows,nrows,'f',eigenvectors_);
}

void MultiplyMatrix(double a,double * A,char storageA,bool transposeA,double * B,char storageB,bool transposeB,double b,double * C,char storageC,bool transposeC,int nrowsA,int ncolsA,int nrowsB,int ncolsB,int nrowsC,int ncolsC,int nrows,int ncols,char storage,double * matrix){ // a*A(^T)*B(^T)+b*C(^T)
	EigenMatrix eigenmatrixA_=FormMatrix_eigen(A,nrowsA,ncolsA,storageA);EigenMatrix eigenmatrixA=transposeA?eigenmatrixA_.transpose():eigenmatrixA_;
	EigenMatrix eigenmatrixB_=FormMatrix_eigen(B,nrowsB,ncolsB,storageB);EigenMatrix eigenmatrixB=transposeB?eigenmatrixB_.transpose():eigenmatrixB_;
	EigenMatrix eigenmatrixC_=FormMatrix_eigen(C,nrowsC,ncolsC,storageC);EigenMatrix eigenmatrixC=transposeC?eigenmatrixC_.transpose():eigenmatrixC_;
	EigenMatrix result=a*eigenmatrixA*eigenmatrixB+b*eigenmatrixC;
	FormArray(result,nrows,ncols,storage,matrix);
}

double Trace(double * matrix,int nrows,char storage){
	EigenMatrix eigenmatrix=FormMatrix_eigen(matrix,nrows,nrows,storage);
	return eigenmatrix.trace();
}

void SliceMatrix(double * initialmatrix,int initialnrows,int initialncols,int up,int down,int left,int right,double * finalmatrix){
	double * finalmatrixranger=finalmatrix;
	for (int irow=up;irow<=down;irow++){
		for (int icol=left;icol<=right;icol++){
			*(finalmatrixranger++)=initialmatrix[irow*initialncols+icol];
		}
	}
}


/*
int main(){
	double C_occ[]={-0.439204,-0.164756,-0.439204,-0.164756};
	double densitymatrix[]={0,0,0,0,0,0,0,0,0,0};
	MultiplyMatrix(1,C_occ,'f',0,C_occ,'f',1,0,densitymatrix,'l',0,4,1,4,1,4,4,4,4,'l',densitymatrix);
	PrintMatrix_eigen(densitymatrix,4,4,'l');
	
	return 0;
}*/
