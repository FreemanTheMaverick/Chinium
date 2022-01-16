#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <math.h>
#include <iostream>
#include <time.h>

typedef Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;
typedef Eigen::Tensor<double,2> EigenTensor2;
typedef Eigen::Tensor<double,3> EigenTensor3;

EigenMatrix GMatrix(double *repulsion,short int *indices,int nintegrals,EigenMatrix densitymatrix,EigenMatrix &jmatrix,EigenMatrix &kmatrix){ // Memory seats for J matrix and K matrix should be provided.
	double *repulsionranger=repulsion;
	short int *indicesranger=indices;
	int nbasis=jmatrix.rows();
	EigenMatrix rawjmatrix=EigenMatrix::Zero(nbasis,nbasis);
	EigenMatrix rawkmatrix=EigenMatrix::Zero(nbasis,nbasis);
	for (int i=0;i<nintegrals;i++){
		short int a=*indicesranger;indicesranger++; // Moving the ranger pointer to the right, where the next index is located.
		short int b=*indicesranger;indicesranger++;
		short int c=*indicesranger;indicesranger++;
		short int d=*indicesranger;indicesranger++; // Moving the ranger pointer to the right, where the degeneracy factor is located
		double s1s2s3s4_deg=*indicesranger;indicesranger++; // Moving the ranger pointer to the right, where the first index of the next integral is located.
		double value=*repulsionranger;repulsionranger++; // Moving the ranger pointer to the right, where the value of the next integral is located.
		double deg_value=s1s2s3s4_deg*value;
		rawjmatrix(a,b)+=densitymatrix(c,d)*deg_value;
		rawjmatrix(c,d)+=densitymatrix(a,b)*deg_value;
		rawkmatrix(a,c)+=0.5*densitymatrix(b,d)*deg_value;
		rawkmatrix(b,d)+=0.5*densitymatrix(a,c)*deg_value;
		rawkmatrix(a,d)+=0.5*densitymatrix(b,c)*deg_value;
		rawkmatrix(b,c)+=0.5*densitymatrix(a,d)*deg_value;
	}
	jmatrix=(rawjmatrix+rawjmatrix.transpose())/2;
	kmatrix=(rawkmatrix+rawkmatrix.transpose())/2;
	EigenMatrix gmatrix=jmatrix-0.5*kmatrix;
	return gmatrix;
}

void RHF(int nele,EigenMatrix overlap,EigenMatrix Hcore,double *repulsion,short int *indices,int nintegrals,EigenMatrix guesscoefficients,EigenMatrix& coefficients,double& energy){
	std::cout<<" ... Entering RHF ..."<<std::endl;
	int nocc=nele/2;
	EigenMatrix C_occ=guesscoefficients.leftCols(nocc); // Normal Hartree-Fock procedure.
	EigenMatrix densitymatrix=C_occ*C_occ.transpose();
	int nbasis=densitymatrix.cols();
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(overlap);
	EigenMatrix s=eigensolver.eigenvalues();
	EigenMatrix sinversesqrt=Eigen::MatrixXd::Zero(nbasis,nbasis);
	for (int i=0;i<nbasis;i++){
		sinversesqrt(i,i)=1/sqrt(s(i,0));
	}
	EigenMatrix U=eigensolver.eigenvectors();
	EigenMatrix X=U*sinversesqrt;
	energy=114514;
	double lastenergy=1919810;
	int iiteration=1;
	clock_t start,end;
	while (abs(lastenergy-energy)>1.0e-8){
		start=clock();
		lastenergy=energy;
		EigenMatrix jmatrix(nbasis,nbasis);
		EigenMatrix kmatrix(nbasis,nbasis);
		EigenMatrix G=GMatrix(repulsion,indices,nintegrals,densitymatrix,jmatrix,kmatrix);
		EigenMatrix F=Hcore+G;
		EigenMatrix Fprime=X.transpose()*F*X;
		eigensolver.compute(Fprime);
		EigenMatrix epsilon=eigensolver.eigenvalues().asDiagonal();
		EigenMatrix Cprime=eigensolver.eigenvectors();
		coefficients=X*Cprime;
		EigenMatrix C_occ=coefficients.leftCols(nocc);
		densitymatrix=(C_occ*C_occ.transpose()*4+densitymatrix)/5; // Mixing new density matrix with old one. This may improve robustness.
		EigenMatrix Iamgoingtobeanobellaureate=densitymatrix*(Hcore+F); // Intermediate matrix.
		energy=Iamgoingtobeanobellaureate.trace();
		end=clock();
		std::cout<<" Iteration "<<iiteration<<"  energy = "<<energy<<"  elapsed time = "<<double(end-start)/CLOCKS_PER_SEC<<" s"<<std::endl;
		iiteration++;
	}
	std::cout<<" Final RHF energy = "<<energy<<std::endl;
	std::cout<<" ... RHF done ..."<<std::endl;
}		

