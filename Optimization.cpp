#include <Eigen/Dense>
#include <iostream>

#define EigenMatrix Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>
#define EigenZero Eigen::MatrixXd::Zero
#define EigenOne Eigen::MatrixXd::Identity

#include "OSQP.h"

#define __DIIS_determinant_threshold__ -1.e-50
#define __minimum_DIIS_space__ 5
#define __LBFGS_convergence_threshold__ -1.e-50

void PushMatrixQueue(EigenMatrix M,EigenMatrix * Ms,int size){
        Ms[size-1].resize(0,0);
        for (int i=size-1;i>0;i--)
                Ms[i]=Ms[i-1];
        Ms[0]=M;
}

void PushDoubleQueue(double M,double * Ms,int size){
        for (int i=size-1;i>0;i--)
                Ms[i]=Ms[i-1];
        Ms[0]=M;
}

EigenMatrix DIIS(EigenMatrix * Ds,EigenMatrix * Es,int maxsize,double & error2norm){
	double determinant=0;
	int size=maxsize;
	EigenMatrix B(size+1,size+1);
	EigenMatrix b(size+1,1);
	do{
		B=EigenZero(size+1,size+1);
		B(size,size)=0;
		b=EigenZero(size+1,1);
		b(size,0)=-1;
		for (int i=0;i<size;i++){
			for (int j=0;j<=i;j++){
				EigenMatrix bij=Es[i].transpose()*Es[j];
				B(i,j)=bij.trace();
				B(j,i)=bij.trace();
			}
			B(i,size)=-1;
			B(size,i)=-1;
			b(i,0)=0;
		}
		determinant=B.block(0,0,size,size).determinant();
		size--;
	}while (size>__minimum_DIIS_space__&&abs(determinant)<__DIIS_determinant_threshold__);
	size++;
	EigenMatrix x=B.colPivHouseholderQr().solve(b);
	EigenMatrix D=EigenZero(Ds[0].rows(),Ds[0].cols());
	for (int i=0;i<size;i++)
		D+=x(i,0)*Ds[i];
	error2norm=0;
	for (int i=0;i<size;i++)
		for (int j=0;j<=i;j++)
			error2norm+=(i==j)?x(i,0)*x(j,0)*B(i,j):2*x(i,0)*x(j,0)*B(i,j);
	return D;
}

void Dense2CSC(EigenMatrix dense,bool sym,double * elements,int * rowindeces,int * colpointers){
	double * element=elements;
	int * rowindex=rowindeces;
	int * colpointer=colpointers;
	for (int j=0;j<dense.cols();j++){
		*(colpointer++)=sym?(1+j)*j/2:dense.rows()*j;
		for (int i=0;sym?(i<=j):(i<dense.rows());i++){
			*(element++)=dense(i,j);
			*(rowindex++)=i;
		}
	}
	*colpointer=sym?((1+dense.cols())*dense.cols()/2):(dense.rows()*dense.cols());
}

EigenMatrix AEDIIS(char diistype,double * Es,EigenMatrix * Ds,EigenMatrix * Fs,int size){
	
	int nconstraints=size+1; 
	EigenMatrix h(size,size);
	EigenMatrix a=EigenZero(nconstraints,size);
	for (int i=0;i<size;i++){
		for (int j=0;j<=i;j++){
			EigenMatrix hij;
			if (diistype=='e')
				hij=-2*(Ds[i].transpose()-Ds[j].transpose())*(Fs[i]-Fs[j]);
			else if (diistype=='a')
				hij=2*(Ds[i].transpose()-Ds[0].transpose())*(Fs[j]-Fs[0]); // This matrix is naturally symmetric. I have no idea why. Maths is elusive.
			h(i,j)=hij.trace();
			h(j,i)=hij.trace();
		}
		a(0,i)=1;
		a(i+1,i)=1;
	}

	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver; // QP demands the matrix be positive semi-definite. Finding the nearest PSD matrix.
	eigensolver.compute(h);
	const EigenMatrix lambdas=eigensolver.eigenvalues();
	EigenMatrix new_lambdas=EigenZero(size,size);
	for (int i=0;i<size;i++){
		if (lambdas(i)>0) new_lambdas(i,i)=lambdas(i);
		else new_lambdas(i,i)=0;
	}
	h=eigensolver.eigenvectors()*new_lambdas*eigensolver.eigenvectors().transpose();

	int hnnz=(size+1)*size/2;
	double * helements=new double[hnnz];
	int * hrowindeces=new int[hnnz];
	int * hcolpointers=new int[size+1];

	int annz=nconstraints*size;
	double * aelements=new double[annz];
	int * arowindeces=new int[annz];
	int * acolpointers=new int[size+1];

	Dense2CSC(h,1,helements,hrowindeces,hcolpointers);
	Dense2CSC(a,0,aelements,arowindeces,acolpointers);

	double * l=new double[nconstraints];
	double * u=new double[nconstraints];
	for (int i=0;i<nconstraints;i++){
		l[i]=0;
		u[i]=1;
	}
	l[0]=1;
	double * x=new double[size];
	for (int i=0;i<size;i++) x[i]=0;

	if (diistype=='e')
		QuadraticProgramming(size,helements,hrowindeces,hcolpointers,Es,aelements,arowindeces,acolpointers,nconstraints,l,u,x);
	else if (diistype=='a'){
		double * g=new double[size];
		for (int i=0;i<size;i++){
			EigenMatrix gi=(Ds[i].transpose()-Ds[0].transpose())*Fs[0];
			g[i]=2*gi.trace();
		}
		QuadraticProgramming(size,helements,hrowindeces,hcolpointers,g,aelements,arowindeces,acolpointers,nconstraints,l,u,x);
		delete [] g;
	}

	EigenMatrix F=Fs[0]*0;
	for (int i=0;i<size;i++)
		F+=Fs[i]*x[i];

	delete [] helements;
	delete [] hrowindeces;
	delete [] hcolpointers;
	delete [] aelements;
	delete [] arowindeces;
	delete [] acolpointers;
	delete [] l;
	delete [] u;
	delete [] x;
	return F;
}


/*
#include <iostream>
int main(){ // Testing OSQP related functions.
	int size=2;
	int nconstraints=3;
	EigenMatrix h(size,size);h<<4,1,
	                            1,2;
	EigenMatrix a(nconstraints,size);a<<1,1,
	                                    1,0,
	                                    0,1;
	double g[]={1,1};
	double l[]={1,0,0};
	double u[]={1,0.7,0.7};

	int hnnz=(1+h.cols())*h.cols()/2;
	double helements[hnnz]={0};
	int hrowindeces[hnnz]={0};
	int hcolpointers[h.cols()+1]={0};

	int annz=a.rows()*a.cols();
	double aelements[annz]={0};
	int arowindeces[annz]={0};
	int acolpointers[a.cols()+1]={0};

	Dense2CSC(h,1,helements,hrowindeces,hcolpointers);
	Dense2CSC(a,0,aelements,arowindeces,acolpointers);

	std::cout<<"H ="<<std::endl;
	std::cout<<h<<std::endl;
	std::cout<<"H elements = "<<std::endl;
	for (int i=0;i<hnnz;i++)
		std::cout<<helements[i]<<",";
	std::cout<<std::endl;
	std::cout<<"H row indeces = "<<std::endl;
	for (int i=0;i<hnnz;i++)
		std::cout<<hrowindeces[i]<<",";
	std::cout<<std::endl;
	std::cout<<"H column pointers = "<<std::endl;
	for (int i=0;i<h.cols()+1;i++)
		std::cout<<hcolpointers[i]<<",";
	std::cout<<std::endl;

	std::cout<<"A ="<<std::endl;
	std::cout<<a<<std::endl;
	std::cout<<"A elements = ";
	for (int i=0;i<annz;i++)
		std::cout<<aelements[i]<<",";
	std::cout<<std::endl;
	std::cout<<"A row indeces = ";
	for (int i=0;i<annz;i++)
		std::cout<<arowindeces[i]<<",";
	std::cout<<std::endl;
	std::cout<<"A column pointers = "<<std::endl;
	for (int i=0;i<a.cols()+1;i++)
		std::cout<<acolpointers[i]<<",";
	std::cout<<std::endl;

	double x[size];
	QuadraticProgramming(size,helements,hrowindeces,hcolpointers,g,aelements,arowindeces,acolpointers,nconstraints,l,u,x);
	std::cout<<"Solution = ";
	for (int i=0;i<size;i++)
		std::cout<<x[i]<<",";
	std::cout<<std::endl;

	return 0;
}
*/

#define __Clean_Arrays__\
	for (int j=0;j<size;j++){\
		Ys[j].resize(0,0);\
		Ss[j].resize(0,0);\
	}\
	delete [] Ys;\
	delete [] Ss;\
	delete [] Rs;\
	delete [] As;

EigenMatrix LBFGS(EigenMatrix * pGs,EigenMatrix * pXs,int size,EigenMatrix hessiandiag){ // pGs - packed gradients
	EigenMatrix * Ys=new EigenMatrix[size];
	EigenMatrix * Ss=new EigenMatrix[size];
	for (int i=0;i<size;i++){
		Ys[i]=pGs[i]-pGs[i+1];
		Ss[i]=pXs[i]-pXs[i+1];
	}
	EigenMatrix Y=Ys[0];
	EigenMatrix S=Ss[0];
	EigenMatrix q=pGs[0];
	double * Rs=new double[size];
	double * As=new double[size];
	EigenMatrix intermediate1;
	EigenMatrix intermediate2;

	intermediate1=Ys[0].transpose()*Ss[0];
	if (intermediate1(0,0)*intermediate1(0,0)<__LBFGS_convergence_threshold__*__LBFGS_convergence_threshold__){
		__Clean_Arrays__
//std::cout<<"fuck"<<intermediate1(0,0)<<std::endl;
		return pXs[0]*0;
	}

	for (int i=0;i<size;i++){
		intermediate1=Ys[i].transpose()*Ss[i];
		Rs[i]=1/intermediate1(0,0);
		intermediate1=Rs[i]*Ss[i].transpose()*q;
		As[i]=intermediate1(0,0);
		q-=As[i]*Ys[i];
	}
	intermediate1=Ss[0].transpose()*Ys[0];
	intermediate2=Ys[0].transpose()*Ys[0];
	double r=intermediate1(0,0)/intermediate2(0,0);
	EigenMatrix z;
	if (hessiandiag(0)==0) z=-r*q;
	else{
		intermediate1=-hessiandiag.cwiseInverse();
		z=intermediate1.cwiseProduct(q);
	}	
	for (int i=size-1;i>=0;i--){
		intermediate1=-Rs[i]*Ys[i].transpose()*z;
		z-=Ss[i]*(As[i]-intermediate1(0,0));
	}
	__Clean_Arrays__
	return z;
}

/*
// A test for L-BFGS
// f(x1,x2)=(x1-2)^2+(x2-3)^2+x1*x2
// df/dx1(x1,x2)=2*(x1-2)+x2
// df/dx2(x1,x2)=2*(x2-3)+x1
// Solution: fmin=f(2/3,8/3)=11/3
#include <iostream>
int main(){
	int size=3;
	int dimension=2;
	EigenMatrix * Xs=new EigenMatrix[size+1];
	EigenMatrix * Gs=new EigenMatrix[size+1];
	EigenMatrix X(2,1);
	EigenMatrix G(2,1);
	for (int i=size;i>=0;i--){
		X(0)=17*i;X(1)=7*i;
		G(0)=2*(X(0)-2)+X(1);G(1)=2*(X(1)-3)+X(0);
		Xs[i]=X;
		Gs[i]=G;
	}
	std::cout<<"Iteration 0: current x = "<<X.transpose()<<" current f = "<<(X(0)-2)*(X(0)-2)+(X(1)-3)*(X(1)-3)+X(0)*X(1)<<std::endl;
	for (int iter=1;iter<10;iter++){
		EigenMatrix hessiandiag;
		if (iter==1){
			EigenMatrix tmp=EigenOne(dimension,dimension);
			hessiandiag=tmp.diagonal();
		}else hessiandiag=EigenZero(dimension,1);
		X+=LBFGS(Gs,Xs,size<iter+1?size:iter+1,hessiandiag);
		std::cout<<"Iteration "<<iter<<": current x = "<<X.transpose()<<" current f = "<<(X(0)-2)*(X(0)-2)+(X(1)-3)*(X(1)-3)+X(0)*X(1)<<std::endl;
		G(0)=2*(X(0)-2)+X(1);
		G(1)=2*(X(1)-3)+X(0);
		PushMatrixQueue(X,Xs,size+1);
		PushMatrixQueue(G,Gs,size+1);
	}
	return 0;
}
*/

#define __small_constant__ 1.e-7
EigenMatrix AdaGrad(EigenMatrix * pGs,int size){
	EigenMatrix r=pGs[0]*0;
	for (int i=0;i<size;i++)
		r+=pGs[i].cwiseProduct(pGs[i]);
	const EigenMatrix a=1/(__small_constant__+r.array().sqrt());
	return -a.cwiseProduct(pGs[0]);
}

