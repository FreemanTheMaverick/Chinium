#include <Eigen/Dense>

#define EigenMatrix Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>
#define EigenZero Eigen::MatrixXd::Zero
#define EigenOne Eigen::MatrixXd::Identity

#define __DIIS_determinant_threshold__ 1.e-20
#define __minimum_DIIS_space__ 5

void PushQueue(EigenMatrix M,EigenMatrix * Ms,int size){
        Ms[size-1].resize(1,1);
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

EigenMatrix LBFGS(EigenMatrix g,EigenMatrix * Ss,EigenMatrix * Ys,int size,int latest){
	EigenMatrix q=g;
	double * Rs=new double[size];
	double * As=new double[size];
	EigenMatrix intermediate1;
	EigenMatrix intermediate2;
	for (int i=latest,j=0;j<size;i=(i==0?size-1:i-1),j++){
		intermediate1=Ys[i].transpose()*Ss[i];
		Rs[i]=1/intermediate1(0,0);
		intermediate1=Rs[i]*Ss[i].transpose()*q;
		As[i]=intermediate1(0,0);
		q-=As[i]*Ys[i];
	}
	intermediate1=Ss[latest].transpose()*Ys[latest];
	intermediate2=Ys[latest].transpose()*Ys[latest];
	double r=intermediate1(0,0)/intermediate2(0,0);
	EigenMatrix z=-r*q;
	for (int i=(latest==size-1?0:latest+1),j=0;j<size;i=(i==size-1?0:i+1),j++){
		intermediate1=-Rs[i]*Ys[i].transpose()*z;
		double b=intermediate1(0,0);
		z-=Ss[i]*(As[i]-b);
	}
	delete [] Rs;
	delete [] As;
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
	EigenMatrix * Xs=new EigenMatrix[4];
	EigenMatrix * Gs=new EigenMatrix[4];
	for (int i=0;i<4;i++){
		EigenMatrix X(2,1);X(0)=i;X(1)=i;
		EigenMatrix G=X;
		G(0)=2*(X(0)-2)+X(1);
		G(1)=2*(X(1)-3)+X(0);
		Xs[i]=X;
		Gs[i]=G;
	}
	EigenMatrix * Ss=new EigenMatrix[3];
	EigenMatrix * Ys=new EigenMatrix[3];
	for (int i=0;i<3;i++){
		Ss[i]=Xs[i+1]-Xs[i];
		Ys[i]=Gs[i+1]-Gs[i];
	}
	EigenMatrix lastx=Xs[2];
	EigenMatrix lastg=Gs[2];
	EigenMatrix x=lastx;
	EigenMatrix g=lastg;
	for (int iter=0;iter<10;iter++){
		x=lastx+LBFGS(lastg,Ss,Ys,3,(iter+2)%3);
		std::cout<<"Iteration "<<iter<<": current x = "<<x.transpose()<<" current f = "<<(x(0)-2)*(x(0)-2)+(x(1)-3)*(x(1)-3)+x(0)*x(1)<<std::endl;
		g(0)=2*(x(0)-2)+x(1);
		g(1)=2*(x(1)-3)+x(0);
		Ss[iter%3]=x-lastx;
		Ys[iter%3]=g-lastg;
		lastx=x;
		lastg=g;
	}
	return 0;
}
*/
