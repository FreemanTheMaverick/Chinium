#include <Eigen/Dense>

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;

EigenMatrix DIIS(EigenMatrix * Ds,EigenMatrix * Es,int size,double & error2norm){
	EigenMatrix B(size+1,size+1);
	B(size,size)=0;
	EigenMatrix b(size+1,1);
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
	EigenMatrix x=B.colPivHouseholderQr().solve(b);
	EigenMatrix D=Ds[0];
	for (int i=0;i<D.rows();i++)
		for (int j=0;j<D.cols();j++)
			D(i,j)=0;
	for (int i=0;i<size;i++)
		D=D+x(i,0)*Ds[i];
	error2norm=0;
	for (int i=0;i<size;i++)
		for (int j=0;j<=i;j++)
			error2norm+=(i==j)?x(i,0)*x(j,0)*B(i,j):2*x(i,0)*x(j,0)*B(i,j);
	return D;
}
/*
EigenMatrix LBFGS(EigenMatrix g,EigenMatrix * Ss,EigenMatrix * Ys,int size,int latest){
	EigenMatrix q=g;
	double * As=new double[size];
	for (int i=latest,j=0;j<size;i=(i==0?size-1:i-1),j++){
		As[i]=Ss[i].transpose()*q/(Ys[i].transpose()*Ss[i]);
		q-=As[i]*Ys[i];
	}
	double r=Ss[latest].transpose()*Ys[latest]/(Ys[latest].transpose()*Ys[latest]);
	EigenMatrix z=-r*q;
	for (int i=latest,j=0;j<size;i=(latest==size-1?0;i+1),j++){
		double b=-Ys[i].transpose()*z/(Ys[i].transpose()*Ss[i]);
		z-=Ss[i]*(As[i]-b);
	}
	return z;
}
*/
