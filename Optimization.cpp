#include <Eigen/Dense>

#define EigenMatrix Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>
#define EigenZero Eigen::MatrixXd::Zero
#define EigenOne Eigen::MatrixXd::Identity

extern "C"{
	#include "OSQP.h"
}

#define __DIIS_determinant_threshold__ 1.e-20
#define __minimum_DIIS_space__ 5

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
				hij=2*(Ds[i].transpose()-Ds[0].transpose())*(Fs[j]-Fs[0]);
			h(i,j)=hij.trace();
			h(j,i)=hij.trace();
		}
		a(0,i)=1;
		a(i+1,i)=1;
	}

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

/*
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
