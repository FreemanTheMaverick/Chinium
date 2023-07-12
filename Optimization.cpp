#include <Eigen/Dense>
#include <iostream>
#include "Aliases.h"
#include "OSQP.h"

#define __DIIS_determinant_threshold__ -1.e-50
#define __minimum_DIIS_space__ 5
#define __LBFGS_convergence_threshold__ -1.e-50

#define __Push_Queue__\
	for (int i=size-1;i>0;i--)\
		Ms[i]=Ms[i-1];\
	Ms[0]=M;

void PushVectorQueue(EigenVector M,EigenVector * Ms,int size){
	Ms[size-1].resize(0);
	__Push_Queue__
}

void PushMatrixQueue(EigenMatrix M,EigenMatrix * Ms,int size){
	Ms[size-1].resize(0,0);
	__Push_Queue__
}

void PushDoubleQueue(double M,double * Ms,int size){
	__Push_Queue__
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
		Ys[j].resize(0);\
		Ss[j].resize(0);\
	}\
	delete [] Ys;\
	delete [] Ss;\
	delete [] Rs;\
	delete [] As;

EigenVector LBFGS(EigenVector * pGs,EigenVector * pXs,int size,EigenVector hessiandiag){ // pGs - packed gradients
	EigenVector * Ys=new EigenVector[size];
	EigenVector * Ss=new EigenVector[size];
	for (int i=0;i<size;i++){
		Ys[i]=pGs[i]-pGs[i+1];
		Ss[i]=pXs[i]-pXs[i+1];
	}
	EigenVector Y=Ys[0];
	EigenVector S=Ss[0];
	EigenVector q=pGs[0];
	double * Rs=new double[size];
	double * As=new double[size];

	if (abs(Ys[0].dot(Ss[0]))<__LBFGS_convergence_threshold__){
		__Clean_Arrays__
		return pXs[0]*0;
	}

	for (int i=0;i<size;i++){
		Rs[i]=1/Ys[i].dot(Ss[i]);
		As[i]=Rs[i]*Ss[i].dot(q);
		q-=As[i]*Ys[i];
	}
	double r=Ss[0].dot(Ys[0])/Ys[0].dot(Ys[0]);
	EigenVector z;
	if (hessiandiag(0)==0) z=-r*q;
	else{
		const EigenVector inversehessiandiag=hessiandiag.cwiseInverse();
		z=-inversehessiandiag.cwiseProduct(q);
	}	
	for (int i=size-1;i>=0;i--)
		z-=Ss[i]*(As[i]+Rs[i]*Ys[i].dot(z));
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
	EigenVector * Xs=new EigenVector[size+1];
	EigenVector * Gs=new EigenVector[size+1];
	EigenVector X(2,1);
	EigenVector G(2,1);
	for (int i=size;i>=0;i--){
		X(0)=17*i;X(1)=7*i;
		G(0)=2*(X(0)-2)+X(1);G(1)=2*(X(1)-3)+X(0);
		Xs[i]=X;
		Gs[i]=G;
	}
	std::cout<<"Iteration 0: current x = "<<X.transpose()<<" current f = "<<(X(0)-2)*(X(0)-2)+(X(1)-3)*(X(1)-3)+X(0)*X(1)<<std::endl;
	for (int iter=1;iter<10;iter++){
		EigenVector hessiandiag;
		if (iter==1){
			EigenVector tmp=EigenOne(dimension,dimension);
			hessiandiag=tmp.diagonal();
		}else hessiandiag=EigenZero(dimension,1);
		X+=LBFGS(Gs,Xs,size<iter+1?size:iter+1,hessiandiag);
		std::cout<<"Iteration "<<iter<<": current x = "<<X.transpose()<<" current f = "<<(X(0)-2)*(X(0)-2)+(X(1)-3)*(X(1)-3)+X(0)*X(1)<<std::endl;
		G(0)=2*(X(0)-2)+X(1);
		G(1)=2*(X(1)-3)+X(0);
		PushVectorQueue(X,Xs,size+1);
		PushVectorQueue(G,Gs,size+1);
	}
	return 0;
}
*/

EigenVector FABFGS(EigenVector * pGs,EigenVector * pXs,int size,EigenVector hessiandiag){ // A variant of BFGS proposed in https://pubs.acs.org/doi/10.1021/j100203a036 .
	EigenVector w=pGs[0].cwiseProduct(hessiandiag.cwiseInverse());
	if (size==1) return -w;
	EigenVector * DELTAs=new EigenVector[size];
	EigenVector * Deltas=new EigenVector[size];
	EigenVector * Ys=new EigenVector[size];
	for (int i=0;i<size;i++){
		DELTAs[i]=pGs[i]-pGs[i+1];
		Deltas[i]=pXs[i]-pXs[i+1];
		Ys[i]=DELTAs[i].cwiseProduct(hessiandiag.cwiseInverse());
	}
	for (int i=size-1;0<i;i--){
		const double s1=1/Deltas[i].dot(DELTAs[i]);
		const double s2=1/DELTAs[i].dot(Ys[i]);
		const double s3=Deltas[i].dot(pGs[0]);
		const double s4=Ys[i].dot(pGs[0]);
		const double s5=Deltas[i].dot(DELTAs[0]);
		const double s6=Ys[i].dot(DELTAs[0]);
		const double t1=(1+s1/s2)*s1*s3-s1*s4;
		const double t2=s1*s3;
		const double t3=(1+s1/s2)*s1*s5-s1*s6;
		const double t4=s1*s5;
		w+=t1*Deltas[i]-t2*Ys[i];
		Ys[0]+=t3*Deltas[i]-t4*Ys[i];
	}
	const double s1=1/Deltas[0].dot(DELTAs[0]);
	const double s2=1/DELTAs[0].dot(Ys[0]);
	const double s3=Deltas[0].dot(pGs[0]);
	const double s4=Ys[0].dot(pGs[0]);
	const double t1=(1+s1/s2)*s1*s3-s1*s4;
	const double t2=s1*s3;
	w+=t1*Deltas[0]-t2*Ys[0];
	for (int i=0;i<size;i++){
		DELTAs[i].resize(0);
		Deltas[i].resize(0);
		Ys[i].resize(0);
	}
	return -w;
}

#define __small_constant__ 1.e-7
EigenMatrix AdaGrad(EigenMatrix * pGs,int size){
	EigenMatrix r=pGs[0]*0;
	for (int i=0;i<size;i++)
		r+=pGs[i].cwiseProduct(pGs[i]);
	const EigenMatrix a=1/(__small_constant__+r.array().sqrt());
	return -a.cwiseProduct(pGs[0]);
}
