#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <omp.h>

#define __convergence_threshold__ 1.e-8
#define __damping_factor__ 0.

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;

EigenMatrix GMatrix(double * repulsion,short int * indices,int n2integrals,EigenMatrix densitymatrix,const int nprocs,const bool output){
	int nbasis=densitymatrix.cols();
	EigenMatrix zeromatrix(nbasis,nbasis);zeromatrix=zeromatrix*0;
	if (output) std::cout<<"Spawning "<<nprocs<<" threads in G matrix formation  ";
	omp_set_num_threads(nprocs);
	long int nintsperthread_fewer=n2integrals/nprocs;
	int ntimes_fewer=nprocs-n2integrals+nintsperthread_fewer*nprocs; // How many integrals a thread will handle. If the average number is A, the number of each thread is either a or (a+1), where a=floor(A). The number of threads to handle (a) integrals, x, and that to compute (a+1) integrals, y, can be obtained by solving (1) a*x+(a+1)*y=b and (2) x+y=c, where b and c stand for the total numbers of integrals and threads respectively.
	long int iintfirstperthread[nprocs];
	long int nintsperthread[nprocs];
	for (int iproc=0;iproc<nprocs;iproc++){
		iintfirstperthread[iproc]=iproc==0?0:(iintfirstperthread[iproc-1]+nintsperthread[iproc-1]);
		nintsperthread[iproc]=iproc<ntimes_fewer?nintsperthread_fewer:(nintsperthread_fewer+1);
	}
	EigenMatrix * bigrawjmatrix=new EigenMatrix[nprocs];
	EigenMatrix * bigrawkmatrix=new EigenMatrix[nprocs];
	for (int iproc=0;iproc<nprocs;iproc++){
		bigrawjmatrix[iproc]=zeromatrix;
		bigrawkmatrix[iproc]=zeromatrix;
	}
	#pragma omp parallel for
	for (int iproc=0;iproc<nprocs;iproc++){
		long int nints=nintsperthread[iproc];
		long int iintfirst=iintfirstperthread[iproc];
		double * repulsionranger=repulsion+iintfirst;
		short int * indicesranger=indices+iintfirst*5;
		for (int i=0;i<nints;i++){
			const short int a=*(indicesranger++); // Moving the ranger pointer to the right, where the next index is located.
			const short int b=*(indicesranger++);
			const short int c=*(indicesranger++);
			const short int d=*(indicesranger++); // Moving the ranger pointer to the right, where the degeneracy factor is located
			const short int deg=*(indicesranger++);
			const double value=*(repulsionranger++); // Moving the ranger pointer to the right, where the next integral is located.
			const double deg_value=deg*value;
			bigrawjmatrix[iproc](a,b)+=densitymatrix(c,d)*deg_value;
			bigrawjmatrix[iproc](c,d)+=densitymatrix(a,b)*deg_value;
			bigrawkmatrix[iproc](a,c)+=0.5*densitymatrix(b,d)*deg_value;
			bigrawkmatrix[iproc](b,d)+=0.5*densitymatrix(a,c)*deg_value;
			bigrawkmatrix[iproc](a,d)+=0.5*densitymatrix(b,c)*deg_value;
			bigrawkmatrix[iproc](b,c)+=0.5*densitymatrix(a,d)*deg_value;
		}
	}
	EigenMatrix rawjmatrix=zeromatrix;
	EigenMatrix rawkmatrix=zeromatrix;
	for (int iproc=0;iproc<nprocs;iproc++){
		rawjmatrix=rawjmatrix+bigrawjmatrix[iproc];
		rawkmatrix=rawkmatrix+bigrawkmatrix[iproc];
	}
	delete [] bigrawjmatrix;
	delete [] bigrawkmatrix;
	EigenMatrix jmatrix=0.5*(rawjmatrix+rawjmatrix.transpose());
	EigenMatrix kmatrix=0.5*(rawkmatrix+rawkmatrix.transpose());
	EigenMatrix gmatrix=jmatrix-0.5*kmatrix;
	return gmatrix;
}

double RHF(int nele,EigenMatrix overlap,EigenMatrix hcore,double * repulsion,short int * indices,long int n2integrals,EigenMatrix & orbitalenergies,EigenMatrix & coefficients,EigenMatrix & densitymatrix,const int nprocs,const bool output){
	if (output) std::cout<<"Restricted Hartree-Fock ..."<<std::endl;
	int nocc=nele/2;
	int nbasis=overlap.cols();
	EigenMatrix C_occ=coefficients.leftCols(nocc); // Normal Hartree-Fock procedure.
	densitymatrix=C_occ*C_occ.transpose();
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(overlap);
	EigenMatrix s=eigensolver.eigenvalues();
	EigenMatrix sinversesqrt=Eigen::MatrixXd::Zero(nbasis,nbasis);
	for (int i=0;i<nbasis;i++){
		sinversesqrt(i,i)=1/sqrt(s(i,0));
	}
	EigenMatrix U=eigensolver.eigenvectors();
	EigenMatrix X=U*sinversesqrt;
	double energy=114514;
	double lastenergy=1919810;
	int iiteration=0;
	while (abs(lastenergy-energy)>__convergence_threshold__){ // Normal HF SCF procedure.
		if (output) std::cout<<" Iteration "<<iiteration<<"  ";
		clock_t iterstart=clock();
		lastenergy=energy;
		EigenMatrix G=GMatrix(repulsion,indices,n2integrals,densitymatrix,nprocs,output);
		EigenMatrix F=hcore+G;
		EigenMatrix Fprime=X.transpose()*F*X;
		eigensolver.compute(Fprime);
		orbitalenergies=eigensolver.eigenvalues();
		EigenMatrix Cprime=eigensolver.eigenvectors();
		coefficients=X*Cprime;
		EigenMatrix C_occ=coefficients.leftCols(nocc);
		densitymatrix=C_occ*C_occ.transpose()*(1-__damping_factor__)+densitymatrix*__damping_factor__; // Mixing new density matrix with old one. This may improve robustness.
		EigenMatrix Iamgoingtobeanobellaureate=densitymatrix*(hcore+F); // Intermediate matrix.
		energy=Iamgoingtobeanobellaureate.trace();
		clock_t iterend=clock();
		if (output) std::cout<<"energy = "<<std::setprecision(12)<<energy<<" a.u."<<"  elapsed_time = "<<std::setprecision(3)<<double(iterend-iterstart)/CLOCKS_PER_SEC<<" s"<<std::endl;
		iiteration++;
	}
	if (output) std::cout<<"Done; Final RHF energy = "<<std::setprecision(12)<<energy<<" a.u."<<std::endl;
	return energy;
}
