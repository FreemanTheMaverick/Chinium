#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <omp.h>

#define EigenMatrix Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>
#define EigenZero Eigen::MatrixXd::Zero
#define EigenOne Eigen::MatrixXd::Identity

#include "Optimization.h"

#define __damping_start_threshold__ 1.
#define __damping_factor__ 0.25
#define __diis_start_iter__ 3
#define __diis_space_size__ 6
#define __asoscf_start_iter__ 100
#define __lbfgs_space_size__ 6
#define __trah_start_iter__ 50
#define __scf_convergence_energy_threshold__ 1.e-8
#define __scf_convergence_gradient_threshold__ 1.e-5

EigenMatrix GMatrix(double * repulsion,short int * indices,int n2integrals,EigenMatrix density,const int nprocs){
	int nbasis=density.cols();
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
		bigrawjmatrix[iproc]=EigenZero(nbasis,nbasis);
		bigrawkmatrix[iproc]=EigenZero(nbasis,nbasis);
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
			bigrawjmatrix[iproc](a,b)+=density(c,d)*deg_value;
			bigrawjmatrix[iproc](c,d)+=density(a,b)*deg_value;
			bigrawkmatrix[iproc](a,c)+=0.5*density(b,d)*deg_value;
			bigrawkmatrix[iproc](b,d)+=0.5*density(a,c)*deg_value;
			bigrawkmatrix[iproc](a,d)+=0.5*density(b,c)*deg_value;
			bigrawkmatrix[iproc](b,c)+=0.5*density(a,d)*deg_value;
		}
	}
	EigenMatrix rawjmatrix=EigenZero(nbasis,nbasis);
	EigenMatrix rawkmatrix=EigenZero(nbasis,nbasis);
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

EigenMatrix ElectronicGradient_scf(int nocc,EigenMatrix F,EigenMatrix C){
	int nbasis=F.rows();
	EigenMatrix Fmo=C.transpose()*F*C;
	EigenMatrix g(nocc*(nbasis-nocc),1);
	for (int o=0,k=0;o<nocc;o++) // Occupied orbitals.
		for (int v=nocc;v<nbasis;v++,k++) // Virtual orbitals.
			g(k,0)=-4*Fmo(o,v);
	return g;
}

double RHF(int nele,EigenMatrix overlap,EigenMatrix hcore,double * repulsion,short int * indices,long int n2integrals,EigenMatrix & orbitalenergies,EigenMatrix & coefficients,EigenMatrix & density,const int nprocs,const bool output){
	if (output) std::cout<<"Restricted Hartree-Fock ..."<<std::endl;

	// General preparation
	const int nbasis=overlap.cols();
	const int nocc=nele/2;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(overlap);
	const EigenMatrix s=eigensolver.eigenvalues();
	EigenMatrix sinversesqrt=Eigen::MatrixXd::Zero(nbasis,nbasis);
	for (int i=0;i<nbasis;i++){
		sinversesqrt(i,i)=1/sqrt(s(i,0));
	}
	const EigenMatrix U=eigensolver.eigenvectors();
	const EigenMatrix X=U*sinversesqrt*U.transpose();
	EigenMatrix F=hcore+GMatrix(repulsion,indices,n2integrals,density,nprocs); // Fock matrix.
	double energy=114514;
	double lastenergy=1919810;
	EigenMatrix gradient; // Electronic gradient of energy with respect to nonredundant orbital rotational parameters.
	int iiteration=0;

	// Pulay's DIIS preparation
	EigenMatrix * Fs=new EigenMatrix[__diis_space_size__]; // Storing the last __diis_space_size__ density matrices and their error matrices.
	EigenMatrix * Es=new EigenMatrix[__diis_space_size__];
	for (int i=0;i<__diis_space_size__;i++){
		Fs[i]=F;
		Es[i]=EigenZero(nbasis,nbasis);
	}
	double error2norm=-889464; // |e|^2=Sigma_ij(c_i*c_j*e_i*e_j)

	// L-BFGS preparation
	EigenMatrix * Ss=new EigenMatrix[__lbfgs_space_size__];
	EigenMatrix * Ys=new EigenMatrix[__lbfgs_space_size__];
	for (int i=0;i<__lbfgs_space_size__;i++){
		Ss[i]=EigenZero(nocc*(nbasis-nocc),1);
		Ys[i]=EigenZero(nocc*(nbasis-nocc),1);
	}

	// RHF-SCF iterations
	while (abs(lastenergy-energy)>__scf_convergence_energy_threshold__ || gradient.norm()>__scf_convergence_gradient_threshold__){ // Normal RHF SCF procedure.
		if (output) std::cout<<" Iteration "<<iiteration<<":  ";
		const clock_t iterstart=clock();

		// Convergence techniques
		if (iiteration==0){
			if (output) std::cout<<"fock_update = naive  ";
			F=Fs[0];
		}else if (abs(lastenergy-energy)>__damping_start_threshold__){ // Using damping in the beginning and when energy oscillates in a large number of iterations.
			if (output) std::cout<<"fock_update = damping  ";
			F=(1.-__damping_factor__)*Fs[0]+__damping_factor__*Fs[1];
		}else if (iiteration>__diis_start_iter__){ // Starting DIIS after Fs and Es are filled and error2norm is not too large.
			if (output) std::cout<<"fock_update = Pulay's_DIIS  ";
			F=DIIS(Fs,Es,iiteration<__diis_space_size__?iiteration:__diis_space_size__,error2norm); // error2norm is updated.
		}else{
			if (output) std::cout<<"fock_update = naive  ";
			F=Fs[0];
		}

		// Common procedures necessary for all convergence techniques
		const EigenMatrix Fprime=X.transpose()*F*X;
		eigensolver.compute(Fprime);
		orbitalenergies=eigensolver.eigenvalues();
		const EigenMatrix Cprime=eigensolver.eigenvectors();
		coefficients=X*Cprime;
		const EigenMatrix C_occ=coefficients.leftCols(nocc);
		density=C_occ*C_occ.transpose();
		F=hcore+GMatrix(repulsion,indices,n2integrals,density,nprocs);
		gradient=4*(overlap*density*F-F*density*overlap); // [F(D),D] instead of [F,D(F)].
		PushQueue(F,Fs,__diis_space_size__); // The oldest density matrix and its error matrix are replaced by the latest ones.
		PushQueue(sinversesqrt.transpose()*gradient*sinversesqrt,Es,__diis_space_size__); // I don't know why sinversesqrt is important, but its existence accelerates convergence.

		// Iteration information output
		const EigenMatrix Iamgoingtobeanobellaureate=density*(hcore+F); // Intermediate matrix.
		lastenergy=energy;
		energy=Iamgoingtobeanobellaureate.trace();
		const clock_t iterend=clock();
		if (output) std::cout<<"energy = "<<std::setprecision(12)<<energy<<std::setprecision(3)<<" a.u.  ||gradient|| = "<<gradient.norm()<<" a.u.  elapsed_time = "<<double(iterend-iterstart)/CLOCKS_PER_SEC<<" s"<<std::endl;
		iiteration++;
	}
	delete [] Fs;
	delete [] Es;
	delete [] Ss;
	delete [] Ys;
	if (output) std::cout<<"Done; Final RHF energy = "<<std::setprecision(12)<<energy<<" a.u."<<std::endl;
	return energy;
}
