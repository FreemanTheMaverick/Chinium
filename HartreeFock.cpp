#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include "Aliases.h"
#include "Optimization.h"

#define __damping_start_threshold__ 100.
#define __damping_factor__ 0.25
#define __adiis_start_iter__ 1
#define __diis_start_iter__ 4
#define __diis_space_size__ 6
#define __lbfgs_start_threshold__ -1.e-3
#define __lbfgs_space_size__ 10
#define __trah_start_iter__ 50
#define __scf_convergence_energy_threshold__ 1.e-8
#define __scf_convergence_gradient_threshold__ 1.e-5

EigenMatrix GMatrix(double * repulsion,short int * indices,int n2integrals,EigenMatrix density,const int nprocs){
	int nbasis=density.cols();
	omp_set_num_threads(nprocs);
	long int nintsperthread_fewer=n2integrals/nprocs;
	int ntimes_fewer=nprocs-n2integrals+nintsperthread_fewer*nprocs; // How many integrals a thread will handle. If the average number is A, the number of each thread is either a or (a+1), where a=floor(A). The number of threads to handle (a) integrals, x, and that to compute (a+1) integrals, y, can be obtained by solving (1) a*x+(a+1)*y=b and (2) x+y=c, where b and c stand for the total numbers of integrals and threads respectively.
	long int * iintfirstperthread=new long int[nprocs];
	long int * nintsperthread=new long int[nprocs];
	for (int iproc=0;iproc<nprocs;iproc++){
		iintfirstperthread[iproc]=iproc==0?0:(iintfirstperthread[iproc-1]+nintsperthread[iproc-1]);
		nintsperthread[iproc]=iproc<ntimes_fewer?nintsperthread_fewer:(nintsperthread_fewer+1);
	}
	EigenMatrix * rawjs=new EigenMatrix[nprocs];
	EigenMatrix * rawks=new EigenMatrix[nprocs];
	for (int iproc=0;iproc<nprocs;iproc++){
		rawjs[iproc]=EigenZero(nbasis,nbasis);
		rawks[iproc]=EigenZero(nbasis,nbasis);
	}
	#pragma omp parallel for
	for (int iproc=0;iproc<nprocs;iproc++){
		long int nints=nintsperthread[iproc];
		long int iintfirst=iintfirstperthread[iproc];
		double * repulsionranger=repulsion+iintfirst;
		short int * indicesranger=indices+iintfirst*5;
		short int a,b,c,d;
		double deg_value;

		// Without SIMD
		for (long int i=0;i<nints;i++){ // Manual loop unrolling does not help.
			a=*(indicesranger++); // Moving the ranger pointer to the right, where the next index is located.
			b=*(indicesranger++);
			c=*(indicesranger++);
			d=*(indicesranger++); // Moving the ranger pointer to the right, where the degeneracy factor is located.
			deg_value=*(indicesranger++)**(repulsionranger++); // Moving the ranger pointer to the right, where the next integral is located.
			rawjs[iproc](a,b)+=density(c,d)*deg_value;
			rawjs[iproc](c,d)+=density(a,b)*deg_value;
			rawks[iproc](a,c)+=density(b,d)*deg_value;
			rawks[iproc](b,d)+=density(a,c)*deg_value;
			rawks[iproc](a,d)+=density(b,c)*deg_value;
			rawks[iproc](b,c)+=density(a,d)*deg_value;
		}
	}
	EigenMatrix rawj=EigenZero(nbasis,nbasis);
	EigenMatrix rawk=EigenZero(nbasis,nbasis);
	for (int iproc=0;iproc<nprocs;iproc++){
		rawj+=rawjs[iproc];
		rawjs[iproc].resize(0,0);
		rawk+=rawks[iproc];
		rawks[iproc].resize(0,0);
	}
	delete [] iintfirstperthread;
	delete [] nintsperthread;
	delete [] rawjs;
	delete [] rawks;
	EigenMatrix j=0.5*(rawj+rawj.transpose());
	EigenMatrix k=0.25*(rawk+rawk.transpose());
	EigenMatrix g=j-0.5*k;
	return g;
}

#define __Fock_2_Density__\
	const EigenMatrix Fprime=X.transpose()*F*X;\
	eigensolver.compute(Fprime);\
	orbitalenergies=eigensolver.eigenvalues();\
	const EigenMatrix Cprime=eigensolver.eigenvectors();\
	coefficients=X*Cprime;\
	const EigenMatrix C_occ=coefficients.leftCols(nocc);\
	density=C_occ*C_occ.transpose();

#define __Loop_Over_OV__\
	for (int o=0,i=0;o<nocc;o++)\
		for (int v=nocc;v<nbasis;v++,i++)

#define __Print_GX__\
	EigenMatrix x(pX.rows(),__lbfgs_space_size__+1);\
	EigenMatrix g(pG.rows(),__lbfgs_space_size__+1);\
	for (int i=0;i<__lbfgs_space_size__+1;i++){\
		x.col(i)=pXs[i];\
		g.col(i)=pGs[i];\
	}\
	std::cout.unsetf(std::ios::fixed);\
	std::cout<<std::endl;\
	std::cout<<'x'<<std::endl;\
	std::cout<<x<<std::endl;\
	std::cout<<'g'<<std::endl;\
	std::cout<<g<<std::endl;

double RHF(int nele,EigenMatrix overlap,EigenMatrix hcore,double * repulsion,short int * indices,long int n2integrals,EigenVector & orbitalenergies,EigenMatrix & coefficients,EigenMatrix & density,const int nprocs,const bool output){
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
	EigenMatrix G=4*(overlap*density*F-F*density*overlap); // Electronic gradient of energy with respect to nonredundant orbital rotational parameters.
	char update='f';
	int iiteration=0;

	// DIIS preparation
	EigenMatrix * Fs=new EigenMatrix[__diis_space_size__]; // Storing the last __diis_space_size__ density matrices and their error matrices.
	EigenMatrix * Gs=new EigenMatrix[__diis_space_size__];
	EigenMatrix * Ds=new EigenMatrix[__diis_space_size__];
	double * Es=new double[__diis_space_size__]; // Energy.
	for (int i=0;i<__diis_space_size__;i++){
		Fs[i]=F;
		Gs[i]=G;
		Ds[i]=density;
		Es[i]=114514;
	}
	double error2norm=-889464; // |e|^2=Sigma_ij(c_i*c_j*e_i*e_j)

	// L-BFGS preparation
	EigenVector pG(nocc*(nbasis-nocc)); // Packed form of gradient matrix and NR parametres.
	EigenVector pX(nocc*(nbasis-nocc));pX.setZero();
	EigenVector * pGs=new EigenVector[__lbfgs_space_size__+1];
	EigenVector * pXs=new EigenVector[__lbfgs_space_size__+1];
	for (int i=0;i<__lbfgs_space_size__;i++){
		pGs[i]=pG;
		pXs[i]=pX;
	}
	EigenMatrix firstcoefficients;
	int nlbfgs=-1;

	// RHF-SCF iterations
	do{
		if (output) std::cout<<" Iteration "<<iiteration<<":  ";
		const clock_t iterstart_cpu=clock();
		const auto iterstart_wall=std::chrono::system_clock::now();

		// Convergence techniques
		if (iiteration==0 && nlbfgs==-1){
			if (output) std::cout<<"fock_update = naive  ";
			update='f';
			F=Fs[0];
		}else if (abs(Es[0]-Es[1])>__damping_start_threshold__ && nlbfgs==-1){ // Using damping in the beginning and when energy oscillates in a large number of iterations.
			if (output) std::cout<<"fock_update = damping  ";
			update='f';
			F=(1.-__damping_factor__)*Fs[0]+__damping_factor__*Fs[1];
		}else if (__diis_start_iter__>=iiteration && iiteration>__adiis_start_iter__ && nlbfgs==-1){ // Starting A-DIIS in the beginning to facilitate (but not necesarily accelerate) convergence.
			if (output) std::cout<<"fock_update = ADIIS  ";
			update='f';
			F=AEDIIS('a',Es,Ds,Fs,iiteration<__diis_space_size__?iiteration:__diis_space_size__);
		}else if (iiteration>__diis_start_iter__ && pG.norm()>__lbfgs_start_threshold__ && nlbfgs==-1){ // Starting DIIS after A-DIIS to accelerate convergence in the medium-gradient area.
			if (output) std::cout<<"fock_update = Pulay's_DIIS  ";
			update='f';
			F=DIIS(Fs,Gs,iiteration<__diis_space_size__?iiteration:__diis_space_size__,error2norm); // error2norm is updated.
		}else if ((iiteration>__diis_start_iter__ && pG.norm()<__lbfgs_start_threshold__) || nlbfgs>=0){ // Stopping DIIS for ASOSCF (or simply L-BFGS) in the final part to prevent trailing.
			EigenVector hessiandiag(nocc*(nbasis-nocc));
			hessiandiag.setZero();
			if (output) std::cout<<"density_update = L-BFGS  ";
			update='d';
			if (nlbfgs==-1)
				firstcoefficients=coefficients;
			if (nlbfgs<2)
				__Loop_Over_OV__
					hessiandiag(i)=4*(orbitalenergies(v)-orbitalenergies(o));
			if (nlbfgs<2) pX-=pG.cwiseProduct(hessiandiag.cwiseInverse());
			else pX+=LBFGS(pGs,pXs,nlbfgs+1<__lbfgs_space_size__?nlbfgs+1:__lbfgs_space_size__,hessiandiag);
			EigenMatrix A=EigenZero(nbasis,nbasis);
			__Loop_Over_OV__{
				A(o,v)=pX(i);
				A(v,o)=-pX(i);
			}
			coefficients=firstcoefficients*(EigenOne(nbasis,nbasis)+A+0.5*A*A+0.16666666666666666667*A*A*A+0.04166666666666666667*A*A*A*A);
			const EigenMatrix C_occ=coefficients.leftCols(nocc);
			density=C_occ*C_occ.transpose();
			nlbfgs++;
		}else{
			if (output) std::cout<<"fock_update = naive  ";
			update='f';
			F=Fs[0];
		}

		if (update=='f'){__Fock_2_Density__} // Some techniques update Fock matrix. To complete the iteration, we obtain density matrix from it.

		// Common procedures necessary for all convergence techniques
		F=hcore+GMatrix(repulsion,indices,n2integrals,density,nprocs);
		G=4*(overlap*density*F-F*density*overlap); // [F(D),D] instead of [F,D(F)].
		PushMatrixQueue(F,Fs,__diis_space_size__); // Updating Fock matrix.
		PushMatrixQueue(sinversesqrt.transpose()*G*sinversesqrt,Gs,__diis_space_size__); // Updating gradient. I don't know why sinversesqrt is important, but its existence accelerates convergence.
		PushMatrixQueue(density,Ds,__diis_space_size__); // Updating atomic density matrix.

		const EigenMatrix Fmo=coefficients.transpose()*F*coefficients;
		__Loop_Over_OV__ pG(i)=-4*Fmo(o,v);
		PushVectorQueue(pG,pGs,__lbfgs_space_size__+1); // Updating packed gradient matrix.
		PushVectorQueue(pX,pXs,__lbfgs_space_size__+1);

		if (update=='d'){__Fock_2_Density__} // Some techniques update Fock matrix. To complete the iteration, we obtain density matrix from it.

		// Iteration information output
		const EigenMatrix Iamgoingtobeanobellaureate=density*(hcore+F); // Intermediate matrix.
		PushDoubleQueue(Iamgoingtobeanobellaureate.trace(),Es,__diis_space_size__); // Updating energy.
		const std::chrono::duration<double> duration_wall=std::chrono::system_clock::now()-iterstart_wall;
		if (output){
			std::cout.setf(std::ios::fixed);
			std::cout<<"energy = "<<std::setprecision(12)<<Es[0];
			std::cout.unsetf(std::ios::fixed);
			std::cout<<std::setprecision(3)<<" a.u.  ||gradient|| = "<<pG.norm()<<" a.u.  ";
			std::cout.setf(std::ios::fixed);
			std::cout<<"cpu_time = "<<double(clock()-iterstart_cpu)/CLOCKS_PER_SEC;
			std::cout<<" s  wall_time = "<<duration_wall.count()<<" s"<<std::endl;
		}
		iiteration++;
	}while (abs(Es[0]-Es[1])>__scf_convergence_energy_threshold__ || pG.norm()>__scf_convergence_gradient_threshold__);
	if (output) std::cout<<"Done; Final RHF energy = "<<std::setprecision(12)<<Es[0]<<" a.u."<<std::endl;
	const double energy=Es[0];

	delete [] Es;
	for (int i=0;i<__diis_space_size__;i++){
		Fs[i].resize(0,0);
		Gs[i].resize(0,0);
		Ds[i].resize(0,0);
	}
	for (int i=0;i<__lbfgs_space_size__;i++){
		pGs[i].resize(0);
		pXs[i].resize(0);
	}
	delete [] Fs;
	delete [] Gs;
	delete [] Ds;
	delete [] pGs;
	delete [] pXs;
	return energy;
}
