#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <x86intrin.h>

#define EigenMatrix Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>
#define EigenZero Eigen::MatrixXd::Zero
#define EigenOne Eigen::MatrixXd::Identity

#include "Optimization.h"

#define __damping_start_threshold__ 100.
#define __damping_factor__ 0.25
#define __adiis_start_iter__ 1
#define __diis_start_iter__ 4
#define __diis_space_size__ 10
#define __asoscf_start_iter__ 100
#define __lbfgs_start_threshold -1.e-3
#define __lbfgs_space_size__ 6
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

		// Without SIMD
		for (long int i=0;i<nints;i++){ // Manual loop unrolling does not help.
			const short int a=*(indicesranger++); // Moving the ranger pointer to the right, where the next index is located.
			const short int b=*(indicesranger++);
			const short int c=*(indicesranger++);
			const short int d=*(indicesranger++); // Moving the ranger pointer to the right, where the degeneracy factor is located.
			const short int deg=*(indicesranger++);
			const double value=*(repulsionranger++); // Moving the ranger pointer to the right, where the next integral is located.
			const double deg_value=deg*value;
			rawjs[iproc](a,b)+=density(c,d)*deg_value;
			rawjs[iproc](c,d)+=density(a,b)*deg_value;
			rawks[iproc](a,c)+=density(b,d)*deg_value;
			rawks[iproc](b,d)+=density(a,c)*deg_value;
			rawks[iproc](a,d)+=density(b,c)*deg_value;
			rawks[iproc](b,c)+=density(a,d)*deg_value;
		}
/*
		// With SIMD AVX2
		for (long int i=0;i<nints-1;i+=2){
			const short int a1=*(indicesranger++); // Moving the ranger pointer to the right, where the next index is located.
			const short int b1=*(indicesranger++);
			const short int c1=*(indicesranger++);
			const short int d1=*(indicesranger++); // Moving the ranger pointer to the right, where the degeneracy factor is located
			const short int deg1=*(indicesranger++);
			const short int a2=*(indicesranger++);
			const short int b2=*(indicesranger++);
			const short int c2=*(indicesranger++);
			const short int d2=*(indicesranger++);
			const short int deg2=*(indicesranger++); // AVX2 allows 4 double vector operations to be processed simultaneously. Each quartet leads to 6 double vector operations. Therefore, the same procedure must go twice, resulting in 12 double vector operations and 3 AVX2 operations in total.

			const double value1=*(repulsionranger++); // Moving the ranger pointer to the right, where the next integral is located.
			const double value2=*(repulsionranger++);
			const double deg_value1=deg1*value1;
			const double deg_value2=deg2*value2;

			const __m256d v1density=_mm256_set_pd(density(c1,d1),density(a1,b1),density(b1,d1),density(a1,c1));
			const __m256d v1deg_value=_mm256_set_pd(deg_value1,deg_value1,deg_value1,deg_value1);
			const __m256d v1result=_mm256_mul_pd(v1density,v1deg_value);
			double v1dresult[4];
			_mm256_store_pd(v1dresult,v1result);
			rawjs[iproc](a1,b1)+=v1dresult[0]; // rawjs[iproc](a1,b1)+=density(c1,d1)*deg_value1;
			rawjs[iproc](c1,d1)+=v1dresult[1]; // rawjs[iproc](c1,d1)+=density(a1,b1)*deg_value1;
			rawks[iproc](a1,c1)+=v1dresult[2]; // rawks[iproc](a1,c1)+=density(b1,d1)*deg_value1;
			rawks[iproc](b1,d1)+=v1dresult[3]; // rawks[iproc](b1,d1)+=density(a1,c1)*deg_value1;

			const __m256d v2density=_mm256_set_pd(density(b1,c1),density(a1,d1),density(c2,d2),density(a2,b2));
			const __m256d v2deg_value=_mm256_set_pd(deg_value1,deg_value1,deg_value2,deg_value2);
			const __m256d v2result=_mm256_mul_pd(v2density,v2deg_value);
			double v2dresult[4];
			_mm256_store_pd(v2dresult,v2result);
			rawks[iproc](a1,d1)+=v2dresult[0]; // rawks[iproc](a1,d1)+=density(b1,c1)*deg_value1;
			rawks[iproc](b1,c1)+=v2dresult[1]; // rawks[iproc](b1,c1)+=density(a1,d1)*deg_value1;
			rawjs[iproc](a2,b2)+=v2dresult[2]; // rawjs[iproc](a2,b2)+=density(c2,d2)*deg_value2;
			rawjs[iproc](c2,d2)+=v2dresult[3]; // rawjs[iproc](c2,d2)+=density(a2,b2)*deg_value2;

			const __m256d v3density=_mm256_set_pd(density(b2,d2),density(a2,c2),density(b2,c2),density(a2,d2));
			const __m256d v3deg_value=_mm256_set_pd(deg_value2,deg_value2,deg_value2,deg_value2);
			const __m256d v3result=_mm256_mul_pd(v3density,v3deg_value);
			double v3dresult[4];
			_mm256_store_pd(v3dresult,v3result);
			rawks[iproc](a2,c2)+=v3dresult[0]; // rawks[iproc](a2,c2)+=density(b2,d2)*deg_value2;
			rawks[iproc](b2,d2)+=v3dresult[1]; // rawks[iproc](b2,d2)+=density(a2,c2)*deg_value2;
			rawks[iproc](a2,d2)+=v3dresult[2]; // rawks[iproc](a2,d2)+=density(b2,c2)*deg_value2;
			rawks[iproc](b2,c2)+=v3dresult[3]; // rawks[iproc](b2,c2)+=density(a2,d2)*deg_value2;
		}
		if (nints&1){ // If the total number of integers of this thread is odd, the last quartet will have been skipped in the loop. It must be handled individually.
			const short int a=*indicesranger;
			const short int b=*indicesranger;
			const short int c=*indicesranger;
			const short int d=*indicesranger;
			const short int deg=*indicesranger;
			const double value=*repulsionranger;
			const double deg_value=deg*value;
			rawjs[iproc](a,b)+=density(c,d)*deg_value;
			rawjs[iproc](c,d)+=density(a,b)*deg_value;
			rawks[iproc](a,c)+=density(b,d)*deg_value;
			rawks[iproc](b,d)+=density(a,c)*deg_value;
			rawks[iproc](a,d)+=density(b,c)*deg_value;
			rawks[iproc](b,c)+=density(a,d)*deg_value;
		}
*/
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

EigenMatrix ElectronicGradient_scf(int nocc,EigenMatrix F,EigenMatrix C){
	int nbasis=F.rows();
	EigenMatrix Fmo=C.transpose()*F*C;
	EigenMatrix g(nocc*(nbasis-nocc),1);
	for (int o=0,k=0;o<nocc;o++) // Occupied orbitals.
		for (int v=nocc;v<nbasis;v++,k++) // Virtual orbitals.
			g(k,0)=-4*Fmo(o,v);
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
	EigenMatrix gradient=4*(overlap*density*F-F*density*overlap); // Electronic gradient of energy with respect to nonredundant orbital rotational parameters.
	char update='f';
	int iiteration=0;

	// DIIS preparation
	EigenMatrix * Fs=new EigenMatrix[__diis_space_size__]; // Storing the last __diis_space_size__ density matrices and their error matrices.
	EigenMatrix * Gs=new EigenMatrix[__diis_space_size__];
	EigenMatrix * Ds=new EigenMatrix[__diis_space_size__];
	double * Es=new double[__diis_space_size__]; // Energy.
	for (int i=0;i<__diis_space_size__;i++){
		Fs[i]=F;
		Gs[i]=gradient;
		Ds[i]=density;
		Es[i]=114514;
	}
	double error2norm=-889464; // |e|^2=Sigma_ij(c_i*c_j*e_i*e_j)

	// L-BFGS preparation
	EigenMatrix pG=EigenZero(nocc*(nbasis-nocc),1); // Packed form of gradient matrix.
	__Loop_Over_OV__ pG(i)=gradient(o,v);
	EigenMatrix pX=pG;
	EigenMatrix * pGs=new EigenMatrix[__lbfgs_space_size__+1];
	EigenMatrix * pXs=new EigenMatrix[__lbfgs_space_size__+1];
	for (int i=0;i<__lbfgs_space_size__;i++){
		pGs[i]=pG;
		pXs[i]=EigenZero(nocc*(nbasis-nocc),1);
	}
	EigenMatrix lastcoefficients;
	int nlbfgs=-1;

	// RHF-SCF iterations
	do{
		if (output) std::cout<<" Iteration "<<iiteration<<":  ";
		const clock_t iterstart_cpu=clock();
		const auto iterstart_wall=std::chrono::system_clock::now();

		// Convergence techniques
		if (iiteration==0){
			if (output) std::cout<<"fock_update = naive  ";
			update='f';
			F=Fs[0];
/*
		}else if (abs(Es[0]-Es[1])>__damping_start_threshold__){ // Using damping in the beginning and when energy oscillates in a large number of iterations.
			if (output) std::cout<<"fock_update = damping  ";
			update='f';
			F=(1.-__damping_factor__)*Fs[0]+__damping_factor__*Fs[1];
*/
		}else if (__diis_start_iter__>=iiteration && iiteration>__adiis_start_iter__){ // Starting A-DIIS in the beginning to facilitate (but not necesarily accelerate) convergence.
			if (output) std::cout<<"fock_update = ADIIS  ";
			update='f';
			F=AEDIIS('a',Es,Ds,Fs,iiteration<__diis_space_size__?iiteration:__diis_space_size__);
		}else if (iiteration>__diis_start_iter__ && gradient.norm()>__lbfgs_start_threshold){ // Starting DIIS after A-DIIS to accelerate convergence in the medium-gradient area.
			if (output) std::cout<<"fock_update = Pulay's_DIIS  ";
			update='f';
			F=DIIS(Fs,Gs,iiteration<__diis_space_size__?iiteration:__diis_space_size__,error2norm); // error2norm is updated.
		}else if (iiteration>__diis_start_iter__ && gradient.norm()<__lbfgs_start_threshold){ // Stopping DIIS for ASOSCF (or simply L-BFGS) in the final part to prevent trailing.
			EigenMatrix hessiandiag(nocc*(nbasis-nocc),1);
			hessiandiag(0)=0;
			if (nlbfgs==-1){
				if (output) std::cout<<"density_update = Pulay's_DIIS  ";
				update='f';
				F=DIIS(Fs,Gs,iiteration<__diis_space_size__?iiteration:__diis_space_size__,error2norm);
			}else{ // ASOSCF does not work. I don't know why.
				if (output) std::cout<<"density_update = ASOSCF  ";
				update='d';
				__Loop_Over_OV__
					hessiandiag(i)=4*(orbitalenergies(v)-orbitalenergies(o));
				pX+=LBFGS(pGs,pXs,nlbfgs<__lbfgs_space_size__?nlbfgs:__lbfgs_space_size__,hessiandiag);
				EigenMatrix expA=EigenOne(nbasis,nbasis);
				__Loop_Over_OV__{
					expA(o,v)=pX(i);
					expA(v,o)=-pX(i);
				}
				coefficients=coefficients*expA;
				const EigenMatrix C_occ=coefficients.leftCols(nocc);
				density=C_occ*C_occ.transpose();
			}
			nlbfgs++;
		}else{
			if (output) std::cout<<"fock_update = naive  ";
			update='f';
			F=Fs[0];
		}

		if (update=='f'){__Fock_2_Density__} // Some techniques update Fock matrix. To complete the iteration, we obtain density matrix from it.

		// Common procedures necessary for all convergence techniques
		F=hcore+GMatrix(repulsion,indices,n2integrals,density,nprocs);
		gradient=4*(overlap*density*F-F*density*overlap); // [F(D),D] instead of [F,D(F)].
		__Loop_Over_OV__ pG(i)=gradient(o,v);
		PushMatrixQueue(F,Fs,__diis_space_size__); // Updating Fock matrix.
		PushMatrixQueue(sinversesqrt.transpose()*gradient*sinversesqrt,Gs,__diis_space_size__); // Updating gradient. I don't know why sinversesqrt is important, but its existence accelerates convergence.
		PushMatrixQueue(density,Ds,__diis_space_size__); // Updating atomic density matrix.
		PushMatrixQueue(pG,pGs,__lbfgs_space_size__); // Updating packed gradient matrix.

		if (nlbfgs==0) lastcoefficients=coefficients;
		else if (nlbfgs==1){
			const EigenMatrix expA=coefficients*lastcoefficients.inverse();
			const EigenMatrix x=expA.topLeftCorner(nocc,nbasis-nocc);
			__Loop_Over_OV__ pX(i)=gradient(o,v);
		}
		PushMatrixQueue(pX,pXs,__lbfgs_space_size__);

		if (update=='d'){__Fock_2_Density__} // Some techniques update density matrix. To complete the iteration, we obtain Fock matrix from it.

		// Iteration information output
		const EigenMatrix Iamgoingtobeanobellaureate=density*(hcore+F); // Intermediate matrix.
		PushDoubleQueue(Iamgoingtobeanobellaureate.trace(),Es,__diis_space_size__); // Updating energy.
		const std::chrono::duration<double> duration_wall=std::chrono::system_clock::now()-iterstart_wall;
		if (output) std::cout<<"energy = "<<std::setprecision(12)<<Es[0]<<std::setprecision(3)<<" a.u.  ||gradient|| = "<<gradient.norm()<<" a.u.  cpu_time = "<<double(clock()-iterstart_cpu)/CLOCKS_PER_SEC<<" s  wall_time = "<<duration_wall.count()<<" s"<<std::endl;
		iiteration++;
	}while (abs(Es[0]-Es[1])>__scf_convergence_energy_threshold__ || gradient.norm()>__scf_convergence_gradient_threshold__);
	if (output) std::cout<<"Done; Final RHF energy = "<<std::setprecision(12)<<Es[0]<<" a.u."<<std::endl;
	const double energy=Es[0];

	delete [] Es;
	for (int i=0;i<__diis_space_size__;i++){
		Fs[i].resize(0,0);
		Gs[i].resize(0,0);
		Ds[i].resize(0,0);
	}
	for (int i=0;i<__lbfgs_space_size__;i++){
		pGs[i].resize(0,0);
		pXs[i].resize(0,0);
	}
	delete [] Fs;
	delete [] Gs;
	delete [] Ds;
	delete [] pGs;
	delete [] pXs;
	return energy;
}
