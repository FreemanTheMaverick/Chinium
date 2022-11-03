#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include "Optimization.h"

#define EigenMatrix Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>
#define EigenZero Eigen::MatrixXd::Zero
#define EigenOne Eigen::MatrixXd::Identity

#define __damping_start_threshold__ 5.
#define __damping_factor__ 0.25
#define __diis_space_size__ 6
#define __asoscf_start_iter__ 30
#define __lbfgs_space_size__ 6
#define __trah_start_iter__ 50
#define __scf_convergence_threshold__ 1.e-8

EigenMatrix GMatrix(double * repulsion,short int * indices,int n2integrals,EigenMatrix densitymatrix,const int nprocs){
	int nbasis=densitymatrix.cols();
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
			bigrawjmatrix[iproc](a,b)+=densitymatrix(c,d)*deg_value;
			bigrawjmatrix[iproc](c,d)+=densitymatrix(a,b)*deg_value;
			bigrawkmatrix[iproc](a,c)+=0.5*densitymatrix(b,d)*deg_value;
			bigrawkmatrix[iproc](b,d)+=0.5*densitymatrix(a,c)*deg_value;
			bigrawkmatrix[iproc](a,d)+=0.5*densitymatrix(b,c)*deg_value;
			bigrawkmatrix[iproc](b,c)+=0.5*densitymatrix(a,d)*deg_value;
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

double RHF(int nele,EigenMatrix overlap,EigenMatrix hcore,double * repulsion,short int * indices,long int n2integrals,EigenMatrix & orbitalenergies,EigenMatrix & coefficients,EigenMatrix & densitymatrix,const int nprocs,const bool output){
	if (output) std::cout<<"Restricted Hartree-Fock ..."<<std::endl;
	const int nbasis=overlap.cols();
	EigenMatrix * Ds=new EigenMatrix[__diis_space_size__]; // Storing the last __diis_space_size__ density matrices and their error matrices.
	EigenMatrix * Es=new EigenMatrix[__diis_space_size__];
	for (int i=0;i<__diis_space_size__;i++){
		Ds[i]=EigenZero(nbasis,nbasis);
		Es[i]=EigenZero(nbasis,nbasis);
	}
	const int nocc=nele/2;
	EigenMatrix * Ss=new EigenMatrix[__lbfgs_space_size__];
	EigenMatrix * Ys=new EigenMatrix[__lbfgs_space_size__];
	for (int i=0;i<__lbfgs_space_size__;i++){
		Ss[i]=EigenZero(nocc*(nbasis-nocc),1);
		Ys[i]=EigenZero(nocc*(nbasis-nocc),1);
	}
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(overlap);
	const EigenMatrix s=eigensolver.eigenvalues();
	EigenMatrix sinversesqrt=Eigen::MatrixXd::Zero(nbasis,nbasis);
	for (int i=0;i<nbasis;i++){
		sinversesqrt(i,i)=1/sqrt(s(i,0));
	}
	const EigenMatrix U=eigensolver.eigenvectors();
	const EigenMatrix X=U*sinversesqrt*U.transpose();
	EigenMatrix F=EigenZero(nbasis,nbasis); // Fock matrix.
	double energy=114514;
	double lastenergy=1919810;
	EigenMatrix lastdensitymatrix=densitymatrix;
	EigenMatrix g=EigenZero(nocc*(nbasis-nocc),1); // Electronic gradient of energy with respect to nonredundant orbital rotational parameters.
	EigenMatrix lastg=g;
	double error2norm=-889464; // |e|^2=Sigma_ij(c_i*c_j*e_i*e_j)
	int iiteration=0;
	while (abs(lastenergy-energy)>__scf_convergence_threshold__ || error2norm>__scf_convergence_threshold__){ // Normal RHF SCF procedure.
		if (output) std::cout<<" Iteration "<<iiteration<<":  ";
		const clock_t iterstart=clock();
		if (__asoscf_start_iter__>iiteration){ // Converging methods without energy derivatives.
			if (abs(lastenergy-energy)*iiteration>__damping_start_threshold__){ // Using damping in the beginning and when energy oscillates in a large number of iterations.
				if (output) std::cout<<"density_update = damping  ";
				densitymatrix=(1.-__damping_factor__)*densitymatrix+__damping_factor__*lastdensitymatrix;
			}else if (iiteration>=__diis_space_size__){ // Starting DIIS after Ds and Es are filled and error2norm is not too large.
				if (output) std::cout<<"density_update = DIIS  ";
				densitymatrix=DIIS(Ds,Es,__diis_space_size__,error2norm); // error2norm is updated.
			}else if (output) std::cout<<"density_update = naive  ";
			const EigenMatrix G=GMatrix(repulsion,indices,n2integrals,densitymatrix,nprocs);
			F=hcore+G;
			const EigenMatrix Fprime=X.transpose()*F*X;
			eigensolver.compute(Fprime);
			orbitalenergies=eigensolver.eigenvalues();
			const EigenMatrix Cprime=eigensolver.eigenvectors();
			coefficients=X*Cprime;
			const EigenMatrix C_occ=coefficients.leftCols(nocc);
			const EigenMatrix newdensitymatrix=C_occ*C_occ.transpose();
			Ds[iiteration%__diis_space_size__]=newdensitymatrix; // The oldest density matrix and its error matrix are replaced by the latest ones.
			Es[iiteration%__diis_space_size__]=newdensitymatrix-densitymatrix;
			lastdensitymatrix=densitymatrix;
			densitymatrix=newdensitymatrix;
			g=ElectronicGradient_scf(nocc,F,coefficients);
		}else{ // Converging methods with energy derivatives.
			break;/*
			EigenMatrix s=lastg;
			if (__trah_start_iter__>iiteration){
				if (output) std::cout<<"step_update = ASOSCF  ";
				if (__asoscf_start_iter__+__lbfgs_space_size__>iiteration){
					if (__asoscf_start_iter__==iiteration)
						g=ElectronicGradient_scf(nocc,F,coefficients);
					for (int o=0,k=0;o<nocc;o++)
						for (int v=nocc;v<nbasis;v++,k++)
							s(k)=g(k)/2/(orbitalenergies(v)-orbitalenergies(o));
				}else s=LBFGS(lastg,Ss,Ys,__lbfgs_space_size__,(iiteration-__asoscf_start_iter__+__lbfgs_space_size__-1)%__lbfgs_space_size__);
			}else{
				if (output) std::cout<<"step_update = TRAHSCF  ";
			}
			EigenMatrix A=EigenZero(nbasis,nbasis);
			for (int o=0,k=0;o<nocc;o++){
				for (int v=nocc;v<nbasis;v++,k++){
					A(o,v)=s(k);
					A(v,o)=-s(k);
				}
			}
			coefficients=coefficients*(EigenOne(nbasis,nbasis)+A+A*A/2+A*A*A/6);
			const EigenMatrix C_occ=coefficients.leftCols(nocc);
			EigenMatrix densitymatrix=C_occ*C_occ.transpose();
			const EigenMatrix G=GMatrix(repulsion,indices,n2integrals,densitymatrix,nprocs);
			F=hcore+G;
			g=ElectronicGradient_scf(nocc,F,coefficients);
			Ss[(iiteration-__asoscf_start_iter__)%__lbfgs_space_size__]=s;
			Ys[(iiteration-__asoscf_start_iter__)%__lbfgs_space_size__]=g-lastg;
			lastg=g;
			Ds[iiteration%__diis_space_size__]=densitymatrix; // The oldest density matrix and its error matrix are replaced by the latest ones.
			Es[iiteration%__diis_space_size__]=densitymatrix-lastdensitymatrix;
			DIIS(Ds,Es,__diis_space_size__,error2norm); // Here DIIS is used only for error2norm as a convergence criterion.
			lastdensitymatrix=densitymatrix;*/
		}
		const EigenMatrix Iamgoingtobeanobellaureate=densitymatrix*(hcore+F); // Intermediate matrix.
		lastenergy=energy;
		energy=Iamgoingtobeanobellaureate.trace();
		const clock_t iterend=clock();
		if (output) std::cout<<"energy = "<<std::setprecision(12)<<energy<<" a.u."<<"  elapsed_time = "<<std::setprecision(3)<<double(iterend-iterstart)/CLOCKS_PER_SEC<<" s"<<std::endl;
		iiteration++;
	}
	delete [] Ds;
	delete [] Es;
	delete [] Ss;
	delete [] Ys;
	if (output) std::cout<<"Done; Final RHF energy = "<<std::setprecision(12)<<energy<<" a.u."<<std::endl;
	return energy;
}
