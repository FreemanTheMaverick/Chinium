#include <Eigen/Dense>
#include <cmath>
#include <cassert>
#include <chrono>
#include <omp.h>
#include "Aliases.h"
#include "HartreeFock.h"
#include "Optimization.h"
#include "LinearAlgebra.h"
#include "GridIntegrals.h"
#include "DensityFunctional.h"
#include <iostream>

#define __diis_start_iter__ 2
#define __diis_space_size__ 6
#define __cpscf_max_iter__ 50
#define __convergence_threshold__ 1.e-4

EigenMatrix FockOccupationGradientCPSCF(
		double temperature,double * repulsion,short int * indices,long int n2integrals,double kscale,
		EigenMatrix * ovlgrads,EigenMatrix * fskeletons,EigenMatrix * Dxns,EigenVector * exns,int natoms,
		EigenMatrix coefficients,EigenVector occupancies,EigenVector orbitalenergies,
		const int nprocs,const bool output){
	if (output) std::printf("Calculating Fock matrix derivative with respect to occupation numbers for Fock-matrix-based occupation-gradient CPSCF... ");
	auto start=std::chrono::system_clock::now();
	int nbasis=coefficients.cols();
	short int * degs=indices+0*n2integrals;
	short int * bf1s=indices+1*n2integrals;
	short int * bf2s=indices+2*n2integrals;
	short int * bf3s=indices+3*n2integrals;
	short int * bf4s=indices+4*n2integrals;
	long int nintsperthread_fewer=n2integrals/nprocs;
	int ntimes_fewer=nprocs-n2integrals+nintsperthread_fewer*nprocs; // How many integrals a thread will handle. If the average number is A, the number of each thread is either a or (a+1), where a=floor(A). The number of threads to handle (a) integrals, x, and that to compute (a+1) integrals, y, can be obtained by solving (1) a*x+(a+1)*y=b and (2) x+y=c, where b and c stand for the total numbers of integrals and threads respectively.
	long int * iintfirstperthread=new long int[nprocs];
	long int * nintsperthread=new long int[nprocs];
	for (int iproc=0;iproc<nprocs;iproc++){
		iintfirstperthread[iproc]=iproc==0?0:(iintfirstperthread[iproc-1]+nintsperthread[iproc-1]);
		nintsperthread[iproc]=iproc<ntimes_fewer?nintsperthread_fewer:(nintsperthread_fewer+1);
	}
	EigenMatrix ** rawjs=new EigenMatrix*[nprocs];
	EigenMatrix ** rawks=new EigenMatrix*[nprocs];
	for (int iproc=0;iproc<nprocs;iproc++){
		rawjs[iproc]=new EigenMatrix[nbasis];
		rawks[iproc]=new EigenMatrix[nbasis];
		for (int jbasis=0;jbasis<nbasis;jbasis++){
			rawjs[iproc][jbasis]=EigenZero(nbasis,nbasis);
			rawks[iproc][jbasis]=EigenZero(nbasis,nbasis);
		}
	}
	EigenMatrix * scps=new EigenMatrix[nbasis]; // Self-cross-products of each column of coefficient matrix.
	for (int jbasis=0;jbasis<nbasis;jbasis++)
		scps[jbasis]=coefficients.col(jbasis)*coefficients.col(jbasis).transpose();

	omp_set_num_threads(nprocs);
	#pragma omp parallel for
	for (int iproc=0;iproc<nprocs;iproc++){
		Eigen::setNbThreads(1);
		long int nints=nintsperthread[iproc];
		long int iintfirst=iintfirstperthread[iproc];
		double * thisrepulsions=repulsion+iintfirst;
		short int * thisdegs=degs+iintfirst;
		short int * thisbf1s=bf1s+iintfirst;
		short int * thisbf2s=bf2s+iintfirst;
		short int * thisbf3s=bf3s+iintfirst;
		short int * thisbf4s=bf4s+iintfirst;
		short int a,b,c,d;
		double deg_value;
		// Without SIMD
		for (long int i=0;i<nints;i++){ // Manual loop unrolling does not help.
			deg_value=*(thisdegs++)**(thisrepulsions++); // Moving the ranger pointer to the right, where the next integral is located.
			a=*(thisbf1s++);
			b=*(thisbf2s++);
			c=*(thisbf3s++);
			d=*(thisbf4s++);
			for (int jbasis=0;jbasis<nbasis;jbasis++){
				rawjs[iproc][jbasis](a,b)+=scps[jbasis](c,d)*deg_value;
				rawjs[iproc][jbasis](c,d)+=scps[jbasis](a,b)*deg_value;
				if (kscale>0){
					rawks[iproc][jbasis](a,c)+=scps[jbasis](b,d)*deg_value;
					rawks[iproc][jbasis](b,d)+=scps[jbasis](a,c)*deg_value;
					rawks[iproc][jbasis](a,d)+=scps[jbasis](b,c)*deg_value;
					rawks[iproc][jbasis](b,c)+=scps[jbasis](a,d)*deg_value;
				}
			}
		}
	}
	Eigen::setNbThreads(nprocs);
	EigenMatrix * rawjn=new EigenMatrix[nbasis];
	EigenMatrix * rawkn=new EigenMatrix[nbasis];
	for (int jbasis=0;jbasis<nbasis;jbasis++){
		rawjn[jbasis]=EigenZero(nbasis,nbasis);
		rawkn[jbasis]=EigenZero(nbasis,nbasis);
		for (int iproc=0;iproc<nprocs;iproc++){
			rawjn[jbasis]+=rawjs[iproc][jbasis];
			rawjs[iproc][jbasis].resize(0,0);
			if (kscale>0){
				rawkn[jbasis]+=rawks[iproc][jbasis];
				rawks[iproc][jbasis].resize(0,0);
			}
		}
	}
	for (int iproc=0;iproc<nprocs;iproc++){
		delete [] rawjs[iproc];
		delete [] rawks[iproc];
	}
	delete [] iintfirstperthread;
	delete [] nintsperthread;
	delete [] rawjs;
	delete [] rawks;
	EigenMatrix * fn=new EigenMatrix[nbasis];
	for (int jbasis=0;jbasis<nbasis;jbasis++){
		EigenMatrix jn=0.5*(rawjn[jbasis]+rawjn[jbasis].transpose());
		EigenMatrix kn=0.25*(rawkn[jbasis]+rawkn[jbasis].transpose());
		fn[jbasis]=jn-0.5*kscale*kn;
	}
	__Delete_Matrices__(rawjn,nbasis);
	__Delete_Matrices__(rawkn,nbasis);
	std::chrono::duration<double> duration=std::chrono::system_clock::now()-start;
	if (output) std::printf("%f s\n",duration.count());

	/* Checking whether every MO's contribution to G is correct
	EigenMatrix dd=EigenZero(nbasis,nbasis);
	EigenMatrix ff=EigenZero(nbasis,nbasis);
	for (int kbasis=0;kbasis<nbasis;kbasis++){
		dd+=occupancies[kbasis]*coefficients.col(kbasis)*coefficients.col(kbasis).transpose();
		ff+=fn[kbasis]*occupancies[kbasis];
	}
	std::printf("%f\n",(ff-GMatrix(repulsion,indices,n2integrals,dd,kscale,nprocs)).norm());
	*/

	if (output){
		std::printf("Fock-matrix-based occupation-gradient CPSCF ...\n");
		std::printf("| Perturbation | # of iterations | Wall time |\n");
	}
	EigenMatrix nxs=EigenZero(3*natoms,nbasis);
	for (int ipert=0;ipert<3*natoms;ipert++){
		auto pert_start=std::chrono::system_clock::now();
		EigenMatrix fxn=fskeletons[ipert]+GhfMatrix(repulsion,indices,n2integrals,Dxns[ipert],kscale,nprocs);
		EigenMatrix Fx=fxn;
		EigenMatrix csxc=coefficients.transpose()*ovlgrads[ipert]*coefficients;
		EigenMatrix R=EigenZero(nbasis,nbasis);R(0)=114514;
		EigenMatrix * Fxs=new EigenMatrix[__diis_space_size__];
		EigenMatrix * Rs=new EigenMatrix[__diis_space_size__];
		for (int i=0;i<__diis_space_size__;i++){
			Fxs[i]=Fx;
			Rs[i]=R;
		}
		int jiter=0;
		for (jiter=0;R.norm()>__convergence_threshold__;jiter++){
			assert("FOG-CPSCF does not converge." && jiter<__cpscf_max_iter__);
			double error2norm=114514;
			Fx=Fxs[0];
			EigenMatrix Dx=Dxns[ipert];
			if (jiter>__diis_start_iter__)
				Fx=DIIS(Fxs,Rs,jiter<__diis_space_size__?jiter:__diis_space_size__,error2norm);
			for (int kbasis=0;kbasis<nbasis;kbasis++){
				const double exs=coefficients.col(kbasis).transpose()*Fx*coefficients.col(kbasis)-csxc(kbasis,kbasis)*orbitalenergies[kbasis];
				nxs(ipert,kbasis)=occupancies[kbasis]*(occupancies[kbasis]-1.)/temperature*exs;
				Dx+=coefficients.col(kbasis)*coefficients.col(kbasis).transpose()*nxs(ipert,kbasis);
			}
			Fx=fxn;
			for (int kbasis=0;kbasis<nbasis;kbasis++)
				Fx+=fn[kbasis]*nxs(ipert,kbasis);
			R=Fx-Fxs[0];
			PushMatrixQueue(Fx,Fxs,__diis_space_size__);
			PushMatrixQueue(R,Rs,__diis_space_size__);
			//std::cout<<(Fx*D*overlap+F*dx*overlap+F*D*ovlgrads[ipert]-overlap*D*Fx-overlap*dx*F-ovlgrads[ipert]*D*F).norm()<<std::endl;
		}
		const std::chrono::duration<double> duration=std::chrono::system_clock::now()-pert_start;
		if (output) std::printf("|    %6d    |     %7d     | %9.6f |\n",ipert,jiter,duration.count());
	}

	EigenMatrix hessian=EigenZero(3*natoms,3*natoms);
	for (int ipert=0;ipert<3*natoms;ipert++)
		for (int jpert=0;jpert<3*natoms;jpert++)
			for (int kbasis=0;kbasis<nbasis;kbasis++)
				hessian(ipert,jpert)+=exns[ipert](kbasis)*nxs(jpert,kbasis);
	return 0.5*(hessian+hessian.transpose());
}


EigenMatrix DensityOccupationGradientCPSCF(
		int natoms,short int * bf2atom,double temperature,
		EigenMatrix * ovlgrads,EigenMatrix * fskeletons,EigenMatrix * Dxns,EigenVector * exns,
		double * repulsion,short int * indices,long int n2integrals,
		int dfxid,int dfcid,int ngrids,double * ws,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2xxs,double * ao2yys,double * ao2zzs,
		double * ao2xys,double * ao2xzs,double * ao2yzs,
		EigenMatrix coefficients,EigenVector occupancies,EigenVector orbitalenergies,
		const int nprocs,const bool output){
	if (output){
		std::printf("Density-matrix-based occupation-gradient CPSCF ...\n");
		std::printf("| Perturbation | # of iterations | Wall time |\n");
	}

	int nbasis=coefficients.cols();
	EigenMatrix D=EigenZero(nbasis,nbasis);
	D.diagonal()=occupancies;
	D=coefficients*D*coefficients.transpose();
	
	__Initialize_KS__(aos,ao1xs,ao1ys,ao1zs,0,1,0)
	GetDensity(
			aos,
			ao1xs,ao1ys,ao1zs,
			nullptr,
			ngrids,2*D,
			ds,
			d1xs,d1ys,d1zs,cgs,
			nullptr,nullptr);
	__KS_Potential__(1,0)
	double * dns=nullptr;
	double * dnxs,* dnys,* dnzs;
	dnxs=dnys=dnzs=nullptr;
	double * dxns,* dyns,* dzns;
	dxns=dyns=dzns=nullptr;
	double * dxnxs,* dxnys,* dxnzs,* dynxs,* dynys,* dynzs,* dznxs,* dznys,* dznzs;
	dxnxs=dxnys=dxnzs=dynxs=dynys=dynzs=dznxs=dznys=dznzs=nullptr;

	if (dfxid){
		if (aos && ao1xs){
			dns=new double[ngrids]();
			dnxs=new double[ngrids]();
			dnys=new double[ngrids]();
			dnzs=new double[ngrids]();
		}
		if (aos && ao1xs && ao2xxs){
			dxns=new double[ngrids]();
			dyns=new double[ngrids]();
			dzns=new double[ngrids]();
			dxnxs=new double[ngrids]();
			dxnys=new double[ngrids]();
			dxnzs=new double[ngrids]();
			dynxs=new double[ngrids]();
			dynys=new double[ngrids]();
			dynzs=new double[ngrids]();
			dznxs=new double[ngrids]();
			dznys=new double[ngrids]();
			dznzs=new double[ngrids]();
		}
	}
	EigenMatrix nxs=EigenZero(3*natoms,nbasis);
	for (int ipert=0;ipert<3*natoms;ipert++){
		auto pert_start=std::chrono::system_clock::now();
		EigenMatrix fxn=fskeletons[ipert]+GhfMatrix(repulsion,indices,n2integrals,Dxns[ipert],kscale,nprocs);
		if (dfxid && ipert%3==0){
			if (dnxs){
				memset(dnxs,0,ngrids*sizeof(double));
				memset(dnys,0,ngrids*sizeof(double));
				memset(dnzs,0,ngrids*sizeof(double));
			}
			if (dxnxs){
				memset(dxnxs,0,ngrids*sizeof(double));
				memset(dxnys,0,ngrids*sizeof(double));
				memset(dxnzs,0,ngrids*sizeof(double));
				memset(dynxs,0,ngrids*sizeof(double));
				memset(dynys,0,ngrids*sizeof(double));
				memset(dynzs,0,ngrids*sizeof(double));
				memset(dznxs,0,ngrids*sizeof(double));
				memset(dznys,0,ngrids*sizeof(double));
				memset(dznzs,0,ngrids*sizeof(double));
			}
			GetDensitySkeleton( // Grid density skeleton derivatives
				aos,
				ao1xs,ao1ys,ao1zs,
				ao2xxs,ao2yys,ao2zzs,
				ao2xys,ao2xzs,ao2yzs,
				ngrids,2*D,
				ipert/3,bf2atom,
				dnxs,dnys,dnzs,
				dxnxs,dxnys,dxnzs,
				dynxs,dynys,dynzs,
				dznxs,dznys,dznzs);
		}
		if (dfxid){
			if (ipert%3==0){ // Grid density skeleton derivatives
				if (dns)
					memcpy(dns,dnxs,sizeof(double)*ngrids);
				if (dxns){
					memcpy(dxns,dxnxs,sizeof(double)*ngrids);
					memcpy(dyns,dynxs,sizeof(double)*ngrids);
					memcpy(dzns,dznxs,sizeof(double)*ngrids);
				}
			}else if (ipert%3==1){
				if (dns)
					memcpy(dns,dnys,sizeof(double)*ngrids);
				if (dxns){
					memcpy(dxns,dxnys,sizeof(double)*ngrids);
					memcpy(dyns,dynys,sizeof(double)*ngrids);
					memcpy(dzns,dznys,sizeof(double)*ngrids);
				}
			}else if (ipert%3==2){
				if (dns)
					memcpy(dns,dnzs,sizeof(double)*ngrids);
				if (dxns){
					memcpy(dxns,dxnzs,sizeof(double)*ngrids);
					memcpy(dyns,dynzs,sizeof(double)*ngrids);
					memcpy(dzns,dznzs,sizeof(double)*ngrids);
				}
			}
			GetDensity( // Grid density U derivatives
				aos,
				ao1xs,ao1ys,ao1zs,
				nullptr,
				ngrids,2*Dxns[ipert],
				dns,
				dxns,dyns,dzns,nullptr,
				nullptr,nullptr);
			fxn+=FxcUMatrix( // KS part of Fock matrix U derivative
				ws,ngrids,nbasis,
				aos,
				ao1xs,ao1ys,ao1zs,
				d1xs,d1ys,d1zs,
				vsxcs,
				vrrxcs,
				vrsxcs,vssxcs,
				dns,
				dxns,dyns,dzns);
		}
		EigenMatrix Fx=fxn;
		EigenMatrix csxc=coefficients.transpose()*ovlgrads[ipert]*coefficients;
		EigenMatrix R=EigenZero(nbasis,nbasis);R(0)=114514;
		EigenMatrix * Fxs=new EigenMatrix[__diis_space_size__];
		EigenMatrix * Rs=new EigenMatrix[__diis_space_size__];
		for (int i=0;i<__diis_space_size__;i++){
			Fxs[i]=Fx;
			Rs[i]=R;
		}
		double error2norm=114514;

		int jiter=0;
		for (jiter=0;R.norm()>__convergence_threshold__;jiter++){
			assert("DOG-CPSCF does not converge." && jiter<__cpscf_max_iter__);
			Fx=Fxs[0];
			if (jiter>__diis_start_iter__)
				Fx=DIIS(Fxs,Rs,jiter<__diis_space_size__?jiter:__diis_space_size__,error2norm);
			EigenMatrix Dx=Dxns[ipert];
			for (int kbasis=0;kbasis<nbasis;kbasis++){ // Fock 2 density
				const double exs=coefficients.col(kbasis).transpose()*Fx*coefficients.col(kbasis)-csxc(kbasis,kbasis)*orbitalenergies[kbasis];
				nxs(ipert,kbasis)=occupancies[kbasis]*(occupancies[kbasis]-1.)/temperature*exs;
				Dx+=coefficients.col(kbasis)*coefficients.col(kbasis).transpose()*nxs(ipert,kbasis);
			}
			Fx=fskeletons[ipert]+GhfMatrix(repulsion,indices,n2integrals,Dx,kscale,nprocs); // Density 2 Fock
			if (dfxid){
				if (ipert%3==0){ // Grid density skeleton derivatives
					if (dns)
						memcpy(dns,dnxs,sizeof(double)*ngrids);
					if (dxns){
						memcpy(dxns,dxnxs,sizeof(double)*ngrids);
						memcpy(dyns,dynxs,sizeof(double)*ngrids);
						memcpy(dzns,dznxs,sizeof(double)*ngrids);
					}
				}else if (ipert%3==1){
					if (dns)
						memcpy(dns,dnys,sizeof(double)*ngrids);
					if (dxns){
						memcpy(dxns,dxnys,sizeof(double)*ngrids);
						memcpy(dyns,dynys,sizeof(double)*ngrids);
						memcpy(dzns,dznys,sizeof(double)*ngrids);
					}
				}else if (ipert%3==2){
					if (dns)
						memcpy(dns,dnzs,sizeof(double)*ngrids);
					if (dxns){
						memcpy(dxns,dxnzs,sizeof(double)*ngrids);
						memcpy(dyns,dynzs,sizeof(double)*ngrids);
						memcpy(dzns,dznzs,sizeof(double)*ngrids);
					}
				}
				GetDensity( // Grid density U derivatives
					aos,
					ao1xs,ao1ys,ao1zs,
					nullptr,
					ngrids,2*Dx,
					dns,
					dxns,dyns,dzns,nullptr,
					nullptr,nullptr);
				Fx+=FxcUMatrix( // KS part of Fock matrix U derivative
					ws,ngrids,nbasis,
					aos,
					ao1xs,ao1ys,ao1zs,
					d1xs,d1ys,d1zs,
					vsxcs,
					vrrxcs,
					vrsxcs,vssxcs,
					dns,
					dxns,dyns,dzns);
			}
			R=Fx-Fxs[0];
			PushMatrixQueue(Fx,Fxs,__diis_space_size__);
			PushMatrixQueue(R,Rs,__diis_space_size__);
			//std::cout<<(Fx*D*overlap+F*dx*overlap+F*D*ovlgrads[ipert]-overlap*D*Fx-overlap*dx*F-ovlgrads[ipert]*D*F).norm()<<std::endl;
		}
		const std::chrono::duration<double> duration=std::chrono::system_clock::now()-pert_start;
		if (output) std::printf("|    %6d    |     %7d     | %9.6f |\n",ipert,jiter,duration.count());
	}
	EigenMatrix hessian=EigenZero(3*natoms,3*natoms);
	for (int ipert=0;ipert<3*natoms;ipert++)
		for (int jpert=0;jpert<3*natoms;jpert++)
			for (int kbasis=0;kbasis<nbasis;kbasis++)
				hessian(ipert,jpert)+=exns[ipert](kbasis)*nxs(jpert,kbasis);
	std::cout<<"occupancy"<<std::endl;
	std::cout<<occupancies<<std::endl;
	std::cout<<"nxs.row(8)"<<std::endl;
	std::cout<<nxs.row(8).transpose()<<std::endl;
	__Finalize_KS__
	delete [] dns;
	delete [] dnxs;
	delete [] dnys;
	delete [] dnzs;
	delete [] dxns;
	delete [] dyns;
	delete [] dzns;
	delete [] dxnxs;
	delete [] dxnys;
	delete [] dxnzs;
	delete [] dynxs;
	delete [] dynys;
	delete [] dynzs;
	delete [] dznxs;
	delete [] dznys;
	delete [] dznzs;
	return 0.5*(hessian+hessian.transpose());
}

