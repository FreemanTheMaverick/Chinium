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
#define __small_value__ 1.e-9

void NonIdempotentCPSCF(
		int natoms,short int * bf2atom,
		EigenMatrix * ovlgrads,EigenMatrix * fskeletons,
		double * repulsion,short int * indices,long int n2integrals,
		int dfxid,int dfcid,int ngrids,double * ws,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2xxs,double * ao2yys,double * ao2zzs,
		double * ao2xys,double * ao2xzs,double * ao2yzs,
		EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupancies,
		EigenMatrix * Wxns,EigenMatrix * Dxns,EigenVector * exns,
		const int nprocs,const bool output){
	if (output){
		std::printf("Non-idempotent CPSCF ...\n");
		std::printf("| Perturbation | # of iterations | Wall time |\n");
	}

	// Initialization
	int nbasis=coefficients.cols();
	EigenMatrix D=EigenZero(nbasis,nbasis);
	D.diagonal()=occupancies;
	D=coefficients*D*coefficients.transpose();

	// Generating vrrxcs
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

	for (int ipert=0;ipert<natoms*3;ipert++){

		// Initialization
		const auto iterstart_wall=std::chrono::system_clock::now();
		int jiter=0;
		EigenMatrix U(nbasis,nbasis);
		EigenMatrix * Bs=new EigenMatrix[__diis_space_size__];
		EigenMatrix * Rs=new EigenMatrix[__diis_space_size__]; // R for residue.
		for (int i=0;i<__diis_space_size__;i++){
			Bs[i]=EigenZero(nbasis,nbasis);
			Rs[i]=EigenZero(nbasis,nbasis);
		}
		double error2norm=114514;

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

		// Forming B1, common for all iterations.
		const EigenMatrix csxc=coefficients.transpose()*ovlgrads[ipert]*coefficients;
		const EigenMatrix cfsxc=coefficients.transpose()*fskeletons[ipert]*coefficients;
		EigenMatrix B1=csxc.array().rowwise()*orbitalenergies.transpose().array();
		B1-=cfsxc;
		EigenMatrix B=B1;
		EigenMatrix R=EigenZero(nbasis,nbasis);R(0)=114514;

		// Forming N, common for all iterations
		EigenMatrix N=EigenZero(nbasis,nbasis);
		EigenMatrix Mii=EigenZero(nbasis,nbasis);
		EigenMatrix Mij=EigenZero(nbasis,nbasis);
		for (int i=0,ij=0;i<nbasis;i++){
			Mii=coefficients.col(i)*coefficients.col(i).transpose()+coefficients.col(i)*coefficients.col(i).transpose();
			N+=0.5*occupancies[i]*csxc(i,i)*Mii;
			for (int j=0;j<i;j++,ij++){
				Mij=coefficients.col(i)*coefficients.col(j).transpose()+coefficients.col(j)*coefficients.col(i).transpose();
				N+=occupancies[i]*csxc(i,j)*Mij;
			}
		}

		for (jiter=0;R.norm()>__convergence_threshold__;jiter++){
			assert("CPSCF does not converge." && jiter<__cpscf_max_iter__);
			if (jiter>__diis_start_iter__)
				B=DIIS(Bs,Rs,jiter<__diis_space_size__?jiter:__diis_space_size__,error2norm);
			Dxns[ipert]=-N;
			for (int i=0;i<nbasis;i++)
				for (int j=0;j<=i;j++){
					if (std::abs(orbitalenergies[i]-orbitalenergies[j])<__small_value__) // Avoiding "division by zero" error.
						U(i,j)=U(j,i)=-0.5*csxc(i,j);
					else{
						U(i,j)=B(i,j)/(orbitalenergies[i]-orbitalenergies[j]);
						U(j,i)=-U(i,j)-csxc(i,j);
					}
					Mij=coefficients.col(i)*coefficients.col(j).transpose()+coefficients.col(j)*coefficients.col(i).transpose();
					Dxns[ipert]+=(occupancies[j]-occupancies[i])*U(i,j)*Mij;
				}

			EigenMatrix G=GhfMatrix(repulsion,indices,n2integrals,Dxns[ipert],kscale,nprocs);
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
				G+=FxcUMatrix( // KS part of Fock matrix U derivative
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

			B=B1-coefficients.transpose()*G*coefficients; // Forming a new B matrix.
			for (int i=0;i<nbasis;i++)
				for (int j=0;j<nbasis;j++){
					R(i,j)=B(i,j)-(orbitalenergies[i]-orbitalenergies[j])*U(i,j);
					if (std::abs(orbitalenergies[i]-orbitalenergies[j])<__small_value__)
						R(i,j)=0; // Ignoring rotation among degenerate orbitals.
				}
			R-=R.diagonal().asDiagonal();
			PushMatrixQueue(B,Bs,__diis_space_size__);
			PushMatrixQueue(R,Rs,__diis_space_size__);
		}

		if (Wxns){
			Wxns[ipert]=EigenZero(nbasis,nbasis);
			for (int i=0;i<nbasis;i++){
				Wxns[ipert]-=occupancies[i]*B(i,i)*coefficients.col(i)*coefficients.col(i).transpose();
				for (int j=0;j<nbasis;j++){
					Mij=coefficients.col(i)*coefficients.col(j).transpose()+coefficients.col(j)*coefficients.col(i).transpose();
					Wxns[ipert]+=occupancies[j]*orbitalenergies[j]*U(i,j)*Mij;
				}
			}
		}

		if (exns) exns[ipert]=-B.diagonal();
		
		const std::chrono::duration<double> duration_wall=std::chrono::system_clock::now()-iterstart_wall;
		if (output) std::printf("|    %6d    |     %7d     | %9.6f |\n",ipert,jiter,duration_wall.count());
	}
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
}


