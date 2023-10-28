#include <Eigen/Dense>
#include <cmath>
#include <cassert>
#include <chrono>
#include <omp.h>
#include "Aliases.h"
#include "HartreeFock.h"
#include "Optimization.h"
#include "LinearAlgebra.h"

#define __diis_start_iter__ 2
#define __diis_space_size__ 6
#define __cpscf_max_iter__ 50
#define __convergence_threshold__ 1.e-4
#define __small_value__ 1.e-9

void NonIdempotent(int natoms,
                   EigenMatrix * ovlgrads,EigenMatrix * fskeletons,
                   double * repulsion,short int * indices,long int n2integrals,double kscale,
                   EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupancies,
		   EigenMatrix * wxn,EigenMatrix * dxn,
		   const int nprocs,const bool output){
	if (output){
		std::printf("Non-idempotent CPSCF ...\n");
		std::printf("| Perturbation | # of iterations | Wall time |\n");
	}

	// Initialization
	int nbasis=coefficients.cols();
	EigenMatrix d=EigenZero(nbasis,nbasis);
	d.diagonal()=occupancies;
	d=coefficients*d*coefficients.transpose();

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
			dxn[ipert]=-N;
			for (int i=0;i<nbasis;i++)
				for (int j=0;j<=i;j++){
					if (std::abs(orbitalenergies[i]-orbitalenergies[j])<__small_value__) // Avoiding "division by zero" error.
						U(i,j)=U(j,i)=-0.5*csxc(i,j);
					else{
						U(i,j)=B(i,j)/(orbitalenergies[i]-orbitalenergies[j]);
						U(j,i)=-U(i,j)-csxc(i,j);
					}
					Mij=coefficients.col(i)*coefficients.col(j).transpose()+coefficients.col(j)*coefficients.col(i).transpose();
					dxn[ipert]+=(occupancies[j]-occupancies[i])*U(i,j)*Mij;
				}

			B=B1-coefficients.transpose()*GMatrix(repulsion,indices,n2integrals,dxn[ipert],kscale,nprocs)*coefficients; // Forming a new B matrix.
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

		if (wxn){
			wxn[ipert]=EigenZero(nbasis,nbasis);
			for (int i=0;i<nbasis;i++){
				wxn[ipert]-=occupancies[i]*B(i,i)*coefficients.col(i)*coefficients.col(i).transpose();
				for (int j=0;j<nbasis;j++){
					Mij=coefficients.col(i)*coefficients.col(j).transpose()+coefficients.col(j)*coefficients.col(i).transpose();
					wxn[ipert]+=occupancies[j]*orbitalenergies[j]*U(i,j)*Mij;
				}
			}
		}

		const std::chrono::duration<double> duration_wall=std::chrono::system_clock::now()-iterstart_wall;
		if (output) std::printf("|    %6d    |     %7d     | %9.6f |\n",ipert,jiter+1,duration_wall.count());
	}
}


