#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "Aliases.h"
#include "HartreeFock.h"
#include "LinearAlgebra.h"

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
	int nelements=(nbasis-1)*nbasis/2;
	EigenMatrix d=EigenZero(nbasis,nbasis);
	d.diagonal()=occupancies;
	d=coefficients*d*coefficients.transpose();

	for (int ipert=0;ipert<natoms*3;ipert++){

		// Initialization
		const auto iterstart_wall=std::chrono::system_clock::now();
		int jiter=0;
		EigenVector residue(nelements);residue[0]=114514;
		EigenVector U(nelements);

		// Forming B1, common for all iterations
		const EigenMatrix csxc=coefficients.transpose()*ovlgrads[ipert]*coefficients;
		const EigenMatrix cfxc=coefficients.transpose()*fskeletons[ipert]*coefficients;
		EigenVector B1(nelements);
		for (int i=0,ij=0;i<nbasis;i++)
			for (int j=0;j<i;j++,ij++)
				B1[ij]=csxc(i,j)*orbitalenergies(j)-cfxc(i,j);
		EigenVector B=B1;

		// Forming N, common for all iterations
		EigenMatrix N=EigenZero(nbasis,nbasis);
		EigenMatrix Mij=EigenZero(nbasis,nbasis);
		EigenMatrix Mii=EigenZero(nbasis,nbasis);
		for (int i=0,ij=0;i<nbasis;i++){
			for (int j=0;j<i;j++,ij++){
				Mij=coefficients.col(i)*coefficients.col(j).transpose()+coefficients.col(j)*coefficients.col(i).transpose();
				N+=occupancies[i]*csxc(i,j)*Mij;
			}
			Mii=coefficients.col(i)*coefficients.col(i).transpose()+coefficients.col(i)*coefficients.col(i).transpose();
			N+=0.5*occupancies[i]*csxc(i,i)*Mii;
		}

		for (jiter=0;residue.norm()>__convergence_threshold__;jiter++){

			dxn[ipert]=-N;
			for (int i=0,ij=0;i<nbasis;i++)
				for (int j=0;j<i;j++,ij++){
					if (std::abs(orbitalenergies[i]-orbitalenergies[j])<__small_value__ || std::abs(occupancies[i]-occupancies[j])<__small_value__)
						U[ij]=-0.5*csxc(i,j);
					else{
						U[ij]=B[ij]/(orbitalenergies[i]-orbitalenergies[j]);
					}
					Mij=coefficients.col(i)*coefficients.col(j).transpose()+coefficients.col(j)*coefficients.col(i).transpose();
					dxn[ipert]+=(occupancies[j]-occupancies[i])*U[ij]*Mij;
				}

			const EigenMatrix B2matrix=coefficients.transpose()*GMatrix(repulsion,indices,n2integrals,dxn[ipert],kscale,nprocs)*coefficients;
			B=B1;
			for (int i=0,ij=0;i<nbasis;i++)
				for (int j=0;j<i;j++,ij++)
					B[ij]-=B2matrix(i,j);
			for (int i=0,ij=0;i<nbasis;i++)
				for (int j=0;j<i;j++,ij++)
					residue[ij]=B[ij]-(orbitalenergies[i]-orbitalenergies[j])*U[ij];

		}

		const std::chrono::duration<double> duration_wall=std::chrono::system_clock::now()-iterstart_wall;
		if (output) std::printf("|    %6d    |     %7d     | %9.6f |\n",ipert,jiter+1,duration_wall.count());
	}
}


