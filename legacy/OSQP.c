#include <stdlib.h>
#include "osqp.h"

void QuadraticProgramming(int size,double * helements,int * hrowindeces,int * hcolpointers,double * g,double * aelements,int * arowindeces,int * acolpointers,int nconstraints,double * l,double * u,double * x){
	OSQPInt Hnnz=(1+size)*size/2;
	OSQPFloat Helements[Hnnz];
	OSQPInt Hrowindeces[Hnnz];
	OSQPInt Hcolpointers[size+1];Hcolpointers[0]=0;
	OSQPFloat G[size];
	for (int i=0;i<Hnnz;i++){
		Helements[i]=helements[i];
		Hrowindeces[i]=hrowindeces[i];
	}
	for (int i=0;i<size;i++){
		Hcolpointers[i+1]=hcolpointers[i+1];
		G[i]=g[i];
	}

	OSQPInt Annz=nconstraints*size;
	OSQPFloat Aelements[Annz];
	OSQPInt Arowindeces[Annz];
	OSQPInt Acolpointers[size+1];Acolpointers[0]=0;
	OSQPFloat L[Annz];
	OSQPFloat U[Annz];
	for (int i=0;i<Annz;i++){
		Aelements[i]=aelements[i];
		Arowindeces[i]=arowindeces[i];
		L[i]=l[i];
		U[i]=u[i];
	}
	for (int i=0;i<size+1;i++)
		Acolpointers[i]=acolpointers[i];
	OSQPCscMatrix * H=malloc(sizeof(OSQPCscMatrix));
	csc_set_data(H,size,size,Hnnz,Helements,Hrowindeces,Hcolpointers);
	OSQPCscMatrix * A=malloc(sizeof(OSQPCscMatrix));
	csc_set_data(A,nconstraints,size,Annz,Aelements,Arowindeces,Acolpointers);

	OSQPSettings * settings=malloc(sizeof(OSQPSettings));
	osqp_set_default_settings(settings);
	settings->polishing=1;
	settings->verbose=0;
	OSQPSolver * solver=NULL;
	osqp_setup(&solver,H,G,A,L,U,nconstraints,size,settings);
	osqp_solve(solver);
	for (int i=0;i<size;i++)
		x[i]=solver->solution->x[i];

	osqp_cleanup(solver);
	if (H) free(H);
	if (A) free(A);
	if (settings) free(settings);
}


