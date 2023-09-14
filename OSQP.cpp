#include <stdlib.h>
#include "osqp.h"

void QuadraticProgramming(int size,double * helements,int * hrowindeces,int * hcolpointers,double * g,double * aelements,int * arowindeces,int * acolpointers,int nconstraints,double * l,double * u,double * x){
	OSQPInt Hnnz=(1+size)*size/2;
	OSQPFloat * Helements=new OSQPFloat[Hnnz];
	OSQPInt * Hrowindeces=new OSQPInt[Hnnz];
	OSQPInt * Hcolpointers=new OSQPInt[size+1];Hcolpointers[0]=0;
	OSQPFloat * G=new OSQPFloat[size];
	for (int i=0;i<Hnnz;i++){
		Helements[i]=helements[i];
		Hrowindeces[i]=hrowindeces[i];
	}
	for (int i=0;i<size;i++){
		Hcolpointers[i+1]=hcolpointers[i+1];
		G[i]=g[i];
	}

	OSQPInt Annz=nconstraints*size;
	OSQPFloat * Aelements=new OSQPFloat[Annz];
	OSQPInt * Arowindeces=new OSQPInt[Annz];
	OSQPInt * Acolpointers=new OSQPInt[size+1];Acolpointers[0]=0;
	OSQPFloat * L=new OSQPFloat[Annz];
	OSQPFloat * U=new OSQPFloat[Annz];
	for (int i=0;i<Annz;i++){
		Aelements[i]=aelements[i];
		Arowindeces[i]=arowindeces[i];
		L[i]=l[i];
		U[i]=u[i];
	}
	for (int i=0;i<size+1;i++)
		Acolpointers[i]=acolpointers[i];
	OSQPCscMatrix * H=new OSQPCscMatrix;
	csc_set_data(H,size,size,Hnnz,Helements,Hrowindeces,Hcolpointers);
	OSQPCscMatrix * A=new OSQPCscMatrix;
	csc_set_data(A,nconstraints,size,Annz,Aelements,Arowindeces,Acolpointers);

	OSQPSettings * settings=new OSQPSettings;
	osqp_set_default_settings(settings);
	settings->polishing=1;
	settings->verbose=0;
	OSQPSolver * solver=nullptr;
	osqp_setup(&solver,H,G,A,L,U,nconstraints,size,settings);
	osqp_solve(solver);
	for (int i=0;i<size;i++)
		x[i]=solver->solution->x[i];

	osqp_cleanup(solver);
	if (H) delete H;
	if (A) delete A;
	if (settings) delete settings;
	delete [] Helements;
	delete [] Hrowindeces;
	delete [] Hcolpointers;
	delete [] G;
	delete [] Aelements;
	delete [] Arowindeces;
	delete [] Acolpointers;
	delete [] L;
	delete [] U;
}


