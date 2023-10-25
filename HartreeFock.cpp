#include <Eigen/Dense>
#include <cmath>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include "Aliases.h"
#include "Libint2.h"
#include "GridIntegrals.h"
#include "DensityFunctional.h"
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
#define __density_purification_niterations__ 1024
#define __density_purification_threshold__ 1.e-15

void PurifyDensity(EigenMatrix overlap,EigenMatrix & D){
	const int nbasis=D.rows();
	EigenMatrix error=EigenOne(nbasis,nbasis)*114514;
	for (int i=0;i<__density_purification_niterations__ && error.norm()>__density_purification_threshold__;i++){
		D=3*D*overlap*D-2*D*overlap*D*overlap*D;
		error=D*overlap*D*overlap-D*overlap;
	}
	assert("Density matrix purification failed. Please change to another initial guess." && error.norm()<__density_purification_threshold__);
}

EigenMatrix GMatrix(double * repulsion,short int * indices,long int n2integrals,EigenMatrix D,double kscale,const int nprocs){
	int nbasis=D.cols();
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
	EigenMatrix * rawjs=new EigenMatrix[nprocs];
	EigenMatrix * rawks=new EigenMatrix[nprocs];

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
		EigenMatrix * thisrawj=&rawjs[iproc];
		*thisrawj=EigenZero(nbasis,nbasis);
		EigenMatrix * thisrawk=&rawks[iproc];
		*thisrawk=EigenZero(nbasis,nbasis);
		// Without SIMD
		for (long int i=0;i<nints;i++){ // Manual loop unrolling does not help.
			deg_value=*(thisdegs++)**(thisrepulsions++); // Moving the ranger pointer to the right, where the next integral is located.
			a=*(thisbf1s++);
			b=*(thisbf2s++);
			c=*(thisbf3s++);
			d=*(thisbf4s++);
			(*thisrawj)(a,b)+=D(c,d)*deg_value;
			(*thisrawj)(c,d)+=D(a,b)*deg_value;
			if (kscale>0){
				(*thisrawk)(a,c)+=D(b,d)*deg_value;
				(*thisrawk)(b,d)+=D(a,c)*deg_value;
				(*thisrawk)(a,d)+=D(b,c)*deg_value;
				(*thisrawk)(b,c)+=D(a,d)*deg_value;
			}
		}
	}
	Eigen::setNbThreads(nprocs);
	EigenMatrix rawj=EigenZero(nbasis,nbasis);
	EigenMatrix rawk=EigenZero(nbasis,nbasis);
	for (int iproc=0;iproc<nprocs;iproc++){
		rawj+=rawjs[iproc];
		rawjs[iproc].resize(0,0);
		if (kscale>0){
			rawk+=rawks[iproc];
			rawks[iproc].resize(0,0);
		}
	}
	delete [] iintfirstperthread;
	delete [] nintsperthread;
	delete [] rawjs;
	delete [] rawks;
	EigenMatrix j=0.5*(rawj+rawj.transpose());
	EigenMatrix k=0.25*(rawk+rawk.transpose());
	EigenMatrix g=j-0.5*kscale*k;
	return g;
}

#define __Density_2_Fock__\
	F=hcore+GMatrix(repulsion,indices,n2integrals,D,kscale,nprocs); /* Fock matrix. */\
	if (dfxid){\
		GetDensity(aos,\
		           ao1xs,ao1ys,ao1zs,\
		           ao2ls,\
		           ngrids,2*D,\
		           ds,\
		           d1xs,d1ys,d1zs,cgs,\
		           d2s,ts);\
		getEVxc(dfxid,ds,cgs,d2s,ts,ngrids,excs,vrxcs,vsxcs,vlxcs,vtxcs);\
		if (dfcid && dfxid!=dfcid){\
			getEVxc(dfcid,ds,cgs,d2s,ts,ngrids,ecs,vrcs,vscs,vlcs,vtcs);\
			VectorAddition(excs,ecs,ngrids);\
			VectorAddition(vrxcs,vrcs,ngrids);\
			VectorAddition(vsxcs,vscs,ngrids);\
			VectorAddition(vlxcs,vlcs,ngrids);\
			VectorAddition(vtxcs,vtcs,ngrids);\
		}\
		Exc=SumUp(excs,gridweights,ngrids);\
		Fxc=FxcMatrix(aos,vrxcs,\
                              d1xs,d1ys,d1zs,\
                              ao1xs,ao1ys,ao1zs,vsxcs,\
		              ao2ls,vlxcs,vtxcs,\
                              gridweights,ngrids,nbasis);\
		F+=Fxc;\
	}\
	G=4*(overlap*D*F-F*D*overlap); // Electronic gradient of energy with respect to nonredundant orbital rotational parameters. [F(D),D] instead of [F,D(F)].

#define __Fock_2_Density__\
	const EigenMatrix Fprime=X.transpose()*F*X;\
	eigensolver.compute(Fprime);\
	orbitalenergies=eigensolver.eigenvalues();\
	const EigenMatrix Cprime=eigensolver.eigenvectors();\
	coefficients=X*Cprime;\
	D=EigenZero(nbasis,nbasis);\
	if (!std::isnan(temperature+chemicalpotential))\
		for (int i=0;i<nbasis;i++){\
			occupation[i]=1./(1.+std::exp((orbitalenergies[i]-chemicalpotential)/temperature));\
		}\
	else{\
		for (int i=0;i<nbasis;i++)\
			occupation[i]=(double)(i<nocc);\
	}\
	D.diagonal()=occupation;\
	D=coefficients*D*coefficients.transpose();\
	//PurifyDensity(overlap,D);

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

double RKS(int nele,double temperature,double chemicalpotential,
           EigenMatrix overlap,EigenMatrix hcore,
           double * repulsion,short int * indices,long int n2integrals,
           int dfxid,int dfcid,int ngrids,double * gridweights,
           double * aos,
           double * ao1xs,double * ao1ys,double * ao1zs,
           double * ao2ls,
           double *& d1xs,double *& d1ys,double *& d1zs,
           double *& vrxcs,double *& vsxcs,
           EigenVector & orbitalenergies,EigenMatrix & coefficients,
           EigenVector & occupation,EigenMatrix & D,EigenMatrix & F,
           const int nprocs,const bool output){
	if (output){
		std::printf("Restricted ");
		if (std::isnormal(temperature+chemicalpotential) || temperature+chemicalpotential==0)
			std::printf("finite-temperature ");
		else if (std::isnormal(temperature))
			std::printf("thermally-assisted-occupation ");
		if (dfxid) std::printf("Kohn-Sham ...\n");
		else std::printf("Hartree-Fock ...\n");
		if (std::isnormal(temperature+chemicalpotential) || temperature+chemicalpotential==0)
			std::printf("| Iteration | Fock update |    Grand potential    | Gradient norm | Wall time |\n");
		else if (std::isnormal(temperature))
			std::printf("| Iteration | Fock update |      Free energy      | Gradient norm | Wall time |\n");
		else
			std::printf("| Iteration | Fock update |        Energy         | Gradient norm | Wall time |\n");
	}

	// General preparation
	Eigen::setNbThreads(nprocs);
	const int nbasis=overlap.cols();
	const int nocc=nele/2;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(overlap);
	const EigenMatrix s=eigensolver.eigenvalues();
	EigenMatrix sinversesqrt=EigenZero(nbasis,nbasis);
	for (int i=0;i<nbasis;i++)
		sinversesqrt(i,i)=1/sqrt(s(i,0));
	const EigenMatrix U=eigensolver.eigenvectors();
	const EigenMatrix X=U*sinversesqrt*U.transpose();

	EigenMatrix G=EigenZero(nbasis,nbasis);
	EigenMatrix Fxc=EigenZero(nbasis,nbasis);

	// KS preparation
	double Exc=0;
	double * ds=nullptr;
	double * d2s=nullptr;
	double * ts=nullptr;
	double * cgs=nullptr;
	double * excs=nullptr;
	double * vlxcs=nullptr;
	double * vtxcs=nullptr;
	double * ecs=nullptr;
	double * vrcs=nullptr;
	double * vscs=nullptr;
	double * vlcs=nullptr;
	double * vtcs=nullptr;
	int xkind,ckind,xfamily,cfamily;
	xkind=ckind=xfamily=cfamily=114514;
	double kscale=1;
	if (dfxid){
		ds=new double[ngrids]();
		d1xs=new double[ngrids]();
		d1ys=new double[ngrids]();
		d1zs=new double[ngrids]();
		d2s=new double[ngrids]();
		ts=new double[ngrids]();
		cgs=new double[ngrids]();
		char rubbish[64];
		if(dfcid){
			XCInfo(dfcid,rubbish,ckind,cfamily,kscale);
			ecs=new double[ngrids]();
			vrcs=new double[ngrids]();
			vscs=new double[ngrids]();
			vlcs=new double[ngrids]();
			vtcs=new double[ngrids]();
		}
		XCInfo(dfxid,rubbish,xkind,xfamily,kscale);
		excs=new double[ngrids]();
		vrxcs=new double[ngrids]();
		vsxcs=new double[ngrids]();
		vlxcs=new double[ngrids]();
		vtxcs=new double[ngrids]();
	}

	// Initial guess
	if (D.rows()){__Density_2_Fock__}
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
		Ds[i]=D;
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
		if (output) std::printf("|   %5d   |",iiteration);
		const auto iterstart_wall=std::chrono::system_clock::now();

		// Convergence techniques
		if (iiteration==0 && nlbfgs==-1){
			if (output) std::printf("    naive    |");
			update='f';
			F=Fs[0];
		}else if (abs(Es[0]-Es[1])>__damping_start_threshold__ && nlbfgs==-1){ // Using damping in the beginning and when energy oscillates in a large number of iterations.
			if (output) std::printf("   damping   |");
			update='f';
			F=(1.-__damping_factor__)*Fs[0]+__damping_factor__*Fs[1];
		}else if (__diis_start_iter__>=iiteration && iiteration>__adiis_start_iter__ && nlbfgs==-1){ // Starting A-DIIS in the beginning to facilitate (but not necesarily accelerate) convergence.
			if (output) std::printf("    ADIIS    |");
			update='f';
			F=AEDIIS('a',Es,Ds,Fs,iiteration<__diis_space_size__?iiteration:__diis_space_size__);
		}else if (iiteration>__diis_start_iter__ && pG.norm()>__lbfgs_start_threshold__ && nlbfgs==-1){ // Starting DIIS after A-DIIS to accelerate convergence in the medium-gradient area.
			if (output) std::printf("    CDIIS    |");
			update='f';
			F=DIIS(Fs,Gs,iiteration<__diis_space_size__?iiteration:__diis_space_size__,error2norm); // error2norm is updated.
		}else if ((iiteration>__diis_start_iter__ && pG.norm()<__lbfgs_start_threshold__) || nlbfgs>=0){ // Stopping DIIS for ASOSCF (or simply L-BFGS) in the final part to prevent trailing.
			EigenVector hessiandiag(nocc*(nbasis-nocc));
			hessiandiag.setZero();
			if (output) std::printf("    L-BFGS   |");
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
			D=C_occ*C_occ.transpose();
			nlbfgs++;
		}else{
			if (output) std::printf("    naive    |");
			update='f';
			F=Fs[0];
		}

		if (update=='f'){__Fock_2_Density__} // Some techniques update Fock matrix. To complete the iteration, we obtain density matrix from it.

		// Common procedures necessary for all convergence techniques
		__Density_2_Fock__
		PushMatrixQueue(F,Fs,__diis_space_size__); // Updating Fock matrix.
		PushMatrixQueue(sinversesqrt.transpose()*G*sinversesqrt,Gs,__diis_space_size__); // Updating gradient. I don't know why sinversesqrt is important, but its existence accelerates convergence.
		PushMatrixQueue(D,Ds,__diis_space_size__); // Updating atomic density matrix.

		const EigenMatrix Fmo=coefficients.transpose()*F*coefficients;
		__Loop_Over_OV__ pG(i)=-Fmo(o,v)*(occupation[o]-occupation[v]);
		PushVectorQueue(pG,pGs,__lbfgs_space_size__+1); // Updating packed gradient matrix.
		PushVectorQueue(pX,pXs,__lbfgs_space_size__+1);

		if (update=='d'){__Fock_2_Density__} // Some techniques update Fock matrix. To complete the iteration, we obtain density matrix from it.

		// Iteration information output
		const EigenMatrix Iamgoingtobeanobellaureate=D*(hcore+F-Fxc); // Intermediate matrix.
		PushDoubleQueue(Iamgoingtobeanobellaureate.trace()+Exc,Es,__diis_space_size__); // Updating energy.

		if (std::isnormal(temperature)){
			double entropy=0;
			for (int i=0;i<nbasis;i++)
				entropy+=std::log(std::pow(occupation[i],occupation[i]))+std::log(std::pow(1.-occupation[i],1.-occupation[i]));
			Es[0]-=temperature*2*entropy;
			if (std::isnormal(temperature+chemicalpotential) || temperature+chemicalpotential==0)
				Es[0]-=chemicalpotential*2*occupation.sum();
		}

		const std::chrono::duration<double> duration_wall=std::chrono::system_clock::now()-iterstart_wall;
		if (output) std::printf("   %17.10f   | %13.6f | %9.6f |\n",Es[0],pG.norm(),duration_wall.count());
		iiteration++;
	}while (abs(Es[0]-Es[1])>__scf_convergence_energy_threshold__ || G.norm()>__scf_convergence_gradient_threshold__);
	if (output) std::cout<<"| Done; Final SCF energy = "<<std::setprecision(12)<<Es[0]<<" a.u."<<std::endl;
	const double energy=Es[0];

	// Printing orbital information
	if (output){
		std::printf("Orbital information ...\n");
		std::printf("| Index |  Energy (Eh)  |  Energy (eV)  | Occupancy |\n");
		for (int i=0;i<nbasis;i++)
			std::printf("| %4d  | %13.6f | %13.6f | %9.6f |\n",i,orbitalenergies[i],orbitalenergies[i]*__hartree2ev__,2*occupation[i]);
		std::cout<<"Total number of electrons ... "<<occupation.sum()*2<<std::endl;
	}

	if (ds) delete [] ds;
	if (d2s) delete [] d2s;
	if (ts) delete [] ts;
	if (cgs) delete [] cgs;
	if (ecs) delete [] ecs;
	if (vrcs) delete [] vrcs;
	if (vscs) delete [] vscs;
	if (vlxcs) delete [] vlxcs;
	if (vlcs) delete [] vlcs;
	if (vtxcs) delete [] vtxcs;
	if (vtcs) delete [] vtcs;

	for (int i=0;i<__diis_space_size__;i++){
		Fs[i].resize(0,0);
		Gs[i].resize(0,0);
		Ds[i].resize(0,0);
	}
	for (int i=0;i<__lbfgs_space_size__;i++){
		pGs[i].resize(0);
		pXs[i].resize(0);
	}
	delete [] Es;
	delete [] Fs;
	delete [] Gs;
	delete [] Ds;
	delete [] pGs;
	delete [] pXs;
	return energy;
}

double RHF(int nele,double temperature,double chemicalpotential,
           EigenMatrix overlap,EigenMatrix hcore,
           double * repulsion,short int * indices,long int n2integrals,
           EigenVector & orbitalenergies,EigenMatrix & coefficients,
           EigenVector & occupation,EigenMatrix & D,EigenMatrix & F,
           const int nprocs,const bool output){
	double * dummy=nullptr;
	return RKS(nele,temperature,chemicalpotential,
	           overlap,hcore,
                   repulsion,indices,n2integrals,
                   0,0,0,nullptr,
	           nullptr,
	           nullptr,nullptr,nullptr,
	           nullptr,
	           dummy,dummy,dummy,
	           dummy,dummy,
                   orbitalenergies,coefficients,
	           occupation,D,F,
                   nprocs,output);
}

