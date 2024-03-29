#include <Eigen/Dense>
#include <libint2.hpp>
#include <cmath>
#include <ctime>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include "Aliases.h"
#include "GridIntegrals.h"
#include "DensityFunctional.h"

#define __Loop_Over_XYZ__(iatom,position)\
	for (int t=0;t<3;t++){\
		tmp=abcd_deg*buf_vec[3*(position)+t][f1234];\
		rawjskeletons[3*(iatom)+t](bf1,bf2)+=tmp*D(bf3,bf4);\
		rawjskeletons[3*(iatom)+t](bf3,bf4)+=tmp*D(bf1,bf2);\
		if (kscale>0){\
			rawkskeletons[3*(iatom)+t](bf1,bf3)+=tmp*D(bf2,bf4);\
			rawkskeletons[3*(iatom)+t](bf2,bf4)+=tmp*D(bf1,bf3);\
			rawkskeletons[3*(iatom)+t](bf1,bf4)+=tmp*D(bf2,bf3);\
			rawkskeletons[3*(iatom)+t](bf2,bf3)+=tmp*D(bf1,bf4);\
		}\
	}

void GhfSkeletons(const int natoms,double * atoms,const char * basisset,
		EigenMatrix D,
                double kscale,
                EigenMatrix * ghfskeletons){
 __Basis_From_Atoms__
 __nBasis_From_OBS__

 EigenMatrix * rawjskeletons=new EigenMatrix[3*natoms];
 EigenMatrix * rawkskeletons=new EigenMatrix[3*natoms];
 for (int it=0;it<3*natoms;it++){
  rawjskeletons[it]=EigenZero(nbasis,nbasis);
  rawkskeletons[it]=EigenZero(nbasis,nbasis);
 }
 libint2::initialize();
 libint2::Engine engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l(),1);
 const auto & buf_vec=engine.results();
 auto shell2bf=obs.shell2bf();
 auto shell2atom=obs.shell2atom(libint2atoms);
 int atomlist[]={114,514,1919,810};
 for (short int s1=0;s1<(short int)obs.size();s1++){
  atomlist[0]=shell2atom[s1];
  const short int bf1_first=shell2bf[s1];
  const short int n1=obs[s1].size();
  for (short int s2=0;s2<=s1;s2++){
   atomlist[1]=shell2atom[s2];
   const short int bf2_first=shell2bf[s2];
   const short int n2=obs[s2].size();
   for (short int s3=0;s3<=s1;s3++){
    atomlist[2]=shell2atom[s3];
    const short int bf3_first=shell2bf[s3];
    const short int n3=obs[s3].size();
    for (short int s4=0;s4<=std::max(s2,s3);s4++){
     atomlist[3]=shell2atom[s4];
     const short int bf4_first=shell2bf[s4];
     const short int n4=obs[s4].size();
     engine.compute(obs[s1],obs[s2],obs[s3],obs[s4]);
     if (! buf_vec[0]) continue;
     for (short int f1=0,f1234=0;f1!=n1;f1++){
      const short int bf1=bf1_first+f1;
      for (short int f2=0;f2!=n2;f2++){
       const short int bf2=bf2_first+f2;
       const double ab_deg=(bf1==bf2)?1:2;
       for (short int f3=0;f3!=n3;f3++){
        const short int bf3=bf3_first+f3;
        for (short int f4=0;f4!=n4;f4++,f1234++){
         const short int bf4=bf4_first+f4;
         if (bf2<=bf1 && bf3<=bf1 && bf4<=((bf1==bf3)?bf2:bf3)){
          const double cd_deg=(bf3==bf4)?1:2;
          const double ab_cd_deg=(bf1==bf3)?(bf2==bf4?1:2):2;
          const double abcd_deg=ab_deg*cd_deg*ab_cd_deg;
          double tmp=114514;
          int xpert=1919810;
	  for (int p=0,pt=0;p<4;p++){
           for (int t=0;t<3;t++,pt++){
            tmp=abcd_deg*buf_vec[pt][f1234];
            xpert=3*(atomlist[p])+t;
            rawjskeletons[xpert](bf1,bf2)+=tmp*D(bf3,bf4);
            rawjskeletons[xpert](bf3,bf4)+=tmp*D(bf1,bf2);
            if (kscale>0){
             rawkskeletons[xpert](bf1,bf3)+=tmp*D(bf2,bf4);
             rawkskeletons[xpert](bf2,bf4)+=tmp*D(bf1,bf3);
             rawkskeletons[xpert](bf1,bf4)+=tmp*D(bf2,bf3);
             rawkskeletons[xpert](bf2,bf3)+=tmp*D(bf1,bf4);
            }
           }
          }
         }
        }
       }
      }
     }
    }
   }
  }
 }
 libint2::finalize();
 for (int it=0;it<3*natoms;it++){
  const EigenMatrix jskeleton=0.5*(rawjskeletons[it]+rawjskeletons[it].transpose());
  const EigenMatrix kskeleton=0.25*(rawkskeletons[it]+rawkskeletons[it].transpose());
  ghfskeletons[it]+=jskeleton-0.5*kscale*kskeleton;
  rawjskeletons[it].resize(0,0);
  rawkskeletons[it].resize(0,0);
 }
 delete [] rawjskeletons;
 delete [] rawkskeletons;
}

void GxcSkeletons(const int natoms,double * atoms,const char * basisset,
		EigenMatrix D,
                int dfxid,int dfcid,int ngrids,double * ws,
                double * aos,
                double * ao1xs,double * ao1ys,double * ao1zs,
                double * ao2xxs,double * ao2yys,double * ao2zzs,
                double * ao2xys,double * ao2xzs,double * ao2yzs,
                EigenMatrix * gxcskeletons){
 __Basis_From_Atoms__
 __nBasis_From_OBS__
 if (!aos) return;
 __Initialize_KS__(aos,ao1xs,ao1ys,ao1zs,0,0,0)
	GetDensity(
			aos,
			ao1xs,ao1ys,ao1zs,
			0,
			ngrids,2*D,
			ds,
			d1xs,d1ys,d1zs,cgs,
			d2s,ts);
 __KS_Potential__(0,0)

 auto shell2bf=obs.shell2bf();
 auto shell2atom=obs.shell2atom(libint2atoms);
 for (short int s1=0;s1<(short int)obs.size() && aos && ao1xs;s1++){
  const short int atom1=shell2atom[s1];
  const short int bf1_first=shell2bf[s1];
  const short int n1=obs[s1].size();
  for (short int s2=0;s2<=s1;s2++){
   const short int atom2=shell2atom[s2];
   const short int bf2_first=shell2bf[s2];
   const short int n2=obs[s2].size();
   for (short int f1=0;f1!=n1;f1++){
    const short int bf1=bf1_first+f1;
    double * thisaos=aos+bf1*ngrids;
    double * thisao1xs=ao1xs+bf1*ngrids;
    double * thisao1ys=ao1ys+bf1*ngrids;
    double * thisao1zs=ao1zs+bf1*ngrids;
    double * thisao2xxs,*thisao2yys,*thisao2zzs,*thisao2xys,*thisao2xzs,*thisao2yzs;
    thisao2xxs=thisao2yys=thisao2zzs=thisao2xys=thisao2xzs=thisao2yzs=nullptr;
    if (ao2xxs){
     thisao2xxs=ao2xxs+bf1*ngrids;
     thisao2yys=ao2yys+bf1*ngrids;
     thisao2zzs=ao2zzs+bf1*ngrids;
     thisao2xys=ao2xys+bf1*ngrids;
     thisao2xzs=ao2xzs+bf1*ngrids;
     thisao2yzs=ao2yzs+bf1*ngrids;
    }
    for (short int f2=0;f2!=n2;f2++){
     const short int bf2=bf2_first+f2;
     if (bf2>bf1) break;
     double * thataos=aos+bf2*ngrids;
     double * thatao1xs=ao1xs+bf2*ngrids;
     double * thatao1ys=ao1ys+bf2*ngrids;
     double * thatao1zs=ao1zs+bf2*ngrids;
     double * thatao2xxs,*thatao2yys,*thatao2zzs,*thatao2xys,*thatao2xzs,*thatao2yzs;
     thatao2xxs=thatao2yys=thatao2zzs=thatao2xys=thatao2xzs=thatao2yzs=nullptr;
     if (ao2xxs){
      thatao2xxs=ao2xxs+bf2*ngrids;
      thatao2yys=ao2yys+bf2*ngrids;
      thatao2zzs=ao2zzs+bf2*ngrids;
      thatao2xys=ao2xys+bf2*ngrids;
      thatao2xzs=ao2xzs+bf2*ngrids;
      thatao2yzs=ao2yzs+bf2*ngrids;
     }
     double tmp=114514;
     for (long int kgrid=0;kgrid<ngrids;kgrid++){
      tmp=-2*ws[kgrid]*vrxcs[kgrid];
      gxcskeletons[3*atom1+0](bf1,bf2)+=tmp*thisao1xs[kgrid]*thataos[kgrid];
      gxcskeletons[3*atom1+1](bf1,bf2)+=tmp*thisao1ys[kgrid]*thataos[kgrid];
      gxcskeletons[3*atom1+2](bf1,bf2)+=tmp*thisao1zs[kgrid]*thataos[kgrid];
      if (bf2!=bf1){
       gxcskeletons[3*atom2+0](bf2,bf1)+=tmp*thatao1xs[kgrid]*thisaos[kgrid];
       gxcskeletons[3*atom2+1](bf2,bf1)+=tmp*thatao1ys[kgrid]*thisaos[kgrid];
       gxcskeletons[3*atom2+2](bf2,bf1)+=tmp*thatao1zs[kgrid]*thisaos[kgrid];
      }
      if (ao2xxs){
       tmp=-4*ws[kgrid]*vsxcs[kgrid];
       gxcskeletons[3*atom1+0](bf1,bf2)+=tmp*(d1xs[kgrid]*(thisao2xxs[kgrid]*thataos[kgrid]+thatao1xs[kgrid]*thisao1xs[kgrid])
                                             +d1ys[kgrid]*(thisao2xys[kgrid]*thataos[kgrid]+thatao1ys[kgrid]*thisao1xs[kgrid])
                                             +d1zs[kgrid]*(thisao2xzs[kgrid]*thataos[kgrid]+thatao1zs[kgrid]*thisao1xs[kgrid]));
       gxcskeletons[3*atom1+1](bf1,bf2)+=tmp*(d1xs[kgrid]*(thisao2xys[kgrid]*thataos[kgrid]+thatao1xs[kgrid]*thisao1ys[kgrid])
                                             +d1ys[kgrid]*(thisao2yys[kgrid]*thataos[kgrid]+thatao1ys[kgrid]*thisao1ys[kgrid])
                                             +d1zs[kgrid]*(thisao2yzs[kgrid]*thataos[kgrid]+thatao1zs[kgrid]*thisao1ys[kgrid]));
       gxcskeletons[3*atom1+2](bf1,bf2)+=tmp*(d1xs[kgrid]*(thisao2xzs[kgrid]*thataos[kgrid]+thatao1xs[kgrid]*thisao1zs[kgrid])
                                             +d1ys[kgrid]*(thisao2yzs[kgrid]*thataos[kgrid]+thatao1ys[kgrid]*thisao1zs[kgrid])
                                             +d1zs[kgrid]*(thisao2zzs[kgrid]*thataos[kgrid]+thatao1zs[kgrid]*thisao1zs[kgrid]));
       if (bf2!=bf1){
        gxcskeletons[3*atom2+0](bf2,bf1)+=tmp*(d1xs[kgrid]*(thatao2xxs[kgrid]*thisaos[kgrid]+thisao1xs[kgrid]*thatao1xs[kgrid])
                                              +d1ys[kgrid]*(thatao2xys[kgrid]*thisaos[kgrid]+thisao1ys[kgrid]*thatao1xs[kgrid])
                                              +d1zs[kgrid]*(thatao2xzs[kgrid]*thisaos[kgrid]+thisao1zs[kgrid]*thatao1xs[kgrid]));
        gxcskeletons[3*atom2+1](bf2,bf1)+=tmp*(d1xs[kgrid]*(thatao2xys[kgrid]*thisaos[kgrid]+thisao1xs[kgrid]*thatao1ys[kgrid])
                                              +d1ys[kgrid]*(thatao2yys[kgrid]*thisaos[kgrid]+thisao1ys[kgrid]*thatao1ys[kgrid])
                                              +d1zs[kgrid]*(thatao2yzs[kgrid]*thisaos[kgrid]+thisao1zs[kgrid]*thatao1ys[kgrid]));
        gxcskeletons[3*atom2+2](bf2,bf1)+=tmp*(d1xs[kgrid]*(thatao2xzs[kgrid]*thisaos[kgrid]+thisao1xs[kgrid]*thatao1zs[kgrid])
                                              +d1ys[kgrid]*(thatao2yzs[kgrid]*thisaos[kgrid]+thisao1ys[kgrid]*thatao1zs[kgrid])
                                              +d1zs[kgrid]*(thatao2zzs[kgrid]*thisaos[kgrid]+thisao1zs[kgrid]*thatao1zs[kgrid]));
       }
      }
     }
    }
   }
  }
 }
 __Finalize_KS__;
 for (int it=0;it<natoms*3;it++){
  const EigenMatrix tmp=gxcskeletons[it];
  gxcskeletons[it]=0.5*(tmp+tmp.transpose());
 }
}

// This one is slower but more concise in maths. It is used when G skeleton derivatives are requested.
EigenMatrix RKSG_concise(
		const int natoms,double * atoms,const char * basisset,
		EigenMatrix * ovlgrads,EigenMatrix * hcoregrads,EigenMatrix * fskeletons,
		int dfxid,int dfcid,int ngrids,double * ws,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2xxs,double * ao2yys,double * ao2zzs,
		double * ao2xys,double * ao2xzs,double * ao2yzs,
		EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupancies,
		const int nprocs,const bool output){
 if (output && aos) std::cout<<"Calculating repulsion integral derivatives w.r.t nuclear coordinates and summing up RKS gradient ... ";
 if (output && !aos) std::cout<<"Calculating repulsion integral derivatives w.r.t nuclear coordinates and summing up RHF gradient ... ";
 time_t start=time(0);
 __Basis_From_Atoms__
 __nBasis_From_OBS__
 // One-electron contributions
 EigenMatrix D=EigenZero(nbasis,nbasis);
 EigenMatrix W=EigenZero(nbasis,nbasis);
 for (int i=0;i<nbasis;i++){
    D+=occupancies[i]*coefficients.col(i)*coefficients.col(i).transpose();
    W+=occupancies[i]*coefficients.col(i)*coefficients.col(i).transpose()*orbitalenergies[i];
 }
 EigenMatrix gradient1=EigenZero(natoms,3);
 for (int iatom=0,it=0;iatom<natoms;iatom++)
  for (int t=0;t<3;t++,it++)
   gradient1(iatom,t)=2*(D*hcoregrads[it]).trace()-2*(W*ovlgrads[it]).trace();

 // Two-electron contributions
 EigenMatrix * ghfskeletons=new EigenMatrix[3*natoms];
 EigenMatrix * gxcskeletons=new EigenMatrix[3*natoms];
 for (int iatom=0,it=0;iatom<natoms;iatom++)
  for (int t=0;t<3;t++,it++)
   ghfskeletons[it]=gxcskeletons[it]=EigenZero(nbasis,nbasis);
 char xname[64];
 int xkind=1919;
 int xfamily=810;
 double kscale=1;
 XCInfo(dfxid,xname,xkind,xfamily,kscale);
 GhfSkeletons(
		natoms,atoms,basisset,
		D,
		kscale,
		ghfskeletons);
 GxcSkeletons(
		natoms,atoms,basisset,
	 	D,
		dfxid,dfcid,ngrids,ws,
		aos,
		ao1xs,ao1ys,ao1zs,
		ao2xxs,ao2yys,ao2zzs,
		ao2xys,ao2xzs,ao2yzs,
		gxcskeletons);
 EigenMatrix gradient2=EigenZero(natoms,3);
 //std::cout<<hcoregrads[0]+ghfskeletons[0]+0.5*gxcskeletons[0]<<std::endl;
 for (int iatom=0,it=0;iatom<natoms;iatom++)
  for (int t=0;t<3;t++,it++)
   gradient2(iatom,t)+=(D*(ghfskeletons[it]+2*gxcskeletons[it])).trace(); // The factor 2 stems from two Gxc skeleton matrices, Gxc^alpha and Gxc^beta.

 // Saving skeleton derivative of Fock matrix
 if (fskeletons)
  for (int it=0;it<3*natoms;it++)
   fskeletons[it]=hcoregrads[it]+ghfskeletons[it]+gxcskeletons[it];

 __Delete_Matrices__(ghfskeletons,3*natoms);
 __Delete_Matrices__(gxcskeletons,3*natoms);
 time_t end=time(0);
 if (output) std::cout<<"done "<<end-start<<" s"<<std::endl;
 return gradient1+gradient2;
}

// This one is faster but uglier. It is used when G matrix skeleton derivatives are not needed.
EigenMatrix RKSG_fast(
		const int natoms,double * atoms,const char * basisset,
		EigenMatrix * ovlgrads,EigenMatrix * hcoregrads,
		int dfxid,int dfcid,int ngrids,double * ws,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2xxs,double * ao2yys,double * ao2zzs,
		double * ao2xys,double * ao2xzs,double * ao2yzs,
		EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupancies,
		const int nprocs,const bool output){
	if (output && aos) std::cout<<"Calculating repulsion integral derivatives w.r.t nuclear coordinates and summing up RKS gradient ... ";
	if (output && !aos) std::cout<<"Calculating repulsion integral derivatives w.r.t nuclear coordinates and summing up RHF gradient ... ";
	time_t start=time(0);
	__Basis_From_Atoms__
	__nBasis_From_OBS__

	EigenMatrix D=EigenZero(nbasis,nbasis);
	EigenMatrix W=EigenZero(nbasis,nbasis);
	for (int i=0;i<nbasis;i++){
		D+=occupancies[i]*coefficients.col(i)*coefficients.col(i).transpose();
		W+=occupancies[i]*coefficients.col(i)*coefficients.col(i).transpose()*orbitalenergies[i];
	}

	__Initialize_KS__(aos,ao1xs,ao1ys,ao1zs,0,0,0)
		GetDensity(
				aos,
				ao1xs,ao1ys,ao1zs,
				0,
				ngrids,2*D,
				ds,
				d1xs,d1ys,d1zs,cgs,
				d2s,ts);
	__KS_Potential__(0,0)
	EigenMatrix Ghf=EigenZero(natoms,3);
	EigenMatrix Gxc=EigenZero(natoms,3);
	EigenMatrix * rawjskeletons=new EigenMatrix[3*natoms];
	EigenMatrix * rawkskeletons=new EigenMatrix[3*natoms];
	for (int i=0;i<3*natoms;i++)
		 rawjskeletons[i]=rawkskeletons[i]=EigenZero(nbasis,nbasis);
 libint2::initialize();
 libint2::Engine engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l(),1);
 const auto & buf_vec=engine.results();
 auto shell2bf=obs.shell2bf();
 auto shell2atom=obs.shell2atom(libint2atoms);
 for (short int s1=0;s1<(short int)obs.size();s1++){
  const short int atom1=shell2atom[s1];
  const short int bf1_first=shell2bf[s1];
  const short int n1=obs[s1].size();
  for (short int s2=0;s2<=s1;s2++){
   const short int atom2=shell2atom[s2];
   const short int bf2_first=shell2bf[s2];
   const short int n2=obs[s2].size();
   for (short int s3=0;s3<=s1;s3++){
    const short int atom3=shell2atom[s3];
    const short int bf3_first=shell2bf[s3];
    const short int n3=obs[s3].size();
    for (short int s4=0;s4<=std::max(s2,s3);s4++){
     const short int atom4=shell2atom[s4];
     const short int bf4_first=shell2bf[s4];
     const short int n4=obs[s4].size();
     engine.compute(obs[s1],obs[s2],obs[s3],obs[s4]);
     if (! buf_vec[0]) continue;
     for (short int f1=0,f1234=0;f1!=n1;f1++){
      const short int bf1=bf1_first+f1;
      for (short int f2=0;f2!=n2;f2++){
       const short int bf2=bf2_first+f2;
       const double ab_deg=(bf1==bf2)?1:2;
       for (short int f3=0;f3!=n3;f3++){
        const short int bf3=bf3_first+f3;
        for (short int f4=0;f4!=n4;f4++,f1234++){
         const short int bf4=bf4_first+f4;
         if (bf2<=bf1 && bf3<=bf1 && bf4<=((bf1==bf3)?bf2:bf3)){
          const double cd_deg=(bf3==bf4)?1:2;
          const double ab_cd_deg=(bf1==bf3)?(bf2==bf4?1:2):2;
          const double abcd_deg=ab_deg*cd_deg*ab_cd_deg;
          double tmp=114514;
          if (kscale==1)
           tmp=abcd_deg*(4*D(bf1,bf2)*D(bf3,bf4)-(D(bf1,bf3)*D(bf2,bf4)+D(bf1,bf4)*D(bf2,bf3)));
          else if (kscale==0)
           tmp=abcd_deg*(4*D(bf1,bf2)*D(bf3,bf4));
          else
           tmp=abcd_deg*(4*D(bf1,bf2)*D(bf3,bf4)-kscale*(D(bf1,bf3)*D(bf2,bf4)+D(bf1,bf4)*D(bf2,bf3)));
          Ghf(atom1,0)+=tmp*buf_vec[0][f1234];
          Ghf(atom1,1)+=tmp*buf_vec[1][f1234];
          Ghf(atom1,2)+=tmp*buf_vec[2][f1234];
          Ghf(atom2,0)+=tmp*buf_vec[3][f1234];
          Ghf(atom2,1)+=tmp*buf_vec[4][f1234];
          Ghf(atom2,2)+=tmp*buf_vec[5][f1234];
          Ghf(atom3,0)+=tmp*buf_vec[6][f1234];
          Ghf(atom3,1)+=tmp*buf_vec[7][f1234];
          Ghf(atom3,2)+=tmp*buf_vec[8][f1234];
          Ghf(atom4,0)+=tmp*buf_vec[9][f1234];
          Ghf(atom4,1)+=tmp*buf_vec[10][f1234];
          Ghf(atom4,2)+=tmp*buf_vec[11][f1234];
         }
        }
       }
      }
     }
    }
   }
   if (aos && ao1xs){
    for (short int f1=0;f1!=n1;f1++){
     const short int bf1=bf1_first+f1;
     double * thisaos=aos+bf1*ngrids;
     double * thisao1xs=ao1xs+bf1*ngrids;
     double * thisao1ys=ao1ys+bf1*ngrids;
     double * thisao1zs=ao1zs+bf1*ngrids;
     double * thisao2xxs,*thisao2yys,*thisao2zzs,*thisao2xys,*thisao2xzs,*thisao2yzs;
     thisao2xxs=thisao2yys=thisao2zzs=thisao2xys=thisao2xzs=thisao2yzs=nullptr;
     if (ao2xxs){
      thisao2xxs=ao2xxs+bf1*ngrids;
      thisao2yys=ao2yys+bf1*ngrids;
      thisao2zzs=ao2zzs+bf1*ngrids;
      thisao2xys=ao2xys+bf1*ngrids;
      thisao2xzs=ao2xzs+bf1*ngrids;
      thisao2yzs=ao2yzs+bf1*ngrids;
     }
     for (short int f2=0;f2!=n2;f2++){
      const short int bf2=bf2_first+f2;
      if (bf2>bf1) break;
      double * thataos=aos+bf2*ngrids;
      double * thatao1xs=ao1xs+bf2*ngrids;
      double * thatao1ys=ao1ys+bf2*ngrids;
      double * thatao1zs=ao1zs+bf2*ngrids;
      double * thatao2xxs,*thatao2yys,*thatao2zzs,*thatao2xys,*thatao2xzs,*thatao2yzs;
      thatao2xxs=thatao2yys=thatao2zzs=thatao2xys=thatao2xzs=thatao2yzs=nullptr;
      if (ao2xxs){
       thatao2xxs=ao2xxs+bf2*ngrids;
       thatao2yys=ao2yys+bf2*ngrids;
       thatao2zzs=ao2zzs+bf2*ngrids;
       thatao2xys=ao2xys+bf2*ngrids;
       thatao2xzs=ao2xzs+bf2*ngrids;
       thatao2yzs=ao2yzs+bf2*ngrids;
      }
      double tmp=0;
      for (long int kgrid=0;kgrid<ngrids;kgrid++){
       tmp=ws[kgrid]*vrxcs[kgrid]*D(bf1,bf2);
       Gxc(atom1,0)+=tmp*thisao1xs[kgrid]*thataos[kgrid];
       Gxc(atom1,1)+=tmp*thisao1ys[kgrid]*thataos[kgrid];
       Gxc(atom1,2)+=tmp*thisao1zs[kgrid]*thataos[kgrid];
       if (bf2!=bf1){
        Gxc(atom2,0)+=tmp*thatao1xs[kgrid]*thisaos[kgrid];
        Gxc(atom2,1)+=tmp*thatao1ys[kgrid]*thisaos[kgrid];
        Gxc(atom2,2)+=tmp*thatao1zs[kgrid]*thisaos[kgrid];
       }
       if (ao2xxs){
        tmp=2*ws[kgrid]*vsxcs[kgrid]*D(bf1,bf2);
        Gxc(atom1,0)+=tmp*(d1xs[kgrid]*(thisao2xxs[kgrid]*thataos[kgrid]+thatao1xs[kgrid]*thisao1xs[kgrid])
                          +d1ys[kgrid]*(thisao2xys[kgrid]*thataos[kgrid]+thatao1ys[kgrid]*thisao1xs[kgrid])
                          +d1zs[kgrid]*(thisao2xzs[kgrid]*thataos[kgrid]+thatao1zs[kgrid]*thisao1xs[kgrid]));
        Gxc(atom1,1)+=tmp*(d1xs[kgrid]*(thisao2xys[kgrid]*thataos[kgrid]+thatao1xs[kgrid]*thisao1ys[kgrid])
                          +d1ys[kgrid]*(thisao2yys[kgrid]*thataos[kgrid]+thatao1ys[kgrid]*thisao1ys[kgrid])
                          +d1zs[kgrid]*(thisao2yzs[kgrid]*thataos[kgrid]+thatao1zs[kgrid]*thisao1ys[kgrid]));
        Gxc(atom1,2)+=tmp*(d1xs[kgrid]*(thisao2xzs[kgrid]*thataos[kgrid]+thatao1xs[kgrid]*thisao1zs[kgrid])
                          +d1ys[kgrid]*(thisao2yzs[kgrid]*thataos[kgrid]+thatao1ys[kgrid]*thisao1zs[kgrid])
                          +d1zs[kgrid]*(thisao2zzs[kgrid]*thataos[kgrid]+thatao1zs[kgrid]*thisao1zs[kgrid]));
        if (bf2!=bf1){
         Gxc(atom2,0)+=tmp*(d1xs[kgrid]*(thatao2xxs[kgrid]*thisaos[kgrid]+thisao1xs[kgrid]*thatao1xs[kgrid])
                           +d1ys[kgrid]*(thatao2xys[kgrid]*thisaos[kgrid]+thisao1ys[kgrid]*thatao1xs[kgrid])
                           +d1zs[kgrid]*(thatao2xzs[kgrid]*thisaos[kgrid]+thisao1zs[kgrid]*thatao1xs[kgrid]));
         Gxc(atom2,1)+=tmp*(d1xs[kgrid]*(thatao2xys[kgrid]*thisaos[kgrid]+thisao1xs[kgrid]*thatao1ys[kgrid])
                           +d1ys[kgrid]*(thatao2yys[kgrid]*thisaos[kgrid]+thisao1ys[kgrid]*thatao1ys[kgrid])
                           +d1zs[kgrid]*(thatao2yzs[kgrid]*thisaos[kgrid]+thisao1zs[kgrid]*thatao1ys[kgrid]));
         Gxc(atom2,2)+=tmp*(d1xs[kgrid]*(thatao2xzs[kgrid]*thisaos[kgrid]+thisao1xs[kgrid]*thatao1zs[kgrid])
                           +d1ys[kgrid]*(thatao2yzs[kgrid]*thisaos[kgrid]+thisao1ys[kgrid]*thatao1zs[kgrid])
                           +d1zs[kgrid]*(thatao2zzs[kgrid]*thisaos[kgrid]+thisao1zs[kgrid]*thatao1zs[kgrid]));
        }
       }
      }
     }
    }
   }
  }
 }
 libint2::finalize();
 __Finalize_KS__
 Ghf/=4;
 Gxc*=-4;
 for (int iatom=0;iatom<natoms;iatom++){
  Ghf(iatom,0)+=hcoregrads[iatom*3+0].cwiseProduct(D).sum()-ovlgrads[iatom*3+0].cwiseProduct(W).sum();
  Ghf(iatom,1)+=hcoregrads[iatom*3+1].cwiseProduct(D).sum()-ovlgrads[iatom*3+1].cwiseProduct(W).sum();
  Ghf(iatom,2)+=hcoregrads[iatom*3+2].cwiseProduct(D).sum()-ovlgrads[iatom*3+2].cwiseProduct(W).sum();
 }
 Ghf*=2;
 time_t end=time(0);
 if (output) std::cout<<"done "<<end-start<<" s"<<std::endl;
 return Ghf+Gxc;
}

EigenMatrix RKSG(
		const int natoms,double * atoms,const char * basisset,
		EigenMatrix * ovlgrads,EigenMatrix * hcoregrads,EigenMatrix * fskeletons,
		int dfxid,int dfcid,int ngrids,double * ws,
		double * aos,
		double * ao1xs,double * ao1ys,double * ao1zs,
		double * ao2xxs,double * ao2yys,double * ao2zzs,
		double * ao2xys,double * ao2xzs,double * ao2yzs,
		EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupancies,
		const int nprocs,const bool output){
	if (fskeletons)
		return RKSG_concise(
				natoms,atoms,basisset,
				ovlgrads,hcoregrads,fskeletons,
				dfxid,dfcid,ngrids,ws,
   				aos,
				ao1xs,ao1ys,ao1zs,
				ao2xxs,ao2yys,ao2zzs,
				ao2xys,ao2xzs,ao2yzs,
				coefficients,orbitalenergies,occupancies,
				nprocs,output);
	else
		return RKSG_fast(natoms,atoms,basisset,
				ovlgrads,hcoregrads,
				dfxid,dfcid,ngrids,ws,
				aos,
				ao1xs,ao1ys,ao1zs,
				ao2xxs,ao2yys,ao2zzs,
				ao2xys,ao2xzs,ao2yzs,
				coefficients,orbitalenergies,occupancies,
				nprocs,output);
}

EigenMatrix RHFG(
		const int natoms,double * atoms,const char * basisset,
		EigenMatrix * ovlgrads,EigenMatrix * hcoregrads,EigenMatrix * fskeletons,
		EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occupancies,
		const int nprocs,const bool output){
 return RKSG(natoms,atoms,basisset,
             ovlgrads,hcoregrads,fskeletons,
             0,0,0,nullptr,
             nullptr,
             nullptr,nullptr,nullptr,
             nullptr,nullptr,nullptr,
             nullptr,nullptr,nullptr,
             coefficients,orbitalenergies,occupancies,
             nprocs,output);
}
