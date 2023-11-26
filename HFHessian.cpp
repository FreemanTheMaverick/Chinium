#include <Eigen/Dense>
#include <libint2.hpp>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include "Aliases.h"

EigenMatrix RKSH(
		const int natoms,double * atoms,const char * basisset,
		EigenMatrix d,EigenMatrix * Dxns,
		EigenMatrix w,EigenMatrix * Wxns,
		EigenMatrix * ovlgrads,EigenMatrix * fskeletons,
		double kscale,
		const int nprocs,const bool output){
	if (output) std::printf("Analytical hessian ...\n");
	auto big_start=std::chrono::system_clock::now();
	__Basis_From_Atoms__
	__nBasis_From_OBS__
	EigenMatrix tmp=EigenZero(3*natoms,3*natoms);

	if (output) std::printf("| Calculating first-order one- and two-electron integral derivative contributions ... ");
	EigenMatrix hessian1=EigenZero(3*natoms,3*natoms);
	auto small_start=std::chrono::system_clock::now();
	for (int x=0;x<3*natoms;x++)
		for (int y=0;y<3*natoms;y++)
			hessian1(x,y)+=2*(Dxns[y]*fskeletons[x]).trace()-(Wxns[y]*ovlgrads[x]).trace();
	std::chrono::duration<double> duration=std::chrono::system_clock::now()-small_start;
	if (output) std::printf("%f s\n",duration.count());
	tmp=0.5*(hessian1+hessian1.transpose());
	hessian1=tmp;
	std::cout<<hessian1<<std::endl;

	if (output) std::printf("| Calculating second-order one-electron integral derivative contributions ... ");
	small_start=std::chrono::system_clock::now();
	EigenMatrix hessian21=EigenZero(3*natoms,3*natoms);
	int atomlist[]={114,514,1919,810};
	libint2::initialize();
	libint2::Engine ovl_engine(libint2::Operator::overlap,obs.max_nprim(),obs.max_l(),2);
	libint2::Engine knt_engine(libint2::Operator::kinetic,obs.max_nprim(),obs.max_l(),2);
	libint2::Engine ncl_engine(libint2::Operator::nuclear,obs.max_nprim(),obs.max_l(),2);
	ncl_engine.set_params(libint2::make_point_charges(libint2atoms));
	const auto & ovl_buf=ovl_engine.results();
	const auto & knt_buf=knt_engine.results();
	const auto & ncl_buf=ncl_engine.results();
	auto shell2bf=obs.shell2bf();
	auto shell2atom=obs.shell2atom(libint2atoms);
	for (short int s1=0;s1!=(short int)obs.size();s1++){
	 atomlist[0]=shell2atom[s1];
	 const short int bf1_first=shell2bf[s1];
	 const short int n1=obs[s1].size();
	 for (short int s2=0;s2<=s1;s2++){
	  atomlist[1]=shell2atom[s2];
	  const short int bf2_first=shell2bf[s2];
	  const short int n2=obs[s2].size();

	  // Overlap
	  ovl_engine.compute(obs[s1],obs[s2]);
	  //if (ovl_buf[0]){
	  if (ovl_buf[0] && atomlist[0]!=atomlist[1]){
	   for (short int f1=0,f12=0;f1!=n1;f1++){
	    const short int bf1=bf1_first+f1;
	    for (short int f2=0;f2!=n2;f2++,f12++){
	     const short int bf2=bf2_first+f2;
	     if (bf2<=bf1){
	      const double tmp=((bf1==bf2)?1:2)*w(bf1,bf2);
	      int xpert=114514;
	      int ypert=1919810;
	      for (int p=0,ptqs=0;p<2;p++) for (int t=0;t<3;t++){
	       xpert=3*atomlist[p]+t;
	       for (int q=p;q<2;q++) for (int s=((q==p)?t:0);s<3;s++,ptqs++){
	        ypert=3*atomlist[q]+s;
	        hessian21(xpert,ypert)-=tmp*ovl_buf[ptqs][f12];
	       }
	      }
	     }
	    }
	   }
	  }

	  // Kinetic
	  knt_engine.compute(obs[s1],obs[s2]);
	  if (knt_buf[0] && atomlist[0]!=atomlist[1]){
	   for (short int f1=0,f12=0;f1!=n1;f1++){
	    const short int bf1=bf1_first+f1;
	    for (short int f2=0;f2!=n2;f2++,f12++){
	     const short int bf2=bf2_first+f2;
	     if (bf2<=bf1){
	      const double tmp=((bf1==bf2)?1:2)*d(bf1,bf2);
	      int xpert=114514;
	      int ypert=1919810;
	      for (int p=0,ptqs=0;p<2;p++) for (int t=0;t<3;t++){
	       xpert=3*atomlist[p]+t;
	       for (int q=p;q<2;q++) for (int s=((q==p)?t:0);s<3;s++,ptqs++){
	        ypert=3*atomlist[q]+s;
	        hessian21(xpert,ypert)+=0*tmp*knt_buf[ptqs][f12];
	       }
	      }
	     }
	    }
	   }
	  }

	  // Nuclear
	  ncl_engine.compute(obs[s1],obs[s2]);
	  if (ncl_buf[0]){
	   for (short int f1=0,f12=0;f1!=n1;f1++){
	    const short int bf1=bf1_first+f1;
	    for (short int f2=0;f2!=n2;f2++,f12++){
	     const short int bf2=bf2_first+f2;
	     if (bf2<=bf1){
	      const double tmp=((bf1==bf2)?1:2)*d(bf1,bf2);
	      int xpert=114514;
	      int ypert=1919810;
	      for (int p=0,ptqs=0;p<2+natoms;p++) for (int t=0;t<3;t++){
	       xpert=3*(p<2?atomlist[p]:p-2)+t;
	       for (int q=p;q<2+natoms;q++) for (int s=((q==p)?t:0);s<3;s++,ptqs++){
	        ypert=3*(q<2?atomlist[q]:q-2)+s;
	        hessian21(xpert,ypert)+=0*tmp*ncl_buf[ptqs][f12];
	       }
	      }
	     }
	    }
	   }
	  }
	 }
	}
	libint2::finalize();
	tmp=hessian21+hessian21.transpose();
	EigenMatrix diagonal(tmp.diagonal().asDiagonal());
	hessian21=tmp-0.5*diagonal;
	hessian21*=2;
	duration=std::chrono::system_clock::now()-small_start;
	if (output) std::printf("%f s\n",duration.count());
	std::cout<<hessian21<<std::endl;
 
	if (output) std::printf("| Calculating second-order two-electron integral derivative contributions ... ");
	small_start=std::chrono::system_clock::now();
	EigenMatrix hessianj=EigenZero(3*natoms,3*natoms);
	EigenMatrix hessiank=EigenZero(3*natoms,3*natoms);
	libint2::initialize();
	libint2::Engine engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l(),2);
	const auto & buf_vec=engine.results();
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
	         int xpert=114514;
	         int ypert=1919810;
	         for (int p=0,ptqs=0;p<4;p++) for (int t=0;t<3;t++){
	          xpert=3*atomlist[p]+t;
	          for (int q=p;q<4;q++) for (int s=((q==p)?t:0);s<3;s++,ptqs++){
	           ypert=3*atomlist[q]+s;
		   if (atomlist[0]==atomlist[1] && atomlist[1]==atomlist[2] && atomlist[2]==atomlist[3]) continue;
	           tmp=abcd_deg*buf_vec[ptqs][f1234];
	           hessianj(xpert,ypert)+=d(bf1,bf2)*tmp*d(bf3,bf4);
	           hessianj(xpert,ypert)+=d(bf3,bf4)*tmp*d(bf1,bf2);
	           if (kscale>0){
	            hessiank(xpert,ypert)+=d(bf1,bf3)*tmp*d(bf2,bf4);
	            hessiank(xpert,ypert)+=d(bf2,bf4)*tmp*d(bf1,bf3);
	            hessiank(xpert,ypert)+=d(bf1,bf4)*tmp*d(bf2,bf3);
	            hessiank(xpert,ypert)+=d(bf2,bf3)*tmp*d(bf1,bf4);
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
	hessiank*=0.5*kscale;
	EigenMatrix hessiang2=hessianj-0.5*hessiank;
	tmp=hessiang2+hessiang2.transpose();
	EigenMatrix diagonalg2(tmp.diagonal().asDiagonal());
	hessiang2=tmp-0.5*diagonalg2;
	duration=std::chrono::system_clock::now()-small_start;
	if (output) std::printf("%f s\n",duration.count());
	std::cout<<hessiang2<<std::endl;

	duration=std::chrono::system_clock::now()-big_start;
	if (output) std::printf("| Done; %f s\n",duration.count());
	return hessian1+hessian21+hessiang2;

}










