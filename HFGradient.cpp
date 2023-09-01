#include <Eigen/Dense>
#include <libint2.hpp>
#include <cmath>
#include <ctime>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include "Aliases.h"
#include "Libint2.h"

#define __Loop_Over_XYZ_2__(iatom,position){\
 for (short int f1=0,f1234=0;f1!=n1;f1++){\
  const short int bf1=bf1_first+f1;\
  for (short int f2=0;f2!=n2;f2++){\
   const short int bf2=bf2_first+f2;\
   const double ab_deg=(bf1==bf2)?1:2;\
   for (short int f3=0;f3!=n3;f3++){\
    const short int bf3=bf3_first+f3;\
    for (short int f4=0;f4!=n4;f4++,f1234++){\
     const short int bf4=bf4_first+f4;\
     if (bf2<=bf1 && bf3<=bf1 && bf4<=((bf1==bf3)?bf2:bf3)){\
      const double cd_deg=(bf3==bf4)?1:2;\
      const double ab_cd_deg=(bf1==bf3)?(bf2==bf4?1:2):2;\
      const double abcd_deg=ab_deg*cd_deg*ab_cd_deg;\
      G((iatom),0)+=abcd_deg*(4*D(bf1,bf2)*D(bf3,bf4)-D(bf1,bf3)*D(bf2,bf4)-D(bf1,bf4)*D(bf2,bf3))*buf_vec[(position)*3+0][f1234];\
      G((iatom),1)+=abcd_deg*(4*D(bf1,bf2)*D(bf3,bf4)-D(bf1,bf3)*D(bf2,bf4)-D(bf1,bf4)*D(bf2,bf3))*buf_vec[(position)*3+1][f1234];\
      G((iatom),2)+=abcd_deg*(4*D(bf1,bf2)*D(bf3,bf4)-D(bf1,bf3)*D(bf2,bf4)-D(bf1,bf4)*D(bf2,bf3))*buf_vec[(position)*3+2][f1234];\
     }\
    }\
   }\
  }\
 }\
}

EigenMatrix RHFG(const int natoms,double * atoms,const char * basisset,
                 EigenMatrix * ovlgrads,EigenMatrix * hcoregrads,
                 EigenMatrix coefficients,EigenVector orbitalenergies,EigenVector occs,
                const int nprocs,const bool output){
	if (output) std::cout<<"Calculating repulsion integral derivatives w.r.t nuclear coordinates and summing up RHF gradient ... ";
	time_t start=time(0);
	__Basis_From_Atoms__
	const int nbasis=nBasis_from_obs(obs);
	EigenMatrix D=EigenZero(nbasis,nbasis);
	EigenMatrix W=EigenZero(nbasis,nbasis);
	for (int a=0;a<nbasis;a++)
 		for (int b=0;b<nbasis;b++)
			for (int i=0;i<nbasis;i++){
				D(a,b)+=occs(i)*coefficients(a,i)*coefficients(b,i);
				W(a,b)+=occs(i)*coefficients(a,i)*coefficients(b,i)*orbitalenergies(i);
			}
	EigenMatrix G=EigenZero(natoms,3);
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
					__Loop_Over_XYZ_2__(atom1,0)
					__Loop_Over_XYZ_2__(atom2,1)
					__Loop_Over_XYZ_2__(atom3,2)
					__Loop_Over_XYZ_2__(atom4,3)
				}
			}
		}
	}
	libint2::finalize();
	G/=4;
	for (int iatom=0;iatom<natoms;iatom++){
		G(iatom,0)+=(hcoregrads[iatom*3+0].cwiseProduct(D).sum()-ovlgrads[iatom*3+0].cwiseProduct(W).sum());
		G(iatom,1)+=(hcoregrads[iatom*3+1].cwiseProduct(D).sum()-ovlgrads[iatom*3+1].cwiseProduct(W).sum());
		G(iatom,2)+=(hcoregrads[iatom*3+2].cwiseProduct(D).sum()-ovlgrads[iatom*3+2].cwiseProduct(W).sum());
	}
	G*=2;
	time_t end=time(0);
	if (output) std::cout<<"done "<<end-start<<" s"<<std::endl;
	return G;
}


