#include <libint2.hpp>
#include <Eigen/Dense> // Eigen::Matrix.
#include <vector> // Atom vectors.
#include <math.h> // Square root in integral screening.
#include <algorithm> // Sorting shell pair list.
#include <iostream>
#include <time.h>

typedef Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;
typedef Eigen::Matrix<double,3,1> EigenVector;

int nBasis(libint2::BasisSet obs){ // Size of basis set.
	int n=0;
	for (const auto& shell:obs){
		n=n+shell.size();
	}
	return n;
}

EigenMatrix Overlap(libint2::BasisSet obs){ // Calculating overlap matrix.
	std::cout<<" ... Calculating overlap integrals ..."<<std::endl;
	clock_t start,end;
	start=clock();
	const int nbasis=nBasis(obs); // Size of basis set.
	EigenMatrix overlapmatrix(nbasis,nbasis);
	libint2::Engine s_engine(libint2::Operator::overlap,obs.max_nprim(),obs.max_l());
	const auto& buf_vec=s_engine.results(); // Preallocating memory for overlap integrals, also indicating the size of overlap matrix.
	auto shell2bf=obs.shell2bf(); // Feeding the index of a shell to 'shell2bf' returns the index of the first basis function of that shell among all basis functions.
	for (short int s1=0;s1!=obs.size();s1++){ // Looping over all shells.
		short int bf1=shell2bf[s1];
		short int n1=obs[s1].size(); // # of basis functions of a shell.
		for (short int s2=s1;s2!=obs.size();s2++){ // Looping over all shells, avoiding replication.
			s_engine.compute(obs[s1],obs[s2]); // Calculating overlap integrals of basis functions in those two shells.
			auto ints_shellset=buf_vec[0]; // Overlap integrals of this shell pair are stored in memory location buf_vec.
			if (ints_shellset==nullptr){ // If there are no integrals at all, continue. Will this ever happen?
				continue;
			}
			short int bf2=shell2bf[s2];
			short int n2=obs[s2].size();
			for (short int f1=0;f1!=n1;f1++){ // Looping over all basis functions in shell 1.
				for (short int f2=0;f2!=n2;f2++){ // Looping over all basis functions in shell 2. No need to worry about replication, since the two shells are different.
					double thisvalue=ints_shellset[f1*n2+f2]; // Integral of basis functions f1 and f2 is stored in the (f1*n2+f2)^th memory location.
					overlapmatrix(bf1+f1,bf2+f2)=thisvalue;
					overlapmatrix(bf2+f2,bf1+f1)=thisvalue; // Overlap matrix is symmetric.
				}
			}
		}
	}
	end=clock();
	std::cout<<" Overlap integrals elapsed time = "<<double(end-start)/CLOCKS_PER_SEC<<" s"<<std::endl;
	std::cout<<" ... Overlap integrals done ..."<<std::endl;
	return overlapmatrix;
}

EigenMatrix Kinetic(libint2::BasisSet obs){ // Similar to overlap matrix.
	std::cout<<" ... Calculating kinetic integrals ..."<<std::endl;
	clock_t start,end;
	start=clock();
	const int nbasis=nBasis(obs);
	EigenMatrix kineticmatrix(nbasis,nbasis);
	libint2::Engine k_engine(libint2::Operator::kinetic,obs.max_nprim(),obs.max_l());
	const auto& buf_vec=k_engine.results();
	auto shell2bf=obs.shell2bf();
	for (short int s1=0;s1!=obs.size();s1++){
		short int bf1=shell2bf[s1];
		short int n1=obs[s1].size();
		for (short int s2=s1;s2!=obs.size();s2++){
			k_engine.compute(obs[s1],obs[s2]);
			auto ints_shellset=buf_vec[0];
			if (ints_shellset==nullptr){
				continue;
			}
			short int bf2=shell2bf[s2];
			short int n2=obs[s2].size();
			for (short int f1=0;f1!=n1;f1++){
				for (short int f2=0;f2!=n2;f2++){
					double thisvalue=ints_shellset[f1*n2+f2];
					kineticmatrix(bf1+f1,bf2+f2)=thisvalue;
					kineticmatrix(bf2+f2,bf1+f1)=thisvalue;
				}
			}
		}
	}
	end=clock();
	std::cout<<" Kinetic integrals elapsed time = "<<double(end-start)/CLOCKS_PER_SEC<<" s"<<std::endl;
	std::cout<<" ... Kinetic integrals done ..."<<std::endl;
	return kineticmatrix;
}

EigenMatrix Nuclear(libint2::BasisSet obs,std::vector<libint2::Atom> atoms){ // Similar to overlap matrix and kinetic matrix.
	std::cout<<" ... Calculating nuclear integrals ..."<<std::endl;
	clock_t start,end;
	start=clock();
	const int nbasis=nBasis(obs);
	EigenMatrix nuclearmatrix(nbasis,nbasis);
	libint2::Engine v_engine(libint2::Operator::nuclear,obs.max_nprim(),obs.max_l());
	v_engine.set_params(make_point_charges(atoms)); // Including nuclear information.
	const auto& buf_vec=v_engine.results();
	auto shell2bf=obs.shell2bf();
	for (short s1=0;s1!=obs.size();s1++){
		short int bf1=shell2bf[s1];
		short int n1=obs[s1].size();
		for (short int s2=s1;s2!=obs.size();s2++){
			v_engine.compute(obs[s1],obs[s2]);
			auto ints_shellset=buf_vec[0];
			if (ints_shellset==nullptr){
				continue;
			}
			short int bf2=shell2bf[s2];
			short int n2=obs[s2].size();
			for (short int f1=0;f1!=n1;f1++){
				for (short int f2=0;f2!=n2;f2++){
					double thisvalue=ints_shellset[f1*n2+f2];
					nuclearmatrix(bf1+f1,bf2+f2)=thisvalue;
					nuclearmatrix(bf2+f2,bf1+f1)=thisvalue;
				}
			}
		}
	}
	end=clock();
	std::cout<<" Nuclear integrals elapsed time = "<<double(end-start)/CLOCKS_PER_SEC<<" s"<<std::endl;
	std::cout<<" ... Nuclear integrals done ..."<<std::endl;
	return nuclearmatrix;
}

void Repulsion(libint2::BasisSet obs,double **repulsion,short int **indices,int &nintegrals){ // Calculating atomic two-electron integrals. Integral values will be saved to **repulsion and non-negligible integral indices will be saved to **indices.
	std::cout<<" ... Calculating repulsion integrals ..."<<std::endl;
	clock_t start,end;
	start=clock();
	const int nbasis=nBasis(obs);
	EigenMatrix repulsiontensordiag(nbasis,nbasis); // Calculating g_abab to find out two-electron integrals of which shell quartets to set to zero in integral screening.
	libint2::Engine g_engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l());
	const auto& buf_vec=g_engine.results();
	auto shell2bf=obs.shell2bf();
	for (short int s1=0;s1!=obs.size();s1++){ // Similar to one-electron integrals.
		short int bf1=shell2bf[s1];
		short int n1=obs[s1].size();
		for (short int s2=s1;s2!=obs.size();s2++){
			g_engine.compute(obs[s1],obs[s2],obs[s1],obs[s2]);
			auto ints_shellset=buf_vec[0];
			if (ints_shellset==nullptr){ // Once Libint2 detects the integrals in the shell quartet are small, it will save nothing to the memory seat. In that case, continue to next iteration. This is a little bit different that the integral screening code I wrote below. Anyway, this part must exist for the program to run normally.
				continue;
			}
			short int bf2=shell2bf[s2];
			short int n2=obs[s2].size();
			for (short int f1=0;f1!=n1;f1++){
				for (short int f2=0;f2!=n2;f2++){
					double thisvalue=ints_shellset[f1*n2+f2];
					repulsiontensordiag(bf1+f1,bf2+f2)=thisvalue;
					repulsiontensordiag(bf2+f2,bf1+f1)=thisvalue;
				}
			}
		}
	}
	nintegrals=0; // Number of integrals not discarded.
	int nshellquartets=0; // Number of shell quartets not discarded.
	for (short int s1=0;s1<obs.size();s1++){ // In this loop, only the numbers of non-negligible shell quartets and integrals are determined.
		short int bf1_first=shell2bf[s1];
		short int n1=obs[s1].size();
		for (short int s2=0;s2<=s1;s2++){
			short int bf2_first=shell2bf[s2];
			short int n2=obs[s2].size();
			for (short int s3=0;s3<=s1;s3++){
				short int bf3_first=shell2bf[s3];
				short int n3=obs[s3].size();
				for (short int s4=0;s4<=((s1==s3)?s2:s3);s4++){
					short int bf4_first=shell2bf[s4];
					short int n4=obs[s4].size();
					bool discard=true;
					for (short int f1=0;f1!=n1 && discard==true;f1++){
						const short int bf1=f1+bf1_first;
						for (short int f2=0;f2!=n2 && discard==true;f2++){
							const short int bf2=f2+bf2_first;
							for (short int f3=0;f3!=n3 && discard==true;f3++){
								const short int bf3=f3+bf3_first;
								for (short int f4=0;f4!=n4 && discard==true;f4++){
									const short int bf4=f4+bf4_first;
									double integral1=repulsiontensordiag(bf1,bf2);
									double integral2=repulsiontensordiag(bf3,bf4);
									double upperbound=sqrt(integral1*integral2); // According to Cauchy-Schwarz inequality, sqrt(integral1*integral2) is the upper bound of (bf1,bf2|bf3,bf4). If the upper bound of any basis function quartet of (bf1,bf2,bf3,bf4) in the shell quartet (s1,s2,s3,s4) is larger than 10^-10, the shell quartet will not be discarded in the following two-electron integral evaluation.
									if (upperbound>1.0e-10){
										discard=false;
										nshellquartets=nshellquartets+1; // This shell quartet will not be discarded.
										nintegrals=nintegrals+n1*n2*n3*n4; // All n1*n2*n3*n4 integrals in this shell quartet will not be discarded.
									}
								}
							}
						}
					}
				}
			}
		}
	}
	double *repulsionhead=new double[nintegrals]; // Head denotes the memory seat of the first repulsion integral. The memory it is in charge of is nintegrals in length.
	double *repulsionranger=repulsionhead; // Ranger will be used to iterate over and write all the repulsion integrals' seats.
	short int *indiceshead=new short int[nintegrals*5]; // The indices and degrees of degeneracy of the integrals, denoted as [bf1_1,bf2_1,bf3_1,bf4_1,deg_1,bf1_2,bf2_2,bf3_2,bf4_2,deg_2,...]. The memory it is in charge of is nintegrals*5 in length.
	short int *integralindicesranger=indiceshead;
	short int *shellindiceshead=new short int[nshellquartets*4]; // The indices of the shell quartets, denoted as [s1_1,s2_1,s3_1,s4_1,s1_2,s2_2,s3_2,s4_2,...]. The memory it is in charge of is nshellquartets*4 in length.
	short int *shellindicesranger=shellindiceshead;
	for (short int s1=0;s1<obs.size();s1++){ // In this loop, the indices of non-negligible shell quartets are collected.
		short int bf1_first=shell2bf[s1];
		short int n1=obs[s1].size();
		for (short int s2=0;s2<=s1;s2++){
			short int bf2_first=shell2bf[s2];
			short int n2=obs[s2].size();
			for (short int s3=0;s3<=s1;s3++){
				short int bf3_first=shell2bf[s3];
				short int n3=obs[s3].size();
				for (short int s4=0;s4<=((s1==s3)?s2:s3);s4++){
					short int bf4_first=shell2bf[s4];
					short int n4=obs[s4].size();
					bool discard=true;
					for (short int f1=0;f1!=n1 && discard==true;f1++){
						const short int bf1=f1+bf1_first; // All indices are saved as short int, which takes only half of int's memory.
						for (short int f2=0;f2!=n2 && discard==true;f2++){
							const short int bf2=f2+bf2_first;
							for (short int f3=0;f3!=n3 && discard==true;f3++){
								const short int bf3=f3+bf3_first;
								for (short int f4=0;f4!=n4 && discard==true;f4++){
									const short int bf4=f4+bf4_first;
									double integral1=repulsiontensordiag(bf1,bf2);
									double integral2=repulsiontensordiag(bf3,bf4);
									double upperbound=sqrt(integral1*integral2); // According to Cauchy-Schwarz inequality, sqrt(integral1*integral2) is the upper bound of (bf1,bf2|bf3,bf4). If the upper bound of any basis function quartet of (bf1,bf2,bf3,bf4) in the shell quartet (s1,s2,s3,s4) is larger than 10^-10, the shell quartet will not be discarded in the following two-electron integral evaluation.
									if (upperbound>1.0e-10){
										discard=false;
										*shellindicesranger=s1;shellindicesranger++; // Collecting undiscarded shell quartet indices.
										*shellindicesranger=s2;shellindicesranger++; // Indices are saved in this way: [s1_1,s2_1,s3_1,s4_1,deg_1,s1_2,s2_2,s3_2,s4_2,deg_2,s1_3,s2_3,s3_3,s4_3,deg_3, ... ...].
										*shellindicesranger=s3;shellindicesranger++;
										*shellindicesranger=s4;shellindicesranger++;
									}
								}
							}
						}
					}
				}
			}
		}
	}
	shellindicesranger=shellindiceshead; // Reinitiating ranger pointer to the first memory seat.
	for (int ishellquartet=0;ishellquartet<nshellquartets;ishellquartet++){ // In this loop, all non-negligible shell quartets are iterated over, and the indices, degree of degenercy and value of each integral are saved.
		short int s1=*shellindicesranger;shellindicesranger++;
		short int s2=*shellindicesranger;shellindicesranger++;
		short int s3=*shellindicesranger;shellindicesranger++;
		short int s4=*shellindicesranger;shellindicesranger++;
                short int s1s2_deg=(s1==s2)?1:2;
                short int s3s4_deg=(s3==s4)?1:2;
                short int s1s2_s3s4_deg=(s1==s3)?(s2==s4?1:2):2;
                short int s1s2s3s4_deg=s1s2_deg*s3s4_deg*s1s2_s3s4_deg;
		short int bf1_first=shell2bf[s1];short int n1=obs[s1].size();
		short int bf2_first=shell2bf[s2];short int n2=obs[s2].size();
		short int bf3_first=shell2bf[s3];short int n3=obs[s3].size();
		short int bf4_first=shell2bf[s4];short int n4=obs[s4].size();
		g_engine.compute(obs[s1],obs[s2],obs[s3],obs[s4]);
		const auto *buf_1234=buf_vec[0];
		if (buf_1234==nullptr){
			continue;
		}
		for (short int f1=0,f1234=0;f1!=n1;f1++){
			const short int bf1=f1+bf1_first;
			for (short int f2=0;f2!=n2;f2++){
				const short int bf2=f2+bf2_first;
				for (short int f3=0;f3!=n3;f3++){
					const short int bf3=f3+bf3_first;
					for (short int f4=0;f4!=n4;f4++,++f1234){
						const short int bf4=f4+bf4_first;
						const double value=buf_1234[f1234];
						*integralindicesranger=bf1;integralindicesranger++; // Indices and degrees of degeneracy are saved in this way: [bf1_1,bf2_1,bf3_1,bf4_1,deg_1,bf1_2,bf2_2,bf3_2,bf4_2,deg_2,bf1_3,bf2_3,bf3_3,bf4_3,deg_3, ... ...].
						*integralindicesranger=bf2;integralindicesranger++;
						*integralindicesranger=bf3;integralindicesranger++;
						*integralindicesranger=bf4;integralindicesranger++;
						*integralindicesranger=s1s2s3s4_deg;integralindicesranger++;
						*repulsionranger=value;repulsionranger++;
					}
				}
			}
		}
	}
	shellindicesranger=NULL; // Killing the donkeys when they finish milling.
	shellindiceshead=NULL;
	integralindicesranger=NULL;
	repulsionranger=NULL;
	*repulsion=repulsionhead;
	*indices=indiceshead;
	end=clock();
	std::cout<<" Repulsion integrals elapsed time = "<<double(end-start)/CLOCKS_PER_SEC<<" s"<<std::endl;
	std::cout<<" ... Repulsion integrals done ..."<<std::endl;
}

int main(int argc,char *argv[]){
	libint2::initialize();
	std::ifstream input(argv[1]);
	std::vector<libint2::Atom> atoms=libint2::read_dotxyz(input);
	libint2::BasisSet obs(argv[2],atoms);

	EigenMatrix overlap=Overlap(obs);
	EigenMatrix kinetic=Kinetic(obs);
	EigenMatrix nuclear=Nuclear(obs,atoms);
	double *repulsion;
	short int *indices;
	int nintegrals;
	Repulsion(obs,&repulsion,&indices,nintegrals);
	libint2::finalize();
	return 0;
}

/*
EigenTensor3  Densityfitting3(libint2::BasisSet obs,libint2::BasisSet dfobs){ // Repulsion integrals between basis functions and auiliary functions.
	const int nauxiliary=nBasis(dfobs); // Size of auxiliary basis set.
	const int nbasis=nBasis(obs);
	EigenTensor3 eri3tensor(nauxiliary,nbasis,nbasis);
	libint2::Engine eri3_engine(libint2::Operator::coulomb,std::max(obs.max_nprim(),dfobs.max_nprim()),std::max(obs.max_l(),dfobs.max_l()),0,std::numeric_limits<double>::epsilon(),libint2::operator_traits<libint2::Operator::coulomb>::default_params(),libint2::BraKet::xs_xx);
	const auto& buf_vec=eri3_engine.results();
	auto shell2bf=obs.shell2bf();
	for (int s0=0;s0!=dfobs.size();s0++){
		int af0=shell2bf[s0];
		int n0=obs[s0].size();
		for (int s1=0;s1!=obs.size();s1++){
			int bf1=shell2bf[s1];
			int n1=obs[s1].size();
			for (int s2=s1;s2!=obs.size();s2++){
				eri3_engine.compute(dfobs[s0],obs[s1],obs[s2]);
				auto ints_shellset=buf_vec[0];
				if (ints_shellset==nullptr){
					continue;
				}
				int bf2=shell2bf[s2];
				int n2=obs[s2].size();
				for (int f0=0,f012=0;f0!=n0;f0++){
					for (int f1=0;f1!=n1;f1++){
						for (int f2=0;f2!=n2;f2++,f012++){
							double thisvalue=ints_shellset[f012];
							eri3tensor(af0+f0,bf1+f1,bf2+f2)=thisvalue;
							eri3tensor(af0+f0,bf2+f2,bf1+f1)=thisvalue;
						}
					}
				}
			}
		}
	}
	return eri3tensor;
}
*/

