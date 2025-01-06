#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector> // Atom vectors.
#include <ctime>
#include <iostream>
#include <omp.h>
#include "Aliases.h"

#define __integral_threshold__ -1.e-12

int nOneElectronIntegrals(const int natoms,double * atoms,const char * basisset,const bool output){ // Number of one-electron integrals.
	__Basis_From_Atoms__
	__nBasis_From_OBS__
	int n1integrals=(1+nbasis)*nbasis/2;
	if (output) std::cout<<"Number of one-electron integrals in total ... "<<n1integrals<<std::endl;
	return n1integrals;
}

EigenMatrix OneElectronIntegrals(const int natoms,double * atoms,const char * basisset,char type,const bool output){ // Computing various one-electron integrals.
	libint2::Operator operator_=libint2::Operator::overlap;
	if (type=='s'){
	        if (output) std::cout<<"Calculating overlap integrals ... ";
		operator_=libint2::Operator::overlap;
	}else if (type=='k'){
	        if (output) std::cout<<"Calculating kinetic integrals ... ";
		operator_=libint2::Operator::kinetic;
	}else if (type=='n'){
	        if (output) std::cout<<"Calculating nuclear integrals ... ";
		operator_=libint2::Operator::nuclear;
	}
	time_t start=time(0);
	__Basis_From_Atoms__
	__nBasis_From_OBS__
	EigenMatrix matrix(nbasis,nbasis);
	libint2::initialize();
	libint2::Engine engine(operator_,obs.max_nprim(),obs.max_l());
	if (type=='n'){
		engine.set_params(libint2::make_point_charges(libint2atoms)); // Including nuclear information.
        }
	const auto & buf_vec=engine.results(); // Preallocating memory for integrals.
	const auto shell2bf=obs.shell2bf(); // Feeding the index of a shell to 'shell2bf' returns the index of the first basis function of that shell among all basis functions.
	for (short int s1=0;s1!=(short int)obs.size();s1++){ // Looping over all shells.
		const short int bf1_first=shell2bf[s1];
		const short int n1=obs[s1].size(); // Number of basis functions in a shell.
		for (short int s2=0;s2<=s1;s2++){ // Looping over all shells, avoiding replication.
			engine.compute(obs[s1],obs[s2]); // Calculating integrals of basis functions in those two shells.
			const auto ints_shellset=buf_vec[0]; // Integrals of this shell pair are stored in memory location buf_vec.
			if (ints_shellset==nullptr){ // If there are no integrals at all, continue. Will this ever happen?
				continue;
			}
			const short int bf2_first=shell2bf[s2];
			const short int n2=obs[s2].size();
			for (short int f1=0;f1!=n1;f1++){ // Looping over all basis functions in shell 1.
				const short int bf1=bf1_first+f1; // Index of the basis function among all basis functions.
				for (short int f2=0;f2!=n2;f2++){ // Looping over all basis functions in shell 2.
					const short int bf2=bf2_first+f2;
					if (bf2<=bf1){ // Considering only unique pairs of basis functions.
						matrix(bf1,bf2)=ints_shellset[f1*n2+f2]; // One-electron integral matrix is symmetric.
						matrix(bf2,bf1)=ints_shellset[f1*n2+f2]; // One-electron integral matrix is symmetric.
					}
				}
			}
		}
	}
	libint2::finalize();
	time_t end=time(0);
	if (output) std::cout<<"done "<<end-start<<" s"<<std::endl;
	return matrix;
}

EigenMatrix Overlap(const int natoms,double * atoms,const char * basisset,const bool output){
	return OneElectronIntegrals(natoms,atoms,basisset,'s',output);
}

EigenMatrix Kinetic(const int natoms,double * atoms,const char * basisset,const bool output){
	return OneElectronIntegrals(natoms,atoms,basisset,'k',output);
}

EigenMatrix Nuclear(const int natoms,double * atoms,const char * basisset,const bool output){
	return OneElectronIntegrals(natoms,atoms,basisset,'n',output);
}

EigenMatrix RepulsionDiag(const int natoms,double * atoms,const char * basisset,const bool output){ // Computing the diagonal elements of electron repulsion tensor for Cauchy-Schwarz screening.
	if (output) std::cout<<"Calculating diagonal elements of repulsion integrals ... ";
	time_t start=time(0);
	__Basis_From_Atoms__
	__nBasis_From_OBS__
	EigenMatrix repulsiondiag(nbasis,nbasis);
	libint2::initialize();
	libint2::Engine engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l());
	const auto & buf_vec=engine.results();
	const auto shell2bf=obs.shell2bf();
	for (short int s1=0;s1<(short int)obs.size();s1++){
		const short int bf1_first=shell2bf[s1];
		const short int n1=obs[s1].size();
		for (short int s2=0;s2<=s1;s2++){
			const short int bf2_first=shell2bf[s2];
			const short int n2=obs[s2].size();
			engine.compute(obs[s1],obs[s2],obs[s1],obs[s2]); // Computing the integrals in the shell quartet (12|12).
			const auto * buf_1234=buf_vec[0];
			if (buf_1234==nullptr){
				continue;
			}
			for (short int f1=0,f1234=0;f1!=n1;f1++){ // Integrals are stored in buffer as four-dimensional tensor.
				const short int bf1=f1+bf1_first;
				for (short int f2=0;f2!=n2;f2++){
					const short int bf2=f2+bf2_first;
					for (short int f3=0;f3!=n1;f3++){
						const short int bf3=f3+bf1_first;
						for (short int f4=0;f4!=n2;f4++,f1234++){
							const short int bf4=f4+bf2_first;
							if (bf1==bf3 && bf2==bf4 && bf1>=bf2){ // Considering only the unique diagonal elements with (bf1==bf3 && bf2==bf4).
								repulsiondiag(bf1,bf2)=buf_1234[f1234]; // Repulsion diagonal integral matrix is symmetric.
								repulsiondiag(bf2,bf1)=buf_1234[f1234];
							}
						}
					}
				}
			}
		}
	}
	libint2::finalize();
	time_t end=time(0);
	if (output) std::cout<<"done "<<end-start<<" s"<<std::endl;
	return repulsiondiag;
}

long int nTwoElectronIntegrals(const int natoms,double * atoms,const char * basisset,EigenMatrix repulsiondiag,int & nshellquartets,const bool output){ // Numbers of two-electron integrals and nonequivalent shell quartets after Cauchy-Schwarz screening.
	__Basis_From_Atoms__
	__nBasis_From_OBS__
	long int n2integrals=nbasis*(nbasis+1)*(nbasis*(nbasis+1)/2+1)/4;
        if (output) std::cout<<"Number of two electron integrals in total ... "<<n2integrals<<std::endl;
	const auto shell2bf=obs.shell2bf();
	n2integrals=0; // Number of integrals not discarded.
	nshellquartets=0; // Number of shell quartets not discarded.
	for (short int s1=0;s1<(short int)obs.size();s1++){
		const short int bf1_first=shell2bf[s1];
		const short int n1=obs[s1].size();
		for (short int s2=0;s2<=s1;s2++){
			const short int bf2_first=shell2bf[s2];
			const short int n2=obs[s2].size();
			for (short int s3=0;s3<=s1;s3++){
				const short int bf3_first=shell2bf[s3];
				const short int n3=obs[s3].size();
				//for (short int s4=0;s4<=((s1==s3)?s2:s3);s4++){ // ((s1==s3)?s2:s3) is not a valid upper bound of s4, because some nonequivalent integrals may be neglected.
				for (short int s4=0;s4<=std::max(s2,s3);s4++){
					const short int bf4_first=shell2bf[s4];
					const short int n4=obs[s4].size();
					bool discard=true;
					int uniquebf=0; // Number of unique basis function quartets in the shell quartets.
					for (short int f1=0;f1!=n1;f1++){
						const short int bf1=f1+bf1_first;
						for (short int f2=0;f2!=n2;f2++){
							const short int bf2=f2+bf2_first;
							for (short int f3=0;f3!=n3;f3++){
								const short int bf3=f3+bf3_first;
								for (short int f4=0;f4!=n4;f4++){
									const short int bf4=f4+bf4_first;
									if (bf2<=bf1 && bf3<=bf1 && bf4<=((bf1==bf3)?bf2:bf3)){
										uniquebf++;
										const double integral1=repulsiondiag(bf1,bf2);
										const double integral2=repulsiondiag(bf3,bf4);
										const double upperbound=sqrt(abs(integral1*integral2)); // According to Cauchy-Schwarz inequality, sqrt(integral1*integral2) is the upper bound of (bf1,bf2|bf3,bf4). If the upper bound of any basis function quartet of (bf1,bf2,bf3,bf4) in the shell quartet (s1,s2,s3,s4) is larger than 10^-10, the shell quartet will not be discarded in the following two-electron integral evaluation.
										if (upperbound>=__integral_threshold__){
											discard=false;
										}
									}
								}
							}
						}
					}
					if (! discard){
						nshellquartets++; // This shell quartet will not be discarded.
						n2integrals=n2integrals+uniquebf; // All unique integrals in this shell quartet will not be discarded.
					}
				}
			}
		}
	}
	if (output) std::cout<<"Number of two electron integrals after screening ... "<<n2integrals<<std::endl;
	return n2integrals;
}

void Repulsion(const int natoms,double * atoms,const char * basisset,int nshellquartets,EigenMatrix repulsiondiag,long int n2integrals,double * repulsion,short int * indices,const int nprocs,const bool output){
	if (output) std::cout<<"Calculating electron repulsion integrals ... ";
	time_t start=time(0);
	__Basis_From_Atoms__
	const auto shell2bf=obs.shell2bf();
	short int * shellquartets=new short int[nshellquartets*4];
	short int * s1s=shellquartets+0*nshellquartets;
	short int * s2s=shellquartets+1*nshellquartets;
	short int * s3s=shellquartets+2*nshellquartets;
	short int * s4s=shellquartets+3*nshellquartets;
	long int s1234=0;
	for (short int s1=0;s1<(short int)obs.size();s1++){ // In this loop, indices of shell quartets to be computed are obtained.
		const short int bf1_first=shell2bf[s1];
		const short int n1=obs[s1].size();
		for (short int s2=0;s2<=s1;s2++){
			const short int bf2_first=shell2bf[s2];
			const short int n2=obs[s2].size();
			for (short int s3=0;s3<=s1;s3++){
				const short int bf3_first=shell2bf[s3];
				const short int n3=obs[s3].size();
				//for (short int s4=0;s4<=((s1==s3)?s2:s3);s4++){ // ((s1==s3)?s2:s3) is not a valid upper bound of s4, because some nonequivalent integrals may be neglected.
				for (short int s4=0;s4<=std::max(s2,s3);s4++){
					const short int bf4_first=shell2bf[s4];
					const short int n4=obs[s4].size();
					bool discard=true;
					for (short int f1=0;f1!=n1 && discard;f1++){
						const short int bf1=f1+bf1_first;
						for (short int f2=0;f2!=n2 && discard;f2++){
							const short int bf2=f2+bf2_first;
							for (short int f3=0;f3!=n3 && discard;f3++){
								const short int bf3=f3+bf3_first;
								for (short int f4=0;f4!=n4 && discard;f4++){
									const short int bf4=f4+bf4_first;
									if (bf2<=bf1 && bf3<=bf1 && bf4<=((bf1==bf3)?bf2:bf3)){
										const double integral1=repulsiondiag(bf1,bf2);
										const double integral2=repulsiondiag(bf3,bf4);
										const double upperbound=sqrt(abs(integral1*integral2)); // According to Cauchy-Schwarz inequality, sqrt(integral1*integral2) is the upper bound of (bf1,bf2|bf3,bf4). If the upper bound of any basis function quartet of (bf1,bf2,bf3,bf4) in the shell quartet (s1,s2,s3,s4) is larger than 10^-10, the shell quartet will not be discarded in the following two-electron integral evaluation.
										if (upperbound>=__integral_threshold__){
											discard=false;
											s1s[s1234]=s1;
											s2s[s1234]=s2;
											s3s[s1234]=s3;
											s4s[s1234]=s4;
											s1234++;
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
	if (output) std::cout<<"Spawning "<<nprocs<<" threads; ";
	long int nsqperthread_fewer=nshellquartets/nprocs; // How many shell quartets a thread will compute. If the average number is A, the number of each thread is either a or (a+1), where a=floor(A). The number of threads to compute a quartets, x, and that to compute (a+1) quartets, y, can be obtained by solving (1) a*x+(a+1)*y=b and (2) x+y=c, where b and c stand for the total numbers of quartes and threads respectively.
	int ntimes_fewer=nprocs-nshellquartets+nsqperthread_fewer*nprocs;
	long int * nsqperthread=new long int[nprocs]; // The number of shell quartets each thread is to compute.
	long int * isqfirstperthread=new long int[nprocs]; // The index of the first shell quartet each thread is to compute.
	long int * nbfperthread=new long int[nprocs]; // The number of basis function quartet each thread is to compute.
	long int * ibffirstperthread=new long int[nprocs]; // The index of the first basis function quartet each thread is to compute.
	for (int iproc=0;iproc<nprocs;iproc++){
		nsqperthread[iproc]=iproc<ntimes_fewer?nsqperthread_fewer:(nsqperthread_fewer+1);
		isqfirstperthread[iproc]=iproc==0?0:(isqfirstperthread[iproc-1]+nsqperthread[iproc-1]);
		nbfperthread[iproc]=0;
		for (int isq=isqfirstperthread[iproc];isq<isqfirstperthread[iproc]+nsqperthread[iproc];isq++){
			const short int s1=s1s[isq];const short int bf1_first=shell2bf[s1];const short int n1=obs[s1].size();
			const short int s2=s2s[isq];const short int bf2_first=shell2bf[s2];const short int n2=obs[s2].size();
			const short int s3=s3s[isq];const short int bf3_first=shell2bf[s3];const short int n3=obs[s3].size();
			const short int s4=s4s[isq];const short int bf4_first=shell2bf[s4];const short int n4=obs[s4].size();
			for (short int f1=0;f1!=n1;f1++){
				const short int bf1=f1+bf1_first;
				for (short int f2=0;f2!=n2;f2++){
					const short int bf2=f2+bf2_first;
					for (short int f3=0;f3!=n3;f3++){
						const short int bf3=f3+bf3_first;
						for (short int f4=0;f4!=n4;f4++){
							const short int bf4=f4+bf4_first;
							if (bf2<=bf1 && bf3<=bf1 && bf4<=((bf1==bf3)?bf2:bf3)){
								nbfperthread[iproc]++;
							}
						}
					}
				}
			}
		}
		ibffirstperthread[iproc]=iproc==0?0:(ibffirstperthread[iproc-1]+nbfperthread[iproc-1]);
	}

	libint2::initialize();
	Eigen::initParallel();
	omp_set_num_threads(nprocs);
	#pragma omp parallel for
	for (int iproc=0;iproc<nprocs;iproc++){
		time_t tstart=time(0);
		long int nsq=nsqperthread[iproc];
		long int isqfirst=isqfirstperthread[iproc];
		long int ibffirst=ibffirstperthread[iproc];
		double * repulsionranger=repulsion+ibffirst;
		short int * thisdegs=indices+0*n2integrals+ibffirst; // Four indices and one degenerate degree.
		short int * thisbf1s=indices+1*n2integrals+ibffirst;
		short int * thisbf2s=indices+2*n2integrals+ibffirst;
		short int * thisbf3s=indices+3*n2integrals+ibffirst;
		short int * thisbf4s=indices+4*n2integrals+ibffirst;
		short int * thiss1s=s1s+isqfirst;
		short int * thiss2s=s2s+isqfirst;
		short int * thiss3s=s3s+isqfirst;
		short int * thiss4s=s4s+isqfirst;
		libint2::Engine engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l());
		const auto & buf_vec=engine.results();
		for (int isq=0,ibfq=0;isq<nsq;isq++){
			const short int s1=thiss1s[isq];const short int bf1_first=shell2bf[s1];const short int n1=obs[s1].size();
			const short int s2=thiss2s[isq];const short int bf2_first=shell2bf[s2];const short int n2=obs[s2].size();
			const short int s3=thiss3s[isq];const short int bf3_first=shell2bf[s3];const short int n3=obs[s3].size();
			const short int s4=thiss4s[isq];const short int bf4_first=shell2bf[s4];const short int n4=obs[s4].size();
			engine.compute(obs[s1],obs[s2],obs[s3],obs[s4]);
			const auto ints_shellset=buf_vec[0];
			if (ints_shellset==nullptr) continue;
			for (short int f1=0,f1234=0;f1<n1;f1++){
				const short int bf1=bf1_first+f1;
				for (short int f2=0;f2<n2;f2++){
					const short int bf2=bf2_first+f2;
					const short int ab_deg=(bf1==bf2)?1:2;
					for (short int f3=0;f3<n3;f3++){
						const short int bf3=bf3_first+f3;
						for (short int f4=0;f4<n4;f4++,f1234++){
							const short int bf4=bf4_first+f4;
							const short int cd_deg=(bf3==bf4)?1:2;
							const short int ab_cd_deg=(bf1==bf3)?(bf2==bf4?1:2):2;
							const short int abcd_deg=ab_deg*cd_deg*ab_cd_deg;
							if (bf2<=bf1 && bf3<=bf1 && bf4<=((bf1==bf3)?bf2:bf3)){
								*(repulsionranger++)=ints_shellset[f1234];
								thisdegs[ibfq]=abcd_deg;
								thisbf1s[ibfq]=bf1;
								thisbf2s[ibfq]=bf2;
								thisbf3s[ibfq]=bf3;
								thisbf4s[ibfq]=bf4;
								ibfq++;
							}
						}
					}
				}
			}
		}
		time_t tend=time(0);
		if (output) std::cout<<"Thread "<<iproc<<" done in "<<tend-tstart<<"s; ";
	}
	libint2::finalize();
	delete [] shellquartets;
	delete [] nsqperthread;
	delete [] isqfirstperthread;
	delete [] nbfperthread;
	delete [] ibffirstperthread;
	time_t end=time(0);
	if (output) std::cout<<"done "<<end-start<<" s"<<std::endl;
}

