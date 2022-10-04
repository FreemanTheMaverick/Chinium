#include <libint2.hpp>
#include <vector> // Atom vectors.
#include <ctime>
#include <iostream>
#include <stdlib.h>
#include <omp.h>

#define __integral_threshold__ 10e-10

std::vector<libint2::Atom> Libint2Atoms(const int natoms,double * atoms){ // Converting atoms array to libint's std::vector<libint2::Atom>.
	std::vector<libint2::Atom> libint2atoms(natoms);
	for (int iatom=0;iatom<natoms;iatom++){
		libint2::Atom atomi;
		atomi.atomic_number=(int)atoms[iatom*4];
		atomi.x=atoms[iatom*4+1];
		atomi.y=atoms[iatom*4+2];
		atomi.z=atoms[iatom*4+3];
		libint2atoms[iatom]=atomi;
	}
	return libint2atoms;
}

double NuclearRepulsion(const int natoms,double * atoms){
	double nuclearrepulsion=0;
	for (int iatom=0;iatom<natoms;iatom++){
		const double Zi=atoms[iatom*4];
		const double xi=atoms[iatom*4+1];
		const double yi=atoms[iatom*4+2];
		const double zi=atoms[iatom*4+3];
		for (int jatom=0;jatom<iatom;jatom++){
			const double Zj=atoms[jatom*4];
			const double xj=atoms[jatom*4+1];
			const double yj=atoms[jatom*4+2];
			const double zj=atoms[jatom*4+3];
			const double dij=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
			nuclearrepulsion=nuclearrepulsion+Zi*Zj/dij;
		}
	}
	std::cout<<"Nuclear repulsion energy ... "<<nuclearrepulsion<<" a.u."<<std::endl;
	return nuclearrepulsion;
}

int nBasis_from_obs(libint2::BasisSet obs){ // Size of basis set directly derived from libint2::BasisSet.
	int n=0;
	for (const auto& shell:obs){
		n=n+shell.size();
	}
	return n;
}

int nBasis(const int natoms,double * atoms,const char * basisset){ // Size of basis set.
	std::vector<libint2::Atom> libint2atoms=Libint2Atoms(natoms,atoms);
	libint2::BasisSet obs(basisset,libint2atoms);
	int n=nBasis_from_obs(obs);
	std::cout<<"Number of atomic bases ... "<<n<<std::endl;
	return n;
}

int nOneElectronIntegrals(const int natoms,double * atoms,const char * basisset){ // Number of one-electron integrals.
	std::vector<libint2::Atom> libint2atoms=Libint2Atoms(natoms,atoms);
	libint2::BasisSet obs(basisset,libint2atoms);
	int nbasis=nBasis_from_obs(obs);
	int n1integrals=(1+nbasis)*nbasis/2;
	std::cout<<"Number of one-electron integrals in total ... "<<n1integrals<<std::endl;
	return n1integrals;
}

void OneElectronIntegrals(const int natoms,double * atoms,const char * basisset,char type,double * matrix){ // Computing various one-electron integrals and saving them in lower triangular matrix.
	libint2::Operator operator_=libint2::Operator::overlap;
	if (type=='s'){
	        std::cout<<"Calculating overlap integrals ... ";
		operator_=libint2::Operator::overlap;
	}else if (type=='k'){
	        std::cout<<"Calculating kinetic integrals ... ";
		operator_=libint2::Operator::kinetic;
	}else if (type=='n'){
	        std::cout<<"Calculating nuclear integrals ... ";
		operator_=libint2::Operator::nuclear;
	}
	time_t start=time(0);
	std::vector<libint2::Atom> libint2atoms=Libint2Atoms(natoms,atoms);
	libint2::BasisSet obs(basisset,libint2atoms);
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
						matrix[bf1*(bf1+1)/2+bf2]=ints_shellset[f1*n2+f2]; // One-electron integral matrix is symmetric, stored in lower triangular format.
					}
				}
			}
		}
	}
	libint2::finalize();
	time_t end=time(0);
	std::cout<<"done "<<end-start<<" s"<<std::endl;
}

void Overlap(const int natoms,double * atoms,const char * basisset,double * overlap){
	OneElectronIntegrals(natoms,atoms,basisset,'s',overlap);
}

void Kinetic(const int natoms,double * atoms,const char * basisset,double * kinetic){
	OneElectronIntegrals(natoms,atoms,basisset,'k',kinetic);
}

void Nuclear(const int natoms,double * atoms,const char * basisset,double * nuclear){
	OneElectronIntegrals(natoms,atoms,basisset,'n',nuclear);
}

void RepulsionDiag(const int natoms,double * atoms,const char * basisset,double * repulsiondiag){ // Computing the diagonal elements of electron repulsion tensor. Used for Cauchy-Schwarz screening.
	std::cout<<"Calculating diagonal elements of repulsion integrals ... ";
	time_t start=time(0);
	std::vector<libint2::Atom> libint2atoms=Libint2Atoms(natoms,atoms);
	libint2::BasisSet obs(basisset,libint2atoms);
	const auto shell2bf=obs.shell2bf();
	libint2::initialize();
	libint2::Engine engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l());
	const auto & buf_vec=engine.results();
	for (short int s1=0;s1<(short int)obs.size();s1++){
		const short int bf1_first=shell2bf[s1];
		const short int n1=obs[s1].size();
		for (short int s2=0;s2<=s1;s2++){
			const short int bf2_first=shell2bf[s2];
			const short int n2=obs[s2].size();
			engine.compute(obs[s1],obs[s2],obs[s1],obs[s2]); // Computing the integral (12|12).
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
								repulsiondiag[bf1*(bf1+1)/2+bf2]=buf_1234[f1234];
							}
						}
					}
				}
			}
		}
	}
	libint2::finalize();
	time_t end=time(0);
	std::cout<<"done "<<end-start<<" s"<<std::endl;
}

long int nTwoElectronIntegrals(const int natoms,double * atoms,const char * basisset,double * repulsiondiag,int & nshellquartets){ // Numbers of two-electron integrals and nonequivalent shell quartets after Cauchy-Schwarz screening.
	std::vector<libint2::Atom> libint2atoms=Libint2Atoms(natoms,atoms);
	libint2::BasisSet obs(basisset,libint2atoms);
        long int nbasis=(long int)nBasis_from_obs(obs);
	long int n2integrals=nbasis*(nbasis+1)*(nbasis*(nbasis+1)/2+1)/4;
        std::cout<<"Number of two electron integrals in total ... "<<n2integrals<<std::endl;
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
				//for (short int s4=0;s4<=((s1==s3)?s2:s3);s4++){
				//for (short int s4=0;s4<=s1;s4++){
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
										const double integral1=repulsiondiag[bf1*(bf1+1)/2+bf2];
										const double integral2=bf3>bf4?repulsiondiag[bf3*(bf3+1)/2+bf4]:repulsiondiag[bf4*(bf4+1)/2+bf3];
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
	std::cout<<"Number of two electron integrals after screening ... "<<n2integrals<<std::endl;
	return n2integrals;
}

void Repulsion(const int natoms,double * atoms,const char * basisset,int nshellquartets,double * repulsiondiag,double * repulsion,short int * indices){
	std::cout<<"Calculating electron repulsion integrals ... ";
	time_t start=time(0);
	std::vector<libint2::Atom> libint2atoms=Libint2Atoms(natoms,atoms);
	libint2::BasisSet obs(basisset,libint2atoms);
	const auto shell2bf=obs.shell2bf();
	short int * shellquartets=new short int[nshellquartets*4];
	short int * shellquartetsranger=shellquartets;
	for (short int s1=0;s1<(short int)obs.size();s1++){
		const short int bf1_first=shell2bf[s1];
		const short int n1=obs[s1].size();
		for (short int s2=0;s2<=s1;s2++){
			const short int bf2_first=shell2bf[s2];
			const short int n2=obs[s2].size();
			for (short int s3=0;s3<=s1;s3++){
				const short int bf3_first=shell2bf[s3];
				const short int n3=obs[s3].size();
				//for (short int s4=0;s4<=((s1==s3)?s2:s3);s4++){
				//for (short int s4=0;s4<=s1;s4++){
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
										const double integral1=repulsiondiag[bf1*(bf1+1)/2+bf2];
										const double integral2=bf3>bf4?repulsiondiag[bf3*(bf3+1)/2+bf4]:repulsiondiag[bf4*(bf4+1)/2+bf3];
										const double upperbound=sqrt(abs(integral1*integral2)); // According to Cauchy-Schwarz inequality, sqrt(integral1*integral2) is the upper bound of (bf1,bf2|bf3,bf4). If the upper bound of any basis function quartet of (bf1,bf2,bf3,bf4) in the shell quartet (s1,s2,s3,s4) is larger than 10^-10, the shell quartet will not be discarded in the following two-electron integral evaluation.
										if (upperbound>=__integral_threshold__){
											discard=false;
											*(shellquartetsranger++)=s1;
											*(shellquartetsranger++)=s2;
											*(shellquartetsranger++)=s3;
											*(shellquartetsranger++)=s4;
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
	int nprocs=atoi(getenv("OMP_NUM_THREADS"));
	std::cout<<"Spawning "<<nprocs<<" threads; ";
	omp_set_num_threads(nprocs);
	long int nsqperthread_fewer=nshellquartets/nprocs;
	int ntimes_fewer=nprocs-nshellquartets+nsqperthread_fewer*nprocs;
	long int nsqperthread[nprocs];
	long int isqfirstperthread[nprocs];
	long int nbfperthread[nprocs];
	long int ibffirstperthread[nprocs];
	for (int iproc=0;iproc<nprocs;iproc++){
		nsqperthread[iproc]=iproc<ntimes_fewer?nsqperthread_fewer:(nsqperthread_fewer+1);
		isqfirstperthread[iproc]=iproc==0?0:(isqfirstperthread[iproc-1]+nsqperthread[iproc-1]);
		nbfperthread[iproc]=0;
		for (int isq=isqfirstperthread[iproc];isq<isqfirstperthread[iproc]+nsqperthread[iproc];isq++){
			const short int s1=shellquartets[isq*4];const short int bf1_first=shell2bf[s1];const short int n1=obs[s1].size();
			const short int s2=shellquartets[isq*4+1];const short int bf2_first=shell2bf[s2];const short int n2=obs[s2].size();
			const short int s3=shellquartets[isq*4+2];const short int bf3_first=shell2bf[s3];const short int n3=obs[s3].size();
			const short int s4=shellquartets[isq*4+3];const short int bf4_first=shell2bf[s4];const short int n4=obs[s4].size();
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
	#pragma omp parallel for
	for (int iproc=0;iproc<nprocs;iproc++){
		time_t tstart=time(0);
		long int nsq=nsqperthread[iproc];
		long int isqfirst=isqfirstperthread[iproc];
		long int ibffirst=ibffirstperthread[iproc];
		double * repulsionranger=repulsion+ibffirst;
		short int * indicesranger=indices+ibffirst*4;
		short int * sqranger=shellquartets+isqfirst*4;
		libint2::Engine engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l());
		const auto & buf_vec=engine.results();
		for (int isq=0;isq<nsq;isq++){
			const short int s1=*(sqranger++);const short int bf1_first=shell2bf[s1];const short int n1=obs[s1].size();
			const short int s2=*(sqranger++);const short int bf2_first=shell2bf[s2];const short int n2=obs[s2].size();
			const short int s3=*(sqranger++);const short int bf3_first=shell2bf[s3];const short int n3=obs[s3].size();
			const short int s4=*(sqranger++);const short int bf4_first=shell2bf[s4];const short int n4=obs[s4].size();
			engine.compute(obs[s1],obs[s2],obs[s3],obs[s4]);
			const auto ints_shellset=buf_vec[0];
			if (ints_shellset==nullptr){
				continue;
			}
			for (short int f1=0,f1234=0;f1<n1;f1++){
				const short int bf1=bf1_first+f1;
				for (short int f2=0;f2<n2;f2++){
					const short int bf2=bf2_first+f2;
					for (short int f3=0;f3<n3;f3++){
						const short int bf3=bf3_first+f3;
						for (short int f4=0;f4<n4;f4++,f1234++){
							const short int bf4=bf4_first+f4;
							if (bf2<=bf1 && bf3<=bf1 && bf4<=((bf1==bf3)?bf2:bf3)){
								*(repulsionranger++)=ints_shellset[f1234];
								*(indicesranger++)=bf1;
								*(indicesranger++)=bf2;
								*(indicesranger++)=bf3;
								*(indicesranger++)=bf4;
							}
						}
					}
				}
			}
		}
		time_t tend=time(0);
		std::cout<<"Thread "<<iproc<<" done in "<<tend-tstart<<"s; ";
	}
	libint2::finalize();
	delete [] shellquartets;
	time_t end=time(0);
	std::cout<<"done "<<end-start<<" s"<<std::endl;
}


/*
void SaveHDF5(hid_t file,int n1integrals,double * matrix,char * title){
	hsize_t dims[1]={(long long unsigned int)n1integrals};
	hid_t space=H5Screate_simple(1,dims,NULL);
	hid_t dcpl=H5Pcreate(H5P_DATASET_CREATE);
	H5Pset_layout(dcpl,H5D_COMPACT);
	hid_t dset=H5Dcreate(file,title,H5T_NATIVE_DOUBLE,space,H5P_DEFAULT,dcpl,H5P_DEFAULT);
	H5Dwrite(dset,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,matrix);
	H5Dclose(dset);
	H5Pclose(dcpl);
	H5Sclose(space);
}
	hid_t file=H5Fcreate("_atomicintegrals.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
	SaveHDF5(file,n1integrals,overlap,(char *)"overlap");
	SaveHDF5(file,n1integrals,kinetic,(char *)"kinetic");
	SaveHDF5(file,n1integrals,nuclear,(char *)"nuclear");
	SaveHDF5(file,n1integrals,repulsiondiag,(char *)"repulsiondiag");
	H5Fclose(file);
*/


