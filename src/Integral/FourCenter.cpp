#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath> // std::abs, std::sqrt
#include <tuple> // std::tuple, std::make_tuple, std::tie
#include <omp.h>

#include "../Macro.h"
#include "../Multiwfn.h" // Requires <Eigen/Dense>, <vector>, <string>, "Macro.h".
#include "Macro.h"

#include <iostream>


EigenMatrix getRepulsionDiag(libint2::BasisSet& obs){ // Computing the diagonal elements of electron repulsion tensor for Cauchy-Schwarz screening.
	const int nbasis = libint2::nbf(obs);
	EigenMatrix repulsiondiag(nbasis, nbasis);
	libint2::initialize();
	libint2::Engine engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l());
	const auto& buf_vec = engine.results();
	const auto shell2bf = obs.shell2bf();
	for ( short int s1 = 0; s1 < (short int)obs.size(); s1++ ){
		const short int bf1_first = shell2bf[s1];
		const short int n1 = obs[s1].size();
		for ( short int s2 = 0; s2 <= s1; s2++ ){
			const short int bf2_first = shell2bf[s2];
			const short int n2 = obs[s2].size();
			engine.compute(obs[s1], obs[s2], obs[s1], obs[s2]); // Computing the integrals in the shell quartet (12|12).
			const auto* buf_1234 = buf_vec[0];
			if ( buf_1234 == nullptr ){
				continue;
			}
			for ( short int f1 = 0, f1234 = 0; f1 < n1; f1++ ){ // Integrals are stored in buffer as four-dimensional tensor.
				const short int bf1 = f1 + bf1_first;
				for ( short int f2 = 0; f2 < n2; f2++ ){
					const short int bf2 = f2 + bf2_first;
					for ( short int f3 = 0; f3 < n1; f3++ ){
						const short int bf3 = f3 + bf1_first;
						for ( short int f4 = 0; f4 < n2; f4++, f1234++ ){
							const short int bf4 = f4 + bf2_first;
							if ( bf1 == bf3 && bf2 == bf4 && bf1 >= bf2 ){ // Considering only the unique diagonal elements with (bf1==bf3 && bf2==bf4).
								repulsiondiag(bf1, bf2) = buf_1234[f1234]; // Repulsion diagonal integral matrix is symmetric.
								repulsiondiag(bf2, bf1) = buf_1234[f1234];
							}
						}
					}
				}
			}
		}
	}
	libint2::finalize();
	return repulsiondiag;
}

std::tuple<long int, long int> getRepulsionLength(libint2::BasisSet& obs, EigenMatrix repulsiondiag, double threshold){ // Numbers of nonequivalent two-electron integrals and shell quartets after Cauchy-Schwarz screening.
	const auto shell2bf=obs.shell2bf();
	long int n2integrals=0; // Number of integrals not discarded.
	long int nshellquartets=0; // Number of shell quartets not discarded.
	for ( short int s1 = 0; s1 < (short int)obs.size() ; s1++ ){
		const short int bf1_first = shell2bf[s1];
		const short int n1 = obs[s1].size();
		for ( short int s2 = 0; s2 <= s1; s2++ ){
			const short int bf2_first = shell2bf[s2];
			const short int n2 = obs[s2].size();
			for ( short int s3 = 0; s3 <= s1; s3++ ){
				const short int bf3_first = shell2bf[s3];
				const short int n3 = obs[s3].size();
				//for ( short int s4 = 0; s4 <= ( (s1 == s3) ? s2 : s3 ); s4++ ){ // ((s1==s3)?s2:s3) is not a valid upper bound of s4, because some nonequivalent integrals may be neglected.
				for ( short int s4 = 0; s4 <= std::max(s2, s3); s4++ ){
					const short int bf4_first = shell2bf[s4];
					const short int n4 = obs[s4].size();
					bool discard = true;
					int uniquebf = 0; // Number of unique basis function quartets in the shell quartets.
					for ( short int f1 = 0; f1 < n1; f1++ ){
						const short int bf1 = f1 + bf1_first;
						for ( short int f2 = 0; f2 < n2; f2++ ){
							const short int bf2 = f2 + bf2_first;
							for ( short int f3 = 0; f3 < n3; f3++ ){
								const short int bf3 = f3 + bf3_first;
								for ( short int f4 = 0; f4 < n4; f4++ ){
									const short int bf4 = f4 + bf4_first;
									if ( bf2 <= bf1 && bf3 <= bf1 && bf4 <= ( (bf1==bf3) ? bf2 : bf3 ) ){
										uniquebf++;
										const double integral1 = repulsiondiag(bf1, bf2);
										const double integral2 = repulsiondiag(bf3, bf4);
										const double upperbound = std::sqrt( std::abs(integral1 * integral2) ); // According to Cauchy-Schwarz inequality, sqrt(integral1*integral2) is the upper bound of (bf1,bf2|bf3,bf4). If the upper bound of any basis function quartet of (bf1,bf2,bf3,bf4) in the shell quartet (s1,s2,s3,s4) is larger than 10^-10, the shell quartet will not be discarded in the following two-electron integral evaluation.
										if ( upperbound > threshold ){
											discard = false;
										}
									}
								}
							}
						}
					}
					if (! discard){
						nshellquartets++; // This shell quartet will not be discarded.
						n2integrals = n2integrals+uniquebf; // All unique integrals in this shell quartet will not be discarded.
					}
				}
			}
		}
	}
	return std::make_tuple(n2integrals, nshellquartets);
}

void getRepulsionIndices(
		libint2::BasisSet& obs, EigenMatrix repulsiondiag,
		double threshold,
		short int* shellis, short int* shelljs,
		short int* shellks, short int* shellls){
	const auto shell2bf = obs.shell2bf();
	short int* s1s = shellis;
	short int* s2s = shelljs;
	short int* s3s = shellks;
	short int* s4s = shellls;
	for ( short int s1 = 0; s1 < (short int)obs.size(); s1++ ){ // In this loop, indices of shell quartets to be computed are obtained.
		const short int bf1_first = shell2bf[s1];
		const short int n1 = obs[s1].size();
		for ( short int s2 = 0; s2 <= s1; s2++ ){
			const short int bf2_first = shell2bf[s2];
			const short int n2 = obs[s2].size();
			for ( short int s3 = 0; s3 <= s1; s3++ ){
				const short int bf3_first = shell2bf[s3];
				const short int n3 = obs[s3].size();
				//for (short int s4=0;s4<=((s1==s3)?s2:s3);s4++){ // ((s1==s3)?s2:s3) is not a valid upper bound of s4, because some nonequivalent integrals may be neglected.
				for ( short int s4 = 0; s4 <= std::max(s2,s3); s4++ ){
					const short int bf4_first = shell2bf[s4];
					const short int n4 = obs[s4].size();
					bool discard = true;
					for ( short int f1 = 0; f1 < n1 && discard; f1++ ){
						const short int bf1 = f1 + bf1_first;
						for ( short int f2 = 0; f2 < n2 && discard; f2++){
							const short int bf2 = f2 + bf2_first;
							for ( short int f3 = 0; f3 < n3 && discard; f3++ ){
								const short int bf3 = f3 + bf3_first;
								for ( short int f4 = 0; f4 < n4 && discard; f4++ ){
									const short int bf4 = f4 + bf4_first;
									if ( bf2 <= bf1 && bf3 <= bf1 && bf4 <= ( (bf1==bf3) ? bf2 : bf3 ) ){
										const double integral1 = repulsiondiag(bf1,bf2);
										const double integral2 = repulsiondiag(bf3,bf4);
										const double upperbound=std::sqrt(std::abs(integral1 * integral2)); // According to Cauchy-Schwarz inequality, sqrt(integral1*integral2) is the upper bound of (bf1,bf2|bf3,bf4). If the upper bound of any basis function quartet of (bf1,bf2,bf3,bf4) in the shell quartet (s1,s2,s3,s4) is larger than 10^-10, the shell quartet will not be discarded in the following two-electron integral evaluation.
										if ( upperbound >= threshold){
											discard = false;
											*(s1s++) = s1;
											*(s2s++) = s2;
											*(s3s++) = s3;
											*(s4s++) = s4;
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

std::tuple<std::vector<long int>, std::vector<long int>> getThreadPointers(
		libint2::BasisSet& obs, long int nshellquartets, int nthreads,
		short int* shellis, short int* shelljs,
		short int* shellks, short int* shellls){
	const auto shell2bf = obs.shell2bf();
	const long int nsq_fewer = nshellquartets / nthreads; // How many shell quartets a thread will compute. If the average number is A, the number of each thread is either a or (a+1), where a=floor(A). The number of threads to compute a quartets, x, and that to compute (a+1) quartets, y, can be obtained by solving (1) a*x+(a+1)*y=b and (2) x+y=c, where b and c stand for the total numbers of quartets and threads respectively.
	const int nfewers = nthreads - nshellquartets + nsq_fewer * nthreads;
	//std::vector<long int> nsqs(nthreads, 0); // The number of shell quartets each thread is to compute.
	long int nsqs = 0;
	std::vector<long int> sqheads(nthreads, 0); // The index of the first shell quartet each thread is to compute.
	std::vector<long int> bqheads(nthreads, 0); // The index of the first basis function quartet each thread is to compute.
	std::vector<long int> nbqs(nthreads, 0); // The number of basis function quartet each thread is to compute.
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		if ( ithread > 0 ) sqheads[ithread] = sqheads[ithread - 1] + nsqs;
		nsqs = ( ithread < nfewers ) ? nsq_fewer : ( nsq_fewer + 1 );
		for ( long int isq = sqheads[ithread]; isq < sqheads[ithread] + nsqs; isq++ ){
			const short int s1 = shellis[isq];
			const short int s2 = shelljs[isq];
			const short int s3 = shellks[isq];
			const short int s4 = shellls[isq];
			const short int bf1_first = shell2bf[s1];
			const short int bf2_first = shell2bf[s2];
			const short int bf3_first = shell2bf[s3];
			const short int bf4_first = shell2bf[s4];
			const short int n1 = obs[s1].size();
			const short int n2 = obs[s2].size();
			const short int n3 = obs[s3].size();
			const short int n4 = obs[s4].size();
			for ( short int f1 = 0; f1 < n1; f1++ ){
				const short int bf1 = f1 + bf1_first;
				for ( short int f2 = 0; f2 < n2; f2++ ){
					const short int bf2 = f2 + bf2_first;
					for ( short int f3 = 0; f3 < n3; f3++ ){
						const short int bf3 = f3 + bf3_first;
						for ( short int f4 = 0; f4 < n4; f4++ ){
							const short int bf4 = f4 + bf4_first;
							if ( bf2 <= bf1 && bf3 <= bf1 && bf4 <= ( (bf1==bf3) ? bf2 : bf3 ) ){
								nbqs[ithread]++;
							}
						}
					}
				}
			}
		}
		if ( ithread > 0 ) bqheads[ithread] = bqheads[ithread - 1] + nbqs[ithread - 1];
	}
	return std::make_tuple(sqheads, bqheads);
}

void getRepulsion0(
		libint2::BasisSet& obs,
		std::vector<long int> sqheads, std::vector<long int> bqheads,
		short int* shellis, short int* shelljs,
		short int* shellks, short int* shellls,
		short int* basisis, short int* basisjs,
		short int* basisks, short int* basisls,
		char* degs, double* repulsions){
	const auto shell2bf = obs.shell2bf();
	libint2::initialize();
	const int nthreads = bqheads.size();
	omp_set_num_threads(nthreads);
	#pragma omp parallel for
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		const long int nsq = sqheads[ithread + 1] - sqheads[ithread];
		const long int sqhead = sqheads[ithread];
		const long int bqhead = bqheads[ithread];
		double* repulsionranger = repulsions + bqhead;
		short int* bf1ranger = basisis + bqhead;
		short int* bf2ranger = basisjs + bqhead;
		short int* bf3ranger = basisks + bqhead;
		short int* bf4ranger = basisls + bqhead;
		char* degranger = degs + bqhead;
		short int* s1ranger = shellis + sqhead;
		short int* s2ranger = shelljs + sqhead;
		short int* s3ranger = shellks + sqhead;
		short int* s4ranger = shellls + sqhead;
		libint2::Engine engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l());
		const auto& buf_vec = engine.results();
		for ( int isq = 0; isq < nsq; isq++ ){
			const short int s1 = *(s1ranger++);
			const short int s2 = *(s2ranger++);
			const short int s3 = *(s3ranger++);
			const short int s4 = *(s4ranger++);
			const short int bf1_first = shell2bf[s1];
			const short int bf2_first = shell2bf[s2];
			const short int bf3_first = shell2bf[s3];
			const short int bf4_first = shell2bf[s4];
			const short int n1 = obs[s1].size();
			const short int n2 = obs[s2].size();
			const short int n3 = obs[s3].size();
			const short int n4 = obs[s4].size();
			engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
			const auto ints_shellset = buf_vec[0];
			if ( ints_shellset == nullptr ) continue;
			for ( short int f1 = 0, f1234 = 0; f1 < n1; f1++ ){
				const short int bf1 = bf1_first + f1;
				for ( short int f2 = 0; f2 < n2; f2++ ){
					const short int bf2 = bf2_first + f2;
					const char ab_deg = (bf1 == bf2) ? 1 : 2;
					for ( short int f3 = 0; f3 < n3; f3++ ){
						const short int bf3 = bf3_first + f3;
						for ( short int f4 = 0; f4 < n4; f4++, f1234++ ){
							const short int bf4 = bf4_first + f4;
							const char cd_deg = (bf3 == bf4) ? 1 : 2;
							const char ab_cd_deg = (bf1 == bf3) ? (bf2 == bf4 ? 1 : 2) : 2;
							const char abcd_deg = ab_deg * cd_deg * ab_cd_deg;
							if ( bf2 <= bf1 && bf3 <= bf1 && bf4 <= ( (bf1 == bf3) ? bf2 : bf3 ) ){
								*(repulsionranger++) = ints_shellset[f1234];
								*(bf1ranger++) = bf1;
								*(bf2ranger++) = bf2;
								*(bf3ranger++) = bf3;
								*(bf4ranger++) = bf4;
								*(degranger++) = abcd_deg;
							}
						}
					}
				}
			}
		}
	}
	libint2::finalize();
}

void Multiwfn::getRepulsion(double threshold, int nthreads, const bool output){
	__Make_Basis_Set__

	if (output) std::printf("Calculating diagonal elements of repulsion integrals ... ");
	auto start = __now__;
	this->RepulsionDiag = getRepulsionDiag(obs);
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));

	if (output) std::printf("Calculating numbers of non-equivalent integrals and shell quartets after Cauchy-Schwarz screening ... ");
	start = __now__;
	long int nshellquartets = 0;
	std::tie(this->RepulsionLength, nshellquartets) = getRepulsionLength(obs, this->RepulsionDiag, threshold);
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));
	const long int nbasis = libint2::nbf(obs);
	const long int nshells = obs.size();
	if (output) std::printf("Before screening: %ld integrals and %ld shell quartets\n", nbasis*(nbasis+1)*(nbasis*(nbasis+1)/2+1)/4, nshells*(nshells+1)*(nshells*(nshells+1)/2+1)/4);
	if (output) std::printf("After screening: %ld integrals and %ld shell quartets\n", this->RepulsionLength, nshellquartets);
	if (output) std::printf("Memory needed for 4c-2e repulsion integrals and their indices: %f GB\n",(double)this->RepulsionLength * ( 2. * 4. + 1. + 8. ) / 1024. / 1024. / 1024.);

	if (output) std::printf("Generating indices of non-equivalent integrals and shell quartets after Cauchy-Schwarz screening ... ");
	start = __now__;
	short int* shellis = new short int[nshellquartets];
	short int* shelljs = new short int[nshellquartets];
	short int* shellks = new short int[nshellquartets];
	short int* shellls = new short int[nshellquartets];
	getRepulsionIndices(
			obs, this->RepulsionDiag, threshold,
			shellis, shelljs, shellks, shellls);
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));

	if (output) std::printf("Arranging for %d threads to calculate 4c-2e repulsion integrals ... ", nthreads);
	start = __now__;
	std::vector<long int> sqheads = {};
	std::vector<long int> bqheads = {};
	std::tie(sqheads, bqheads) = getThreadPointers(
		obs, nshellquartets, nthreads,
		shellis, shelljs, shellks, shellls);
	sqheads.push_back(nshellquartets);
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));

	if (output) std::printf("Calculating 4c-2e repulsion integrals ... ");
	start = __now__;
	this->RepulsionIs = new short int[this->RepulsionLength];
	this->RepulsionJs = new short int[this->RepulsionLength];
	this->RepulsionKs = new short int[this->RepulsionLength];
	this->RepulsionLs = new short int[this->RepulsionLength];
	this->RepulsionDegs = new char[this->RepulsionLength];
	this->Repulsions = new double[this->RepulsionLength];
	getRepulsion0(
		obs, sqheads, bqheads,
		shellis, shelljs, shellks, shellls,
		this->RepulsionIs, this->RepulsionJs,
		this->RepulsionKs, this->RepulsionLs,
		this->RepulsionDegs, this->Repulsions);
	if (output) std::printf("Done in %f s\n", __duration__(start, __now__));

	delete [] shellis;
	delete [] shelljs;
	delete [] shellks;
	delete [] shellls;
}
