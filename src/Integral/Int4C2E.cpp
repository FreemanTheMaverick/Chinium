#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath> // std::abs, std::sqrt
#include <tuple> // std::tuple, std::make_tuple, std::tie
#include <functional>
#include <chrono>
#include <omp.h>
#include <libmwfn.h>

#include "../Macro.h"

#include "Int4C2E.h"
#include "Parallel.h"
#include "Macro.h"

std::tuple<EigenMatrix, EigenMatrix> getRepulsionDiag(libint2::BasisSet& obs){ // Computing the diagonal elements of electron repulsion tensor for Cauchy-Schwarz screening.
	const int nbasis = libint2::nbf(obs);
	EigenMatrix Diag1212 = EigenZero(nbasis, nbasis);
	EigenMatrix Diag1122 = EigenZero(nbasis, nbasis);
	libint2::initialize();
	libint2::Engine engine1212(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l());
	libint2::Engine engine1122(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l());
	const auto& buf_vec1212 = engine1212.results();
	const auto& buf_vec1122 = engine1122.results();
	const auto shell2bf = obs.shell2bf();
	for ( short int s1 = 0; s1 < (short int)obs.size(); s1++ ){
		const short int bf1_first = shell2bf[s1];
		const short int n1 = obs[s1].size();
		for ( short int s2 = 0; s2 <= s1; s2++ ){
			const short int bf2_first = shell2bf[s2];
			const short int n2 = obs[s2].size();
			engine1212.compute(obs[s1], obs[s2], obs[s1], obs[s2]); // Computing the integrals in the shell quartet (12|12).
			const auto* buf_1212 = buf_vec1212[0];
			if ( buf_1212 == nullptr ) goto CALC1122;
			for ( short int f1 = 0, f1234 = 0; f1 < n1; f1++ ){ // Integrals are stored in buffer as four-dimensional tensor.
				const short int bf1 = f1 + bf1_first;
				for ( short int f2 = 0; f2 < n2; f2++ ){
					const short int bf2 = f2 + bf2_first;
					for ( short int f3 = 0; f3 < n1; f3++ ){
						const short int bf3 = f3 + bf1_first;
						for ( short int f4 = 0; f4 < n2; f4++, f1234++ ){
							const short int bf4 = f4 + bf2_first;
							if ( bf1 == bf3 && bf2 == bf4 && bf1 >= bf2 ){ // Considering only the unique diagonal elements with (bf1==bf3 && bf2==bf4).
								Diag1212(bf1, bf2) = buf_1212[f1234]; // Repulsion diagonal integral matrix is symmetric.
								Diag1212(bf2, bf1) = buf_1212[f1234];
							}
						}
					}
				}
			}
			CALC1122: engine1122.compute(obs[s1], obs[s1], obs[s2], obs[s2]); // Computing the integrals in the shell quartet (11|22).
			const auto* buf_1122 = buf_vec1122[0];
			if ( buf_1122 == nullptr ) continue;
			for ( short int f1 = 0, f1234 = 0; f1 < n1; f1++ ){ // Integrals are stored in buffer as four-dimensional tensor.
				const short int bf1 = f1 + bf1_first;
				for ( short int f2 = 0; f2 < n2; f2++ ){
					const short int bf2 = f2 + bf2_first;
					for ( short int f3 = 0; f3 < n1; f3++ ){
						const short int bf3 = f3 + bf1_first;
						for ( short int f4 = 0; f4 < n2; f4++, f1234++ ){
							const short int bf4 = f4 + bf2_first;
							if ( bf1 == bf3 && bf2 == bf4 && bf1 >= bf2 ){ // Considering only the unique diagonal elements with (bf1==bf3 && bf2==bf4).
								Diag1122(bf1, bf2) = buf_1122[f1234]; // Repulsion diagonal integral matrix is symmetric.
								Diag1122(bf2, bf1) = buf_1122[f1234];
							}
						}
					}
				}
			}
		}
	}
	libint2::finalize();
	return std::make_tuple(Diag1212, Diag1122);
}

std::tuple<long int, long int> getRepulsionLength(libint2::BasisSet& obs, EigenMatrix repulsiondiag, double threshold){ // Numbers of nonequivalent two-electron integrals and shell quartets after Cauchy-Schwarz screening.
	const auto shell2bf = obs.shell2bf();
	long int n2integrals = 0; // Number of integrals not discarded.
	long int nshellquartets = 0; // Number of shell quartets not discarded.
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
		double* repulsionints){
	const auto shell2bf = obs.shell2bf();
	libint2::initialize();
	const int nthreads = bqheads.size();
	#pragma omp parallel for num_threads(nthreads)
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		const long int nsq = sqheads[ithread + 1] - sqheads[ithread];
		const long int sqhead = sqheads[ithread];
		const long int bqhead = bqheads[ithread];
		double* repulsionranger = repulsionints + bqhead;
		short int* bf1ranger = basisis + bqhead;
		short int* bf2ranger = basisjs + bqhead;
		short int* bf3ranger = basisks + bqhead;
		short int* bf4ranger = basisls + bqhead;
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
								*(bf1ranger++) = bf1;
								*(bf2ranger++) = bf2;
								*(bf3ranger++) = bf3;
								*(bf4ranger++) = bf4;
								*(repulsionranger++) = ints_shellset[f1234] * abcd_deg;
							}
						}
					}
				}
			}
		}
	}
	libint2::finalize();
}

void EigenMatrixVectorSum( // For OMP reduction
		std::vector<std::vector<EigenMatrix>>& omp_out,
		std::vector<std::vector<EigenMatrix>>& omp_in){
	for ( int iatom = 0; iatom < (int)omp_out.size(); iatom++ )
		for ( int xyz = 0; xyz < 3; xyz++ )
			omp_out[iatom][xyz] += omp_in[iatom][xyz];
}

std::vector<std::vector<EigenMatrix>> getRepulsion1(
		libint2::BasisSet& obs,
		std::vector<int>& shell2atom,
		std::vector<long int> sqheads,
		short int* shellis, short int* shelljs,
		short int* shellks, short int* shellls,
		EigenMatrix D, double kscale){
	const int nbasis = libint2::nbf(obs);
	const int natoms = *std::max_element(shell2atom.begin(), shell2atom.end()) + 1;
	const auto shell2bf = obs.shell2bf();
	std::vector<std::vector<EigenMatrix>> rawjs(natoms);
	for ( std::vector<EigenMatrix>& rawj : rawjs )
		rawj = {EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis)};
	std::vector<std::vector<EigenMatrix>> rawks(natoms);
	for ( std::vector<EigenMatrix>& rawk : rawks )
		rawk = {EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis)};
 	libint2::initialize();
	const int nthreads = sqheads.size() - 1;
	#pragma omp declare reduction(Sum: std::vector<std::vector<EigenMatrix>>: EigenMatrixVectorSum(omp_out, omp_in)) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(Sum: rawjs, rawks) num_threads(nthreads)
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		const long int nsq = sqheads[ithread + 1] - sqheads[ithread];
		const long int sqhead = sqheads[ithread];
		short int* s1ranger = shellis + sqhead;
		short int* s2ranger = shelljs + sqhead;
		short int* s3ranger = shellks + sqhead;
		short int* s4ranger = shellls + sqhead;
		libint2::Engine engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l(), 1);
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
			if ( !buf_vec[0] ) continue;
			const int atomlist[] = {
				shell2atom[s1],
				shell2atom[s2],
				shell2atom[s3],
				shell2atom[s4]
			};
			for ( short int f1 = 0, f1234 = 0; f1 != n1; f1++ ){
				const short int bf1 = bf1_first + f1;
				for ( short int f2 = 0; f2 != n2; f2++ ){
					const short int bf2 = bf2_first + f2;
					const double ab_deg = (bf1 == bf2) ? 1 : 2;
					for ( short int f3 = 0; f3 != n3; f3++ ){
						const short int bf3 = bf3_first + f3;
						for ( short int f4 = 0; f4 != n4; f4++, f1234++ ){
							const short int bf4 = bf4_first + f4;
							if ( bf2 <= bf1 && bf3 <= bf1 && bf4 <= ((bf1 == bf3) ? bf2 : bf3)){
								const double cd_deg = (bf3 == bf4) ? 1 : 2;
								const double ab_cd_deg = (bf1 == bf3) ? (bf2 == bf4 ? 1 : 2) : 2;
								const double abcd_deg = ab_deg * cd_deg * ab_cd_deg;
								double tmp = 114514;
								int atom = 1919810;
								for ( int p = 0, pt = 0; p < 4; p++ ){
									atom = atomlist[p];
									for ( int t = 0; t < 3; t++, pt++ ){
										tmp = abcd_deg * buf_vec[pt][f1234];
										rawjs[atom][t](bf1, bf2) += tmp * D(bf3, bf4);
										rawjs[atom][t](bf3, bf4) += tmp * D(bf1, bf2);
										if ( kscale > 0 ){
											rawks[atom][t](bf1, bf3) += tmp * D(bf2, bf4);
											rawks[atom][t](bf2, bf4) += tmp * D(bf1, bf3);
											rawks[atom][t](bf1, bf4) += tmp * D(bf2, bf3);
											rawks[atom][t](bf2, bf3) += tmp * D(bf1, bf4);
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
	std::vector<std::vector<EigenMatrix>> gs(natoms);
	for ( std::vector<EigenMatrix>& g : gs )
		g = {EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis), EigenZero(nbasis, nbasis)};
	for ( int iatom = 0; iatom < natoms; iatom++ ) for ( int xyz = 0; xyz < 3; xyz++ ){
		const EigenMatrix j = 0.5  * ( rawjs[iatom][xyz] + rawjs[iatom][xyz].transpose() );
		const EigenMatrix k = 0.25 * ( rawks[iatom][xyz] + rawks[iatom][xyz].transpose() );
		gs[iatom][xyz] = j - 0.5 * kscale * k;
	}
	return gs;
}

EigenMatrix getRepulsion2(
		libint2::BasisSet& obs,
		std::vector<int>& shell2atom,
		std::vector<long int> sqheads,
		short int* shellis, short int* shelljs,
		short int* shellks, short int* shellls,
		EigenMatrix D, double kscale){
	const int natoms = *std::max_element(shell2atom.begin(), shell2atom.end()) + 1;
	const auto shell2bf = obs.shell2bf();
	EigenMatrix hessianj = EigenZero(3 * natoms, 3 * natoms);
	EigenMatrix hessiank = EigenZero(3 * natoms, 3 * natoms);
 	libint2::initialize();
	const int nthreads = sqheads.size() - 1;
	#pragma omp declare reduction(Sum: EigenMatrix: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(Sum: hessianj, hessiank) num_threads(nthreads)
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		const long int nsq = sqheads[ithread + 1] - sqheads[ithread];
		const long int sqhead = sqheads[ithread];
		short int* s1ranger = shellis + sqhead;
		short int* s2ranger = shelljs + sqhead;
		short int* s3ranger = shellks + sqhead;
		short int* s4ranger = shellls + sqhead;
		libint2::Engine engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l(), 2);
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
			if ( !buf_vec[0] ) continue;
			const int atomlist[] = {
				shell2atom[s1],
				shell2atom[s2],
				shell2atom[s3],
				shell2atom[s4]
			};
			for ( short int f1 = 0, f1234 = 0; f1 != n1; f1++ ){
				const short int bf1 = bf1_first + f1;
				for ( short int f2 = 0; f2 != n2; f2++ ){
					const short int bf2 = bf2_first + f2;
					const double ab_deg = (bf1 == bf2) ? 1 : 2;
					for ( short int f3 = 0; f3 != n3; f3++ ){
						const short int bf3 = bf3_first + f3;
						for ( short int f4 = 0; f4 != n4; f4++, f1234++ ){
							const short int bf4 = bf4_first + f4;
							if ( bf2 <= bf1 && bf3 <= bf1 && bf4 <= ((bf1 == bf3) ? bf2 : bf3)){
								const double cd_deg = (bf3 == bf4) ? 1 : 2;
								const double ab_cd_deg = (bf1 == bf3) ? (bf2 == bf4 ? 1 : 2) : 2;
								const double abcd_deg = ab_deg * cd_deg * ab_cd_deg;
								double tmp = 114514;
								int xpert = 1919;
								int ypert = 810;
								for ( int p = 0, ptqs = 0; p < 4; p++ ) for ( int t = 0; t < 3; t++ ){
									xpert = 3 * atomlist[p] + t;
									for ( int q = p; q < 4; q++ ) for ( int  s = ((q==p)?t:0); s < 3; s++, ptqs++ ){
										ypert = 3 * atomlist[q] + s;
										double scale = (xpert==ypert && p!=q) ? 2 : 1;
										tmp = scale * abcd_deg * buf_vec[ptqs][f1234];
										hessianj(xpert, ypert) += tmp * D(bf1, bf2) * D(bf3, bf4);
										if ( kscale > 0 ) hessiank(xpert, ypert) += tmp * ( D(bf1, bf3) * D(bf2, bf4) + D(bf1, bf4) * D(bf2, bf3) );
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
	hessianj *= 2;
	EigenMatrix rawhessian = hessianj - 0.5 * kscale * hessiank;
	return rawhessian + rawhessian.transpose() - (EigenMatrix)rawhessian.diagonal().asDiagonal();
}

Int4C2E::Int4C2E(Mwfn& mwfn, double exx, double threshold){
	this->MWFN = &mwfn;
	this->EXX = exx;
	this->Threshold = threshold;
}

void Int4C2E::getRepulsionDiag(int output){
	if (output>0) std::printf("Calculating diagonal elements of repulsion integrals ... ");
	const auto start = __now__;
	__Make_Basis_Set__(this->MWFN)
	this->RepulsionDiags = ::getRepulsionDiag(obs);
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
}

void Int4C2E::getRepulsionLength(int output){
	if (output>0) std::printf("Calculating numbers of non-equivalent integrals and shell quartets after Cauchy-Schwarz screening ... ");
	const auto start = __now__;
	const long int nbasis = this->MWFN->getNumBasis();
	const long int nshells = this->MWFN->getNumShells();
	__Make_Basis_Set__(this->MWFN)
	assert(std::get<0>(this->RepulsionDiags).size() > 0 && "Diagonal elements of repulsion integrals are missing!");
	std::tie(this->RepulsionLength, this->ShellQuartetLength) = ::getRepulsionLength(obs, std::get<0>(this->RepulsionDiags), this->Threshold);
	if (output>0){
		std::printf("Done in %f s\n", __duration__(start, __now__));
		std::printf(
				"Before screening: %ld integrals and %ld shell quartets\n",
				nbasis * ( nbasis + 1 ) * ( nbasis * ( nbasis + 1 ) / 2 + 1 ) / 4,
				nshells * ( nshells + 1 ) * ( nshells * ( nshells + 1 ) / 2 + 1 ) / 4
		);
		std::printf(
				"After screening: %ld integrals and %ld shell quartets\n",
				this->RepulsionLength, this->ShellQuartetLength
		);
		std::printf(
				"Memory needed for 4c-2e repulsion integrals and their indices: %f GB\n",
				this->RepulsionLength * ( 2. * 4. + 8. ) / 1024. / 1024. / 1024.
		);
	}
}

void Int4C2E::getRepulsionIndices(int output){
	if (output>0) std::printf("Generating indices of non-equivalent integrals and shell quartets after Cauchy-Schwarz screening ... ");
	const auto start = __now__;
	__Make_Basis_Set__(this->MWFN)
	assert(this->ShellIs.empty() && "Shell indices have already been obtained!");
	this->ShellIs.resize(this->ShellQuartetLength);
	this->ShellJs.resize(this->ShellQuartetLength);
	this->ShellKs.resize(this->ShellQuartetLength);
	this->ShellLs.resize(this->ShellQuartetLength);
	::getRepulsionIndices(
			obs, std::get<0>(this->RepulsionDiags), this->Threshold,
			this->ShellIs.data(), this->ShellJs.data(),
			this->ShellKs.data(), this->ShellLs.data()
	);
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
}

void Int4C2E::getThreadPointers(int nthreads, int output){
	if (output>0) std::printf("Arranging for %d threads to calculate 4c-2e repulsion integrals ... ", nthreads);
	const auto start = __now__;
	__Make_Basis_Set__(this->MWFN)
	assert(!this->ShellIs.empty() && "Shell indices are missing!");
	auto [sqheads, bqheads] = ::getThreadPointers( // std::vector<long int>
		obs, ShellQuartetLength, nthreads,
		this->ShellIs.data(), this->ShellJs.data(),
		this->ShellKs.data(), this->ShellLs.data()
	);
	sqheads.push_back(this->ShellQuartetLength);
	this->ShellQuartetHeads = sqheads;
	this->BasisQuartetHeads = bqheads;
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
}

void Int4C2E::CalculateIntegrals(int order, int output){
	if (output>0) std::printf("Calculating 4c-2e repulsion integrals of Order %d ... ", order);
	const auto start = __now__;
	__Make_Basis_Set__(this->MWFN)
	if ( order == 0 ){
		this->BasisIs.resize(this->RepulsionLength);
		this->BasisJs.resize(this->RepulsionLength);
		this->BasisKs.resize(this->RepulsionLength);
		this->BasisLs.resize(this->RepulsionLength);
		this->RepulsionInts.resize(this->RepulsionLength);
		getRepulsion0(
			obs, this->ShellQuartetHeads, this->BasisQuartetHeads,
			this->ShellIs.data(), this->ShellJs.data(),
			this->ShellKs.data(), this->ShellLs.data(),
			this->BasisIs.data(), this->BasisJs.data(),
			this->BasisKs.data(), this->BasisLs.data(),
			this->RepulsionInts.data()
		);
	}
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
}

std::vector<long int> getThreadPointers(long int nitems, int nthreads){
	const long int nitem_fewer = nitems / nthreads;
	const int nfewers = nthreads - nitems + nitem_fewer * nthreads;
	std::vector<long int> heads(nthreads, 0);
	long int n = 0;
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		heads[ithread] = n;
		n += (ithread < nfewers) ? nitem_fewer : nitem_fewer + 1;
	}
	return heads;
}

EigenMatrix Grhf(
		short int* is, short int* js, short int* ks, short int* ls,
		double* ints, long int length,
		EigenMatrix D, double kscale, int nthreads){
	Eigen::setNbThreads(1);
	std::vector<long int> heads = getThreadPointers(length, nthreads);
	EigenMatrix rawJ = EigenZero(D.rows(), D.cols());
	EigenMatrix rawK = EigenZero(D.rows(), D.cols());
	#pragma omp declare reduction(EigenMatrixSum: EigenMatrix: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(EigenMatrixSum: rawJ, rawK) num_threads(nthreads)
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		const long int head = heads[ithread];
		const long int nints = ( (ithread == nthreads - 1) ? length : heads[ithread + 1] ) - head;
		short int* iranger = is + head;
		short int* jranger = js + head;
		short int* kranger = ks + head;
		short int* lranger = ls + head;
		double* repulsionranger = ints + head;
		for ( long int iint = 0; iint < nints; iint++ ){
			const short int i = *(iranger++);
			const short int j = *(jranger++);
			const short int k = *(kranger++);
			const short int l = *(lranger++);
			const double repulsion = *(repulsionranger++);
			rawJ(i, j) += D(k, l) * repulsion;
			rawJ(k, l) += D(i, j) * repulsion;
			if ( kscale > 0. ){
				rawK(i, k) += D(j, l) * repulsion;
				rawK(j, l) += D(i, k) * repulsion;
				rawK(i, l) += D(j, k) * repulsion;
				rawK(j, k) += D(i, l) * repulsion;
			}
		}
	}
	Eigen::setNbThreads(nthreads);
	const EigenMatrix J = 0.5 * ( rawJ + rawJ.transpose() );
	const EigenMatrix K = 0.25 * ( rawK + rawK.transpose() );
	return J - 0.5 * kscale * K;
}

EigenMatrix Int4C2E::ContractInts(EigenMatrix D, int nthreads, int output){ // RHF
	if (output>0) std::printf("Contracting 4c-2e repulsion integrals with 1 matrix ... ");
	const auto start = __now__;
	const EigenMatrix G = Grhf(
		this->BasisIs.data(), this->BasisJs.data(),
		this->BasisKs.data(), this->BasisLs.data(),
		this->RepulsionInts.data(), this->RepulsionLength,
		D, this->EXX, nthreads);
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
	return G;
}

std::vector<EigenMatrix> GhfMultiple(
		short int* is, short int* js, short int* ks, short int* ls,
		double* ints, long int length,
		std::vector<EigenMatrix>& Ds, double kscale, int nthreads){
	std::vector<long int> heads = getThreadPointers(length, nthreads);
	const int nbasis = Ds[0].rows();
	const int nmatrices = Ds.size();
	const int nmatrices_redun = std::ceil((double)nmatrices / 8.) * 8;
	std::vector<std::vector<EigenArray>> Darrays = Matrices2Arrays(Ds, nmatrices_redun);
	std::vector<std::vector<EigenArray>> rawJarrays = MultipleMatrixInitialization(nmatrices_redun, nbasis, nbasis);
	std::vector<std::vector<EigenArray>> rawKarrays = MultipleMatrixInitialization(nmatrices_redun, nbasis, nbasis);
	#pragma omp declare reduction(Sum: std::vector<std::vector<EigenArray>>: MultipleMatrixReduction(omp_out, omp_in)) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(Sum: rawJarrays, rawKarrays) firstprivate(Darrays) num_threads(nthreads)
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		const long int head = heads[ithread];
		const long int nints = ( (ithread == nthreads - 1) ? length : heads[ithread + 1] ) - head;
		short int* iranger = is + head;
		short int* jranger = js + head;
		short int* kranger = ks + head;
		short int* lranger = ls + head;
		double* repulsionranger = ints + head;
		for ( long int iint = 0; iint < nints; iint++ ){
			const short int i = *(iranger++);
			const short int j = *(jranger++);
			const short int k = *(kranger++);
			const short int l = *(lranger++);
			const double repulsion = *(repulsionranger++);
			rawJarrays[i][j] += Darrays[k][l] * repulsion;
			rawJarrays[k][l] += Darrays[i][j] * repulsion;
			if ( kscale > 0. ){
				rawKarrays[i][k] += Darrays[j][l] * repulsion;
				rawKarrays[j][l] += Darrays[i][k] * repulsion;
				rawKarrays[i][l] += Darrays[j][k] * repulsion;
				rawKarrays[j][k] += Darrays[i][l] * repulsion;
			}
		}
	}
	std::vector<EigenMatrix> rawJs = Arrays2Matrices(rawJarrays, nmatrices);
	std::vector<EigenMatrix> rawKs = Arrays2Matrices(rawKarrays, nmatrices);
	std::vector<EigenMatrix> rawGs(nmatrices, EigenZero(nbasis, nbasis));
	for ( int k = 0; k < nmatrices; k++ ){
		const EigenMatrix J = 0.5 *  ( rawJs[k] + rawJs[k].transpose() );
		const EigenMatrix K = 0.25 * ( rawKs[k] + rawKs[k].transpose() );
		rawGs[k] = J - 0.5 * kscale * K;
	}
	return rawGs;
}

std::vector<EigenMatrix> Int4C2E::ContractInts(std::vector<EigenMatrix>& Ds, int nthreads, int output){ // Multiple contraction
	const int nmatrices = (int)Ds.size();
	if (output>0) std::printf("Contracting 4c-2e repulsion integrals with %d matrices ... ", nmatrices);
	const auto start = __now__;
	__Make_Basis_Set__(this->MWFN)
	const std::vector<EigenMatrix> Gs = GhfMultiple(
		this->BasisIs.data(), this->BasisJs.data(),
		this->BasisKs.data(), this->BasisLs.data(),
		this->RepulsionInts.data(), this->RepulsionLength,
		Ds, this->EXX, nthreads);
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
	return Gs;
}

std::tuple<EigenMatrix, EigenMatrix, EigenMatrix, EigenMatrix> Guhf(
		short int* is, short int* js, short int* ks, short int* ls,
		double* ints, long int length,
		EigenMatrix Da, EigenMatrix Db, double kscale, int nthreads){
	Eigen::setNbThreads(1);
	std::vector<long int> heads = getThreadPointers(length, nthreads);
	EigenMatrix rawJa = EigenZero(Da.rows(), Da.cols());
	EigenMatrix rawJb = EigenZero(Da.rows(), Da.cols());
	EigenMatrix rawKa = EigenZero(Da.rows(), Da.cols());
	EigenMatrix rawKb = EigenZero(Da.rows(), Da.cols());
	#pragma omp declare reduction(EigenMatrixSum: EigenMatrix: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(EigenMatrixSum: rawJa, rawJb, rawKa, rawKb) num_threads(nthreads)
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		const long int head = heads[ithread];
		const long int nints = ( (ithread == nthreads - 1) ? length : heads[ithread + 1] ) - head;
		short int* iranger = is + head;
		short int* jranger = js + head;
		short int* kranger = ks + head;
		short int* lranger = ls + head;
		double* repulsionranger = ints + head;
		for ( long int iint = 0; iint < nints; iint++ ){
			const short int i = *(iranger++);
			const short int j = *(jranger++);
			const short int k = *(kranger++);
			const short int l = *(lranger++);
			const double repulsion = *(repulsionranger++);
			rawJa(i, j) += Da(k, l) * repulsion;
			rawJa(k, l) += Da(i, j) * repulsion;
			rawJb(i, j) += Db(k, l) * repulsion;
			rawJb(k, l) += Db(i, j) * repulsion;
			if ( kscale > 0. ){
				rawKa(i, k) += Da(j, l) * repulsion;
				rawKa(j, l) += Da(i, k) * repulsion;
				rawKa(i, l) += Da(j, k) * repulsion;
				rawKa(j, k) += Da(i, l) * repulsion;
				rawKb(i, k) += Db(j, l) * repulsion;
				rawKb(j, l) += Db(i, k) * repulsion;
				rawKb(i, l) += Db(j, k) * repulsion;
				rawKb(j, k) += Db(i, l) * repulsion;
			}
		}
	}
	Eigen::setNbThreads(nthreads);
	const EigenMatrix Ja = 0.25 * ( rawJa + rawJa.transpose() );
	const EigenMatrix Jb = 0.25 * ( rawJb + rawJb.transpose() );
	const EigenMatrix Ka = 0.125 * ( rawKa + rawKa.transpose() );
	const EigenMatrix Kb = 0.125 * ( rawKb + rawKb.transpose() );
	return std::make_tuple(Ja, Jb, Ka, Kb);
}

std::tuple<EigenMatrix, EigenMatrix> Int4C2E::ContractInts(EigenMatrix Da, EigenMatrix Db, int nthreads, int output){ // UHF
	if (output>0) std::printf("Contracting 4c-2e repulsion integrals with 1 matrix ... ");
	const auto start = __now__;
	__Make_Basis_Set__(this->MWFN)
	auto [Ja, Jb, Ka, Kb] = Guhf(
		this->BasisIs.data(), this->BasisJs.data(),
		this->BasisKs.data(), this->BasisLs.data(),
		this->RepulsionInts.data(), this->RepulsionLength,
		Da, Db, this->EXX, nthreads);
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
	return std::make_tuple(Ja + Jb - Ka, Ja + Jb - Kb);
}

std::tuple<EigenMatrix, EigenMatrix> Guhf(
		short int* is, short int* js, short int* ks, short int* ls,
		double* ints, long int length,
		EigenMatrix Da, double kscale, int nthreads){
	Eigen::setNbThreads(1);
	std::vector<long int> heads = getThreadPointers(length, nthreads);
	EigenMatrix rawJa = EigenZero(Da.rows(), Da.cols());
	EigenMatrix rawJb = EigenZero(Da.rows(), Da.cols());
	EigenMatrix rawKa = EigenZero(Da.rows(), Da.cols());
	EigenMatrix rawKb = EigenZero(Da.rows(), Da.cols());
	#pragma omp declare reduction(EigenMatrixSum: EigenMatrix: omp_out += omp_in) initializer(omp_priv = omp_orig)
	#pragma omp parallel for reduction(EigenMatrixSum: rawJa, rawJb, rawKa, rawKb) num_threads(nthreads)
	for ( int ithread = 0; ithread < nthreads; ithread++ ){
		const long int head = heads[ithread];
		const long int nints = ( (ithread == nthreads - 1) ? length : heads[ithread + 1] ) - head;
		short int* iranger = is + head;
		short int* jranger = js + head;
		short int* kranger = ks + head;
		short int* lranger = ls + head;
		double* repulsionranger = ints + head;
		for ( long int iint = 0; iint < nints; iint++ ){
			const short int i = *(iranger++);
			const short int j = *(jranger++);
			const short int k = *(kranger++);
			const short int l = *(lranger++);
			const double repulsion = *(repulsionranger++);
			rawJa(i, j) += Da(k, l) * repulsion;
			rawJa(k, l) += Da(i, j) * repulsion;
			if ( kscale > 0. ){
				rawKa(i, k) += Da(j, l) * repulsion;
				rawKa(j, l) += Da(i, k) * repulsion;
				rawKa(i, l) += Da(j, k) * repulsion;
				rawKa(j, k) += Da(i, l) * repulsion;
			}
		}
	}
	Eigen::setNbThreads(nthreads);
	const EigenMatrix Ja = 0.25 * ( rawJa + rawJa.transpose() );
	const EigenMatrix Ka = 0.125 * ( rawKa + rawKa.transpose() );
	return std::make_tuple(Ja, Ka);
}

std::tuple<EigenMatrix, EigenMatrix> Int4C2E::ContractInts2(EigenMatrix Da, int nthreads, int output){ // UHF
	if (output>0) std::printf("Contracting 4c-2e repulsion integrals with 1 matrix ... ");
	const auto start = __now__;
	__Make_Basis_Set__(this->MWFN)
	auto [Ja, Ka] = Guhf(
		this->BasisIs.data(), this->BasisJs.data(),
		this->BasisKs.data(), this->BasisLs.data(),
		this->RepulsionInts.data(), this->RepulsionLength,
		Da, this->EXX, nthreads);
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
	return std::make_tuple(Ja, Ka);
}

std::vector<double> Int4C2E::ContractGrads(EigenMatrix D1, EigenMatrix D2, int output){
	std::vector<EigenMatrix> D1s = {D1};
	return this->ContractGrads(D1s, D2, output)[0];
}

std::vector<std::vector<double>> Int4C2E::ContractGrads(std::vector<EigenMatrix>& D1s, EigenMatrix D2, int output){
	const int natoms = this->MWFN->getNumCenters();
	const std::vector<EigenMatrix> GD2 = this->ContractGrads(D2, output);
	const int nmatrices = (int)D1s.size();
	if (output>0) std::printf("Contracting 4c-2e repulsion integral nuclear gradient with %d matrix from the left and 1 matrix from the right ... ", nmatrices);
	const auto start = __now__;
	std::vector<std::vector<double>> D1GD2(nmatrices, std::vector<double>(3 * natoms));
	for ( int imat = 0; imat < nmatrices; imat++ ) for ( int j = 0; j < 3 * natoms; j++ ){
		D1GD2[imat][j] = D1s[imat].cwiseProduct(GD2[j]).sum();
	}
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
	return D1GD2;
}

std::vector<EigenMatrix> Int4C2E::ContractGrads(EigenMatrix D, int output){
	if (output>0) std::printf("Contracting 4c-2e repulsion integral nuclear gradient with 1 matrix ... ");
	const auto start = __now__;
	const int natoms = this->MWFN->getNumCenters();
	for ( auto& [key, value] : this->GradCache ) if ( key.isApprox(D) ){
		if (output>0) std::printf("Found in cache -> Done in %f s\n", __duration__(start, __now__));
		return value;
	}
	__Make_Basis_Set__(this->MWFN)
	const std::vector<std::vector<EigenMatrix>> GDs_ = getRepulsion1(
			obs,
			shell2atom,
			this->ShellQuartetHeads,
			this->ShellIs.data(), this->ShellJs.data(),
			this->ShellKs.data(), this->ShellLs.data(),
			D, this->EXX
	);
	std::vector<EigenMatrix> GDs; GDs.reserve(3 * natoms);
	for ( int iatom = 0; iatom < natoms; iatom++ ) for ( int xyz = 0; xyz < 3; xyz++ ){
		GDs.push_back(GDs_[iatom][xyz]);
	}
	this->GradCache.push_back(std::make_tuple(D, GDs));
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
	return GDs;
}

std::vector<std::vector<double>> Int4C2E::ContractHesss(EigenMatrix D1, EigenMatrix D2, int output){
	EigenMatrix D = D1; D = D2;
	if (output>0) std::printf("Contracting 4c-2e repulsion integral nuclear hessian with 1 matrix ... ");
	const auto start = __now__;
	const int natoms = this->MWFN->getNumCenters();
	__Make_Basis_Set__(this->MWFN)
	const EigenMatrix GDs_ = getRepulsion2(
		obs,
		shell2atom,
		this->ShellQuartetHeads,
		this->ShellIs.data(), this->ShellJs.data(),
		this->ShellKs.data(), this->ShellLs.data(),
		D, this->EXX
	);
	std::vector<std::vector<double>> GDs(3 * natoms, std::vector<double>(3 * natoms));
	for ( int i = 0; i < 3 * natoms; i++ ) for ( int j = 0; j < 3 * natoms; j++ )
		GDs[i][j] = GDs_(i, j);
	if (output>0) std::printf("Done in %f s\n", __duration__(start, __now__));
	return GDs;
}
