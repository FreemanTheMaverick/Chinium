#include <libint2.hpp>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

using namespace libint2;
using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
Matrix G(BasisSet obs,const int nbasis,Matrix D){
	Matrix g=Matrix::Zero(nbasis,nbasis);
	Engine g_engine(Operator::coulomb,obs.max_nprim(),obs.max_l());
	const auto& buf_vec=g_engine.results();
	auto shell2bf=obs.shell2bf();
	for (int s1=0;s1!=obs.size();s1++){
		int bf1_first=shell2bf[s1];
		int n1=obs[s1].size();
		for (int s2=0;s2<=s1;s2++){
			int bf2_first=shell2bf[s2];
			int n2=obs[s2].size();
			for (int s3=0;s3<=s1;s3++){
				int bf3_first=shell2bf[s3];
				int n3=obs[s3].size();
				for (int s4=0;s4<=((s1==s3)?s2:s3);s4++){
					int bf4_first=shell2bf[s4];
					int n4=obs[s4].size();
					int s12_deg=(s1==s2)?1.0:2.0;
					int s34_deg=(s3==s4)?1.0:2.0;
					int s12_34_deg=(s1==s3)?(s2==s4?1.0:2.0):2.0;
					int s1234_deg=s12_deg*s34_deg*s12_34_deg;
					g_engine.compute(obs[s1],obs[s2],obs[s3],obs[s4]);
					const auto buf_1234=buf_vec[0];
					if (buf_1234==nullptr){
						continue;
					}
					for (int f1=0,f1234=0;f1!=n1;f1++){
						const int bf1=f1+bf1_first;
						for (int f2=0;f2!=n2;f2++){
							const int bf2=f2+bf2_first;
							for (int f3=0;f3!=n3;f3++){
								const int bf3=f3+bf3_first;
								for (int f4=0;f4!=n4;f4++,++f1234){
									const int bf4=f4+bf4_first;
									const double value=buf_1234[f1234];
									g(bf1,bf2)+=D(bf3,bf4)*value*s1234_deg;
                  							g(bf3,bf4)+=D(bf1,bf2)*value*s1234_deg;
                  							g(bf1,bf3)-=0.25*D(bf2,bf4)*value*s1234_deg;
                  							g(bf2,bf4)-=0.25*D(bf1,bf3)*value*s1234_deg;
                  							g(bf1,bf4)-=0.25*D(bf2,bf3)*value*s1234_deg;
                  							g(bf2,bf3)-=0.25*D(bf1,bf4)*value*s1234_deg;
								}
							}
						}
					}
				}
			}
		}
	}
	Matrix gt=g.transpose();
	g=0.5*(g+gt);
	return g;
}

