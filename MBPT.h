#include <libint2.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>

using namespace libint2;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

Matrix DuplicateRow(Matrix matrix){
	int nrows=matrix.rows();
	int ncols=matrix.cols();
	Matrix row[nrows];
	for (int i=0;i<nrows;i++){
		row[i]=matrix.row(i);
	}
	Matrix newmatrix(2*nrows,ncols);
	for (int i=0;i<nrows;i++){
		newmatrix.row(2*i)=row[i];
		newmatrix.row(2*i+1)=row[i];
	}
	return newmatrix;
}

Matrix DuplicateCol(Matrix matrix){
	int nrows=matrix.rows();
	int ncols=matrix.cols();
	Matrix col[ncols];
	for (int i=0;i<ncols;i++){
		col[i]=matrix.col(i);
	}
	Matrix newmatrix(nrows,2*ncols);
	for (int i=0;i<ncols;i++){
		newmatrix.col(2*i)=col[i];
		newmatrix.col(2*i+1)=col[i];
	}
	return newmatrix;
}


class MP2Job{
	public:
		vector<Atom> Atoms;
		string BasisName;
		Matrix CoefficientMatrix;
		Matrix OrbitalEnergies;
		double CorrelationEnergy;
		void setXYZ(std::string xyzfilename);
		void setBasisSet(std::string basisname);
		void setCoefficientMatrix(Matrix coefficientmatrix);
		void setOrbitalEnergies(Matrix orbitalenergies);
		void Compute();
};

void MP2Job::setXYZ(std::string xyzfilename){
	ifstream input_file(xyzfilename);
	Atoms=read_dotxyz(input_file);
}

void MP2Job::setBasisSet(std::string basisname){
	BasisName=basisname;
}

void MP2Job::setCoefficientMatrix(Matrix coefficientmatrix){
	CoefficientMatrix=coefficientmatrix;
}

void MP2Job::setOrbitalEnergies(Matrix orbitalenergies){
	OrbitalEnergies=orbitalenergies;
}

void MP2Job::Compute(){
	BasisSet obs(BasisName,Atoms);
	int nbasis=nBasis(obs);
	int nelectron=nElectron(Atoms);
	CorrelationEnergy=0;
	for (int p=0;p<nelectron-1;p++){
		for (int q=p+1;q<nelectron;q++){
			for (int r=nelectron;r<2*nbasis-1;r++){
				for (int s=r+1;s<2*nbasis;s++){
					double sqrtnumerator=0;
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
													Matrix c=CoefficientMatrix;
		sqrtnumerator+=c(bf1,p)*c(bf3,q)*(c(bf2,s)*c(bf4,r)*(p%2==s%2)*(q%2==r%2)-c(bf2,r)*c(bf4,s)*(p%2==r%2)*(q%2==s%2))*value
			      +c(bf2,p)*c(bf3,q)*(c(bf1,s)*c(bf4,r)*(p%2==s%2)*(q%2==r%2)-c(bf1,r)*c(bf4,s)*(p%2==r%2)*(q%2==s%2))*value*(s12_deg==2)
			      +c(bf1,p)*c(bf4,q)*(c(bf2,s)*c(bf3,r)*(p%2==s%2)*(q%2==r%2)-c(bf2,r)*c(bf3,s)*(p%2==r%2)*(q%2==s%2))*value*(s34_deg==2)
			      +c(bf2,p)*c(bf4,q)*(c(bf1,s)*c(bf3,r)*(p%2==s%2)*(q%2==r%2)-c(bf1,r)*c(bf3,s)*(p%2==r%2)*(q%2==s%2))*value*(s12_deg==2&&s34_deg==2)
			      +c(bf3,p)*c(bf1,q)*(c(bf4,s)*c(bf2,r)*(p%2==s%2)*(q%2==r%2)-c(bf4,r)*c(bf2,s)*(p%2==r%2)*(q%2==s%2))*value*(s12_34_deg==2)
			      +c(bf4,p)*c(bf1,q)*(c(bf3,s)*c(bf2,r)*(p%2==s%2)*(q%2==r%2)-c(bf3,r)*c(bf2,s)*(p%2==r%2)*(q%2==s%2))*value*(s12_deg==2&&s12_34_deg==2)
			      +c(bf3,p)*c(bf2,q)*(c(bf4,s)*c(bf1,r)*(p%2==s%2)*(q%2==r%2)-c(bf4,r)*c(bf1,s)*(p%2==r%2)*(q%2==s%2))*value*(s34_deg==2&&s12_34_deg==2)
			      +c(bf4,p)*c(bf2,q)*(c(bf3,s)*c(bf1,r)*(p%2==s%2)*(q%2==r%2)-c(bf3,r)*c(bf1,s)*(p%2==r%2)*(q%2==s%2))*value*(s12_deg==2&&s34_deg==2&&s12_34_deg==2);
												}
											}
										}
									}
								}
							}
						}
					}
					CorrelationEnergy-=sqrtnumerator*sqrtnumerator/(OrbitalEnergies(r,0)+OrbitalEnergies(s,0)-OrbitalEnergies(p,0)-OrbitalEnergies(q,0));
				}
			}
		}
	}
}



