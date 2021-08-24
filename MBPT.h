#include <libint2.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>

using namespace libint2;

typedef Eigen::Tensor<double,4,Eigen::RowMajor> Tensor;
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
		//vector<Atom> Atoms;
		//string BasisName;
		//Matrix CoefficientMatrix;
		Matrix OrbitalEnergies;
		Tensor MO2e;
		int nElectron;
		double CorrelationEnergy;
		string ShellType;
		//void setBasisSet(std::string basisname);
		//void setCoefficientMatrix(Matrix coefficientmatrix);
		void setOrbitalEnergies(Matrix orbitalenergies);
		void setMO2e(Tensor mo2e);
		void setnElectron(int nelectron);
		void setShellType(string shelltype);
		//void Compute(Tensor two_e);
		void Compute();
};

/*
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
*/
void MP2Job::setOrbitalEnergies(Matrix orbitalenergies){
	OrbitalEnergies=orbitalenergies;
}

void MP2Job::setMO2e(Tensor mo2e){
	MO2e=mo2e;
}

void MP2Job::setnElectron(int nelectron){
	nElectron=nelectron;
}

void MP2Job::setShellType(string shelltype){
	ShellType=shelltype;
}

void MP2Job::Compute(){
	if (ShellType=="Restricted"){
		int nbasis=OrbitalEnergies.rows();
		CorrelationEnergy=0;
		for (int a=0;a<nElectron/2;a++){
			for (int b=0;b<nElectron/2;b++){
				for (int r=nElectron/2;r<nbasis;r++){
					for (int s=nElectron/2;s<nbasis;s++){
						double fuck=MO2e(a,r,b,s);
						double shit=MO2e(a,s,b,r);
						CorrelationEnergy=CorrelationEnergy+(fuck*(2*fuck-shit))/(OrbitalEnergies(a)+OrbitalEnergies(b)-OrbitalEnergies(r)-OrbitalEnergies(s));
					}
				}
			}
		}
	}
	else if (ShellType=="Unrestricted"){
		int nbasis=OrbitalEnergies.rows();
		CorrelationEnergy=0;
		for (int a=0;a<nElectron;a++){
			for (int b=0;b<nElectron;b++){
				for (int r=nElectron;r<nbasis;r++){
					for (int s=nElectron;s<nbasis;s++){
						double fuck=MO2e(a,r,b,s);
						double shit=MO2e(a,s,b,r);
						CorrelationEnergy=CorrelationEnergy+(fuck-shit)*(fuck-shit)/(OrbitalEnergies(a)+OrbitalEnergies(b)-OrbitalEnergies(r)-OrbitalEnergies(s))/4;
					}
				}
			}
		}
	}
}



/*
void MP2Job::Compute(Tensor two_e){
	BasisSet obs(BasisName,Atoms);
	int nbasis=nBasis(obs);
	int nelectron=nElectron(Atoms);
	CorrelationEnergy=0;
	for (int p=0;p<nelectron-1;p++){
		for (int q=p+1;q<nelectron;q++){
			for (int r=nelectron;r<2*nbasis-1;r++){
				for (int s=r+1;s<2*nbasis;s++){
					double sqrtnumerator=0;
					for (int a=0;a<nbasis;a++){
						for (int b=0;b<=a;b++){
							for (int c=0;c<=a;c++){
								for (int d=0;d<=((a==c)?b:c);d++){
									int s12_deg=(a==b)?1.0:2.0;
									int s34_deg=(c==d)?1.0:2.0;
									int s12_34_deg=(a==c)?(b==d?1.0:2.0):2.0;
									int s1234_deg=s12_deg*s34_deg*s12_34_deg;
									double value=two_e(a,b,c,d);
									Matrix cm=CoefficientMatrix;
									sqrtnumerator+=cm(a,p)*cm(c,q)*(cm(b,s)*cm(d,r)*(p%2==s%2)*(q%2==r%2)-cm(b,r)*cm(d,s)*(p%2==r%2)*(q%2==s%2))*value;
			      						if (s12_deg==2) sqrtnumerator+=cm(b,p)*cm(c,q)*(cm(a,s)*cm(d,r)*(p%2==s%2)*(q%2==r%2)-cm(a,r)*cm(d,s)*(p%2==r%2)*(q%2==s%2))*value;
			      						if (s34_deg==2) sqrtnumerator+=cm(a,p)*cm(d,q)*(cm(b,s)*cm(c,r)*(p%2==s%2)*(q%2==r%2)-cm(b,r)*cm(c,s)*(p%2==r%2)*(q%2==s%2))*value;
									if (s12_deg==2&&s34_deg==2) sqrtnumerator+=cm(b,p)*cm(d,q)*(cm(a,s)*cm(c,r)*(p%2==s%2)*(q%2==r%2)-cm(a,r)*cm(c,s)*(p%2==r%2)*(q%2==s%2))*value;
									if (s12_34_deg==2) sqrtnumerator+=cm(c,p)*cm(a,q)*(cm(d,s)*cm(b,r)*(p%2==s%2)*(q%2==r%2)-cm(d,r)*cm(b,s)*(p%2==r%2)*(q%2==s%2))*value;
									if (s34_deg==2&&s12_34_deg==2) sqrtnumerator+=cm(d,p)*cm(a,q)*(cm(c,s)*cm(b,r)*(p%2==s%2)*(q%2==r%2)-cm(c,r)*cm(b,s)*(p%2==r%2)*(q%2==s%2))*value;
									if (s12_deg==2&&s12_34_deg==2) sqrtnumerator+=cm(c,p)*cm(b,q)*(cm(d,s)*cm(a,r)*(p%2==s%2)*(q%2==r%2)-cm(d,r)*cm(a,s)*(p%2==r%2)*(q%2==s%2))*value;
									if (s12_deg==2&&s34_deg==2&&s12_34_deg==2) sqrtnumerator+=cm(d,p)*cm(b,q)*(cm(c,s)*cm(a,r)*(p%2==s%2)*(q%2==r%2)-cm(c,r)*cm(a,s)*(p%2==r%2)*(q%2==s%2))*value;
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
*/
/*
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
						for (int s2=0;s2!=obs.size();s2++){
						//for (int s2=0;s2<=s1;s2++){
							int bf2_first=shell2bf[s2];
							int n2=obs[s2].size();
							for (int s3=0;s3!=obs.size();s3++){
							//for (int s3=0;s3<=s1;s3++){
								int bf3_first=shell2bf[s3];
								int n3=obs[s3].size();
								for (int s4=0;s4!=obs.size();s4++){
								//for (int s4=0;s4<=((s1==s3)?s2:s3);s4++){
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
													//if (f2>f1||f3>f1||f4>((f1==f3)?f2:f3)) continue;
													const double value=buf_1234[f1234];
													//std::cout<<bf1<<" "<<bf2<<" "<<bf3<<" "<<bf4<<" "<<value<<std::endl;
													Matrix c=CoefficientMatrix;
													//int s12_deg=(f1==f2)?1.0:2.0;
													//int s34_deg=(f3==f4)?1.0:2.0;
													//int s12_34_deg=(f1==f3)?(f2==f4?1.0:2.0):2.0;
													//int s1234_deg=s12_deg*s34_deg*s12_34_deg;
		sqrtnumerator+=c(bf1,p)*c(bf3,q)*(c(bf2,s)*c(bf4,r)*(p%2==s%2)*(q%2==r%2)-c(bf2,r)*c(bf4,s)*(p%2==r%2)*(q%2==s%2))*value;
			      //+c(bf2,p)*c(bf3,q)*(c(bf1,s)*c(bf4,r)*(p%2==s%2)*(q%2==r%2)-c(bf1,r)*c(bf4,s)*(p%2==r%2)*(q%2==s%2))*value*(s12_deg==2)
			      //+c(bf1,p)*c(bf4,q)*(c(bf2,s)*c(bf3,r)*(p%2==s%2)*(q%2==r%2)-c(bf2,r)*c(bf3,s)*(p%2==r%2)*(q%2==s%2))*value*(s34_deg==2)
			      //+c(bf2,p)*c(bf4,q)*(c(bf1,s)*c(bf3,r)*(p%2==s%2)*(q%2==r%2)-c(bf1,r)*c(bf3,s)*(p%2==r%2)*(q%2==s%2))*value*(s12_deg==2&&s34_deg==2)
			      //+c(bf3,p)*c(bf1,q)*(c(bf4,s)*c(bf2,r)*(p%2==s%2)*(q%2==r%2)-c(bf4,r)*c(bf2,s)*(p%2==r%2)*(q%2==s%2))*value*(s12_34_deg==2)
			      //+c(bf4,p)*c(bf1,q)*(c(bf3,s)*c(bf2,r)*(p%2==s%2)*(q%2==r%2)-c(bf3,r)*c(bf2,s)*(p%2==r%2)*(q%2==s%2))*value*(s12_deg==2&&s12_34_deg==2)
			      //+c(bf3,p)*c(bf2,q)*(c(bf4,s)*c(bf1,r)*(p%2==s%2)*(q%2==r%2)-c(bf4,r)*c(bf1,s)*(p%2==r%2)*(q%2==s%2))*value*(s34_deg==2&&s12_34_deg==2)
			      //+c(bf4,p)*c(bf2,q)*(c(bf3,s)*c(bf1,r)*(p%2==s%2)*(q%2==r%2)-c(bf3,r)*c(bf1,s)*(p%2==r%2)*(q%2==s%2))*value*(s12_deg==2&&s34_deg==2&&s12_34_deg==2);
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
*/


