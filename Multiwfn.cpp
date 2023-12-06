#include <Eigen/Dense>
#include <sstream>
#include <fstream>
#include "Aliases.h"
#include "Multiwfn.h"
#include <iostream>

#define __Read_Array_Head__\
	int k=0;\
	while (std::getline(file,line) && line.length() && k<total){\
		std::stringstream ss_(line);\
		while (ss_>>word && k<total){

#define __Read_Array_Tail__\
			k++;\
		}\
	}

#define __Load_Matrix__(mat)\
	int irow=0;\
	int jcol=0;\
	int total=(lower?(1+ncols)*ncols/2:nrows*ncols);\
	__Read_Array_Head__\
		if (lower){\
			mat(irow,jcol)=mat(jcol,irow)=std::stof(word);\
			if (irow==jcol){\
				irow++;\
				jcol=0;\
			}else jcol++;\
		}else{\
			mat(irow,jcol)=std::stof(word);\
			if (jcol+1==ncols){\
				irow++;\
				jcol=0;\
			}else jcol++;\
		}\
	__Read_Array_Tail__

Mwfn::Mwfn(std::string filename,const bool output){
	std::ifstream file(filename.c_str());
	if (!file.good()) return;
	if (output)
		std::cout<<"Reading existent Multiwfn file "<<filename<<" ..."<<std::endl;
	std::string line,word;
	int tmp_int=114514;
	while (std::getline(file,line)){
		std::stringstream ss(line);
		ss>>word;

		// Field 3
		if (word.compare("Nbasis=")==0){
			ss>>word;
			this->Nbasis=std::stoi(word);
			this->Type.resize(this->Nbasis);
			this->Energy.resize(this->Nbasis);
			this->Occ.resize(this->Nbasis);
			this->Sym.resize(this->Nbasis);
			this->Coeff.resize(this->Nbasis,this->Nbasis);
			continue;
		}else if (word.compare("Nindbasis")==0){
			ss>>word;
			this->Nindbasis=std::stoi(word);
			continue;
		}

		// Field 4
		if (word.compare("Index=")==0){
			ss>>word;
			tmp_int=std::stoi(word)-1;
			continue;
		}else if (word.compare("Type=")==0){
			ss>>word;
			this->Type[tmp_int]=std::stoi(word);
			continue;
		}else if (word.compare("Energy=")==0){
			ss>>word;
			this->Energy[tmp_int]=std::stof(word);
			continue;
		}else if (word.compare("Occ=")==0){
			ss>>word;
			try{ // In FT theory, some occupation numbers can be of dimension of 1E-50 or less. These small values cannot be stored as doubles and are likely to cause "out_of_range" error.
				this->Occ[tmp_int]=std::stof(word);
			}catch (const std::out_of_range& e){
				this->Occ[tmp_int]=0;
			}
			continue;
		}else if (word.compare("Sym=")==0){
			ss>>word;
			this->Sym[tmp_int]=word;
			continue;
		}else if (word.compare("$Coeff")==0){
			int ibasis=0;
			int total=this->Nbasis;
			__Read_Array_Head__
				this->Coeff(ibasis,tmp_int)=std::stof(word);
				ibasis++;
			__Read_Array_Tail__
			continue;
		}

		// Field 5
		if (word.compare("$Total")==0){
			ss>>word; // "density"
			ss>>word; // "matrix,"
			ss>>word; // "dim="
			ss>>word; // nrows
			const int nrows=std::stoi(word);
			ss>>word; // ncols
			const int ncols=std::stoi(word);
			ss>>word; // "lower="
			ss>>word; // lower
			const int lower=std::stoi(word);
			this->Total_density_matrix=EigenZero(nrows,ncols);
			__Load_Matrix__(this->Total_density_matrix)
			continue;
		}else if (word.compare("$Overlap")==0){
			ss>>word; // "matrix,"
			ss>>word; // "dim="
			ss>>word; // nrows
			const int nrows=std::stoi(word);
			ss>>word; // ncols
			const int ncols=std::stoi(word);
			ss>>word; // "lower="
			ss>>word; // lower
			const int lower=std::stoi(word);
			this->Overlap_matrix=EigenZero(nrows,ncols);
			__Load_Matrix__(this->Overlap_matrix)
			continue;
		}

	}
}


void PrintMatrix(std::FILE * file,EigenMatrix matrix,bool lower){
	for (int i=0;i<matrix.rows();i++){
		for (int j=0;j<(lower?i+1:matrix.cols());j++)
			std::fprintf(file," %E",matrix(i,j));
		std::fprintf(file,"\n");
	}
	std::fprintf(file,"\n");
}

void Mwfn::Save(std::string filename,const bool output){
	std::FILE * file=std::fopen(filename.c_str(),"w");
	if (output) std::cout<<"Exporting wavefunction information to "<<filename<<" ..."<<std::endl;

	// Field 3
	if (this->Nbasis)
		std::fprintf(file,"Nbasis= %d\n",this->Nbasis);
	else if (this->Nindbasis)
		std::fprintf(file,"Nindbasis= %d\n",this->Nindbasis);
	std::fprintf(file,"\n");

	// Field 4
	if (this->Energy.size()){
		for (int i=0;i<this->Nbasis;i++){
			std::fprintf(file,"Index= %9d\n",i+1);
			std::fprintf(file,"Type= %d\n",this->Type[i]);
			std::fprintf(file,"Energy= %E\n",this->Energy[i]);
			std::fprintf(file,"Occ= %E\n",this->Occ[i]);
			std::fprintf(file,"Sym= %s\n",this->Sym[i].c_str());
			std::fprintf(file,"$Coeff\n");
			for (int j=0;j<this->Nbasis;j++)
				std::fprintf(file," %E",this->Coeff(j,i));
			std::fprintf(file,"\n\n");
		}
	}
	std::fprintf(file,"\n");

	// Field 5
	if (this->Total_density_matrix.cols()){
		std::fprintf(file,"$Total density matrix, dim= %d %d lower= 1\n",this->Nbasis,this->Nbasis);
		PrintMatrix(file,this->Total_density_matrix,1);
	}
	if (this->Hamiltonian_matrix.cols()){
		std::fprintf(file,"$1-e Hamiltonian matrix, dim= %d %d lower= 1\n",this->Nbasis,this->Nbasis);
		PrintMatrix(file,this->Hamiltonian_matrix,1);
	}
	if (this->Overlap_matrix.cols()){
		std::fprintf(file,"$Overlap matrix, dim= %d %d lower= 1\n",this->Nbasis,this->Nbasis);
		PrintMatrix(file,this->Overlap_matrix,1);
	}
	if (this->Kinetic_energy_matrix.cols()){
		std::fprintf(file,"$Kinetic energy matrix, dim= %d %d lower= 1\n",this->Nbasis,this->Nbasis);
		PrintMatrix(file,this->Kinetic_energy_matrix,1);
	}
	if (this->Potential_energy_matrix.cols()){
		std::fprintf(file,"$Potential energy matrix, dim= %d %d lower= 1\n",this->Nbasis,this->Nbasis);
		PrintMatrix(file,this->Potential_energy_matrix,1);
	}

	std::fclose(file);
}




