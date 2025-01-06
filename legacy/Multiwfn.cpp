#include <Eigen/Dense>
#include <cmath>
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
			mat(irow,jcol)=mat(jcol,irow)=SafeStof(word);\
			if (irow==jcol){\
				irow++;\
				jcol=0;\
			}else jcol++;\
		}else{\
			mat(irow,jcol)=SafeStof(word);\
			if (jcol+1==ncols){\
				irow++;\
				jcol=0;\
			}else jcol++;\
		}\
	__Read_Array_Tail__

EigenMatrix MwfnMatrixTransform(std::vector<int> shelltypes){ // Chinium orders basis functions in the order like P-1, P0, P+1 and D-2, D-1, D0, D+1, D+2, while .mwfn does like Px, Py, Pz and D0, D+1, D-1, D+2, D-2. This function is used to transform matrices between two forms. 
	EigenMatrix SPDFGHI[7];
	SPDFGHI[0]=EigenZero(1,1);SPDFGHI[0]<<1;
	SPDFGHI[1]=EigenZero(3,3);SPDFGHI[1]<<
		0,0,1,
		1,0,0,
		0,1,0;
	SPDFGHI[2]=EigenZero(5,5);SPDFGHI[2]<<
		0,0,1,0,0,
		0,0,0,1,0,
		0,1,0,0,0,
		0,0,0,0,1,
		1,0,0,0,0;
	SPDFGHI[3]=EigenZero(7,7);SPDFGHI[3]<<
		0,0,0,1,0,0,0,
		0,0,0,0,1,0,0,
		0,0,1,0,0,0,0,
		0,0,0,0,0,1,0,
		0,1,0,0,0,0,0,
		0,0,0,0,0,0,1,
		1,0,0,0,0,0,0;
	SPDFGHI[4]=EigenZero(9,9);SPDFGHI[4]<<
		0,0,0,0,1,0,0,0,0,
		0,0,0,0,0,1,0,0,0,
		0,0,0,1,0,0,0,0,0,
		0,0,0,0,0,0,1,0,0,
		0,0,1,0,0,0,0,0,0,
		0,0,0,0,0,0,0,1,0,
		0,1,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,1,
		1,0,0,0,0,0,0,0,0;
	SPDFGHI[5]=EigenZero(11,11);SPDFGHI[5]<<
		0,0,0,0,0,1,0,0,0,0,0,
		0,0,0,0,0,0,1,0,0,0,0,
		0,0,0,0,1,0,0,0,0,0,0,
		0,0,0,0,0,0,0,1,0,0,0,
		0,0,0,1,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,1,0,0,
		0,0,1,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,1,0,
		0,1,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,1,
		1,0,0,0,0,0,0,0,0,0,0;
	SPDFGHI[6]=EigenZero(13,13);SPDFGHI[6]<<
		0,0,0,0,0,0,1,0,0,0,0,0,0,
		0,0,0,0,0,0,0,1,0,0,0,0,0,
		0,0,0,0,0,1,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,1,0,0,0,0,
		0,0,0,0,1,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,1,0,0,0,
		0,0,0,1,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,1,0,0,
		0,0,1,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,1,0,
		0,1,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,1,
		1,0,0,0,0,0,0,0,0,0,0,0,0;
	int nbasis=0;
	for (int ishell=0;ishell<(int)shelltypes.size();ishell++){
		int l=abs(shelltypes[ishell]);
		nbasis+=2*l+1;
	}
	EigenMatrix transform=EigenZero(nbasis,nbasis);
	for (int ishell=0,jbasis=0;ishell<(int)shelltypes.size();ishell++){
		int l=abs(shelltypes[ishell]);
		transform.block(jbasis,jbasis,2*l+1,2*l+1)=SPDFGHI[l];
		jbasis+=2*l+1;
	}
	return transform;
}

double SafeStof(std::string word){ // In FT theory, some occupation numbers can be of dimension of 1E-50 or less. These small values cannot be stored as doubles and are likely to cause "out_of_range" error.
	double value=1919810;
	try{
		value=std::stof(word);
	}catch (const std::out_of_range& e){
		value=0;
	}
	return value;
}

Mwfn::Mwfn(std::string filename,const bool output){
	std::ifstream file(filename.c_str());
	if (!file.good()) return;
	if (output)
		std::cout<<"Reading existent Multiwfn file "<<filename<<" ..."<<std::endl;
	std::string line,word;
	int tmp_int=-114514;
	EigenMatrix mwfntransform=EigenZero(1,1);
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
		}else if (word.compare("Nprims=")==0){
			ss>>word;
			this->Nprims=std::stoi(word);
			continue;
		}else if (word.compare("Nshell=")==0){
			ss>>word;
			this->Nshell=std::stoi(word);
			continue;
		}else if (word.compare("Nprimshell=")==0){
			ss>>word;
			this->Nprimshell=std::stoi(word);
			continue;
		}else if (word.compare("$Shell")==0){
			ss>>word;
			if (word.compare("types")==0){
				int total=this->Nshell;
				this->Shell_types.resize(total,-114514);
				__Read_Array_Head__
					this->Shell_types[k]=std::stoi(word);
				__Read_Array_Tail__
				mwfntransform=MwfnMatrixTransform(this->Shell_types);
				continue;
			}else if (word.compare("centers")==0){
				int total=this->Nshell;
				this->Shell_centers.resize(total,-114514);
				__Read_Array_Head__
					this->Shell_centers[k]=std::stoi(word);
				__Read_Array_Tail__
				continue;
			}else if (word.compare("contraction")==0){
				int total=this->Nshell;
				this->Shell_contraction_degrees.resize(total,-114514);
				__Read_Array_Head__
					this->Shell_contraction_degrees[k]=std::stoi(word);
				__Read_Array_Tail__
				continue;
			}
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
			this->Energy[tmp_int]=SafeStof(word);
			continue;
		}else if (word.compare("Occ=")==0){
			ss>>word;
			this->Occ[tmp_int]=SafeStof(word);
			continue;
		}else if (word.compare("Sym=")==0){
			ss>>word;
			this->Sym[tmp_int]=word;
			continue;
		}else if (word.compare("$Coeff")==0){
			int ibasis=0;
			int total=this->Nbasis;
			__Read_Array_Head__
				this->Coeff(ibasis,tmp_int)=SafeStof(word);
				ibasis++;
			__Read_Array_Tail__
			this->Coeff=mwfntransform.transpose()*this->Coeff;
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
			this->Total_density_matrix=mwfntransform.transpose()*this->Total_density_matrix*mwfntransform;
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
			this->Overlap_matrix=mwfntransform.transpose()*this->Overlap_matrix*mwfntransform;
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
}

void Mwfn::Export(std::string filename,const bool output){
	std::FILE * file=std::fopen(filename.c_str(),"w");
	if (output) std::cout<<"Exporting wavefunction information to "<<filename<<" ..."<<std::endl;
	std::fprintf(file,"# Generated by Chinium\n");

	// Field 1
	std::fprintf(file,"\n\n# Overview\n");
	std::fprintf(file,"Wfntype= %d\n",0);
	if (this->Charge!=-114514)
		std::fprintf(file,"Charge= %d\n",this->Charge);
	if (this->Naelec!=-114514)
		std::fprintf(file,"Naelec= %d\n",this->Naelec);
	if (this->Nbelec!=-114514)
		std::fprintf(file,"Nbelec= %d\n",this->Nbelec);
	if (this->E_tot!=-114514)
		std::fprintf(file,"E_tot= %f\n",this->E_tot);
	if (this->VT_ratio!=-114514)
		std::fprintf(file,"VT_ratio= %f\n",this->VT_ratio);

	// Field 2
	__Z_2_Name__
	std::fprintf(file,"\n\n# Atoms\n");
	if (this->Ncenter!=-114514)
		std::fprintf(file,"Ncenter= %d\n",this->Ncenter);
	if (Centers.size()){
		std::fprintf(file,"$Centers\n");
		for (int icenter=0;icenter<this->Ncenter;icenter++)
			std::fprintf(
					file,"%d %s %d %f %f %f %f\n",
					icenter+1,
					Z2Name[(int)this->Centers[4*icenter]].c_str(),
					(int)this->Centers[4*icenter],
					this->Centers[4*icenter+0],
					this->Centers[4*icenter+1]/__angstrom2bohr__,
					this->Centers[4*icenter+2]/__angstrom2bohr__,
					this->Centers[4*icenter+3]/__angstrom2bohr__);
	}

	// Field 3
	std::fprintf(file,"\n\n# Basis set\n");
	if (this->Nbasis!=-114514)
		std::fprintf(file,"Nbasis= %d\n",this->Nbasis);
	if (this->Nindbasis!=-114514)
		std::fprintf(file,"Nindbasis= %d\n",this->Nindbasis);
	if (this->Nprims!=-114514)
		std::fprintf(file,"Nprims= %d\n",this->Nprims);
	if (this->Nshell!=-114514)
		std::fprintf(file,"Nshell= %d\n",this->Nshell);
	if (this->Nprimshell!=-114514)
		std::fprintf(file,"Nprimshell= %d\n",this->Nprimshell);
	if (this->Shell_centers.size()){
		std::vector<int> tmp_vec(this->Ncenter,0); // Number of shells of each atom
		for (int ishell=0;ishell<this->Nshell;ishell++)
			tmp_vec[this->Shell_centers[ishell]]++;
		std::fprintf(file,"$Shell types\n");
		for (int icenter=0,kshell=0;icenter<this->Ncenter;icenter++){
			for (int j=0;j<tmp_vec[icenter];j++,kshell++)
				std::fprintf(file," %d",(this->Shell_types[kshell]<2?1:-1)*this->Shell_types[kshell]); // Chinium uses only pure spherical harmonics.
			std::fprintf(file,"\n");
		}
		std::fprintf(file,"$Shell centers\n");
		for (int icenter=0;icenter<this->Ncenter;icenter++){
			for (int j=0;j<tmp_vec[icenter];j++)
				std::fprintf(file," %d",icenter+1);
			std::fprintf(file,"\n");
		}
		std::fprintf(file,"$Shell contraction degrees\n");
		for (int icenter=0,kshell=0;icenter<this->Ncenter;icenter++){
			for (int j=0;j<tmp_vec[icenter];j++,kshell++)
				std::fprintf(file," %d",this->Shell_contraction_degrees[kshell]);
			std::fprintf(file,"\n");
		}

		std::fprintf(file,"$Primitive exponents\n");
		for (int ishell=0,kprim=0;ishell<this->Nshell;ishell++){
			for (int j=0;j<this->Shell_contraction_degrees[ishell];j++,kprim++)
				std::fprintf(file," %f",this->Primitive_exponents[kprim]);
			std::fprintf(file,"\n");
		}
		std::fprintf(file,"$Contraction coefficients\n");
		for (int ishell=0,kprim=0;ishell<this->Nshell;ishell++){
			for (int j=0;j<this->Shell_contraction_degrees[ishell];j++,kprim++)
				std::fprintf(file," %f",this->Contraction_coefficients[kprim]);
			std::fprintf(file,"\n");
		}
	}

	// Field 4
	std::fprintf(file,"\n\n# Orbitals\n");
	EigenMatrix mwfntransform=MwfnMatrixTransform(this->Shell_types);
	EigenMatrix transformedCoeff=mwfntransform*this->Coeff;
	if (this->Energy.size()){
		for (int i=0;i<this->Nbasis;i++){
			std::fprintf(file,"Index= %9d\n",i+1);
			std::fprintf(file,"Type= %d\n",this->Type[i]);
			std::fprintf(file,"Energy= %E\n",this->Energy[i]);
			std::fprintf(file,"Occ= %E\n",this->Occ[i]);
			std::fprintf(file,"Sym= %s\n",this->Sym[i].c_str());
			std::fprintf(file,"$Coeff\n");
			for (int j=0;j<this->Nbasis;j++)
				std::fprintf(file," %E",transformedCoeff(j,i));
			std::fprintf(file,"\n\n");
		}
	}

	// Field 5
	std::fprintf(file,"\n\n# Matrices\n");
	if (this->Total_density_matrix.cols()){
		std::fprintf(file,"$Total density matrix, dim= %d %d lower= 1\n",this->Nbasis,this->Nbasis);
		PrintMatrix(file,mwfntransform*this->Total_density_matrix*mwfntransform.transpose(),1);
	}
	if (this->Hamiltonian_matrix.cols()){
		std::fprintf(file,"$1-e Hamiltonian matrix, dim= %d %d lower= 1\n",this->Nbasis,this->Nbasis);
		PrintMatrix(file,mwfntransform*this->Hamiltonian_matrix*mwfntransform.transpose(),1);
	}
	if (this->Overlap_matrix.cols()){
		std::fprintf(file,"$Overlap matrix, dim= %d %d lower= 1\n",this->Nbasis,this->Nbasis);
		PrintMatrix(file,mwfntransform*this->Overlap_matrix*mwfntransform.transpose(),1);
	}
	if (this->Kinetic_energy_matrix.cols()){
		std::fprintf(file,"$Kinetic energy matrix, dim= %d %d lower= 1\n",this->Nbasis,this->Nbasis);
		PrintMatrix(file,mwfntransform*this->Kinetic_energy_matrix*mwfntransform.transpose(),1);
	}
	if (this->Potential_energy_matrix.cols()){
		std::fprintf(file,"$Potential energy matrix, dim= %d %d lower= 1\n",this->Nbasis,this->Nbasis);
		PrintMatrix(file,mwfntransform*this->Potential_energy_matrix*mwfntransform.transpose(),1);
	}

	std::fclose(file);
}




