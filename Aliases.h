#define __nan__ (0./0.)
#define __angstrom2bohr__ 1.8897259886
#define __hartree2ev__ 27.21139664130791

#define EigenVector Eigen::VectorXd
#define EigenMatrix Eigen::MatrixXd
#define EigenZero Eigen::MatrixXd::Zero
#define EigenOne Eigen::MatrixXd::Identity

#define __Name_2_Z__\
	std::map<std::string,double> Name2Z={\
		{"H",1},{"He",2},{"Li",3},{"Be",4},{"B",5},\
		{"C",6},{"N",7},{"O",8},{"F",9},{"Ne",10},\
		{"Na",11},{"Mg",12},{"Al",13},{"Si",14},{"P",15},\
		{"S",16},{"Cl",17},{"Ar",18},{"K",19},{"Ca",20}\
	};

#define __Z_2_Name__\
	std::string Z2Name[]={"FuckIndexZero",\
		"H","He","Li","Be","B","C","N","O","F","Ne",\
		"Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca"\
	};

#define __Libint2_Atoms__\
	std::vector<libint2::Atom> libint2atoms(natoms);\
	for (int iatom=0;iatom<natoms;iatom++){\
		libint2::Atom atomi;\
		atomi.atomic_number=(int)atoms[iatom*4];\
		atomi.x=atoms[iatom*4+1];\
		atomi.y=atoms[iatom*4+2];\
		atomi.z=atoms[iatom*4+3];\
		libint2atoms[iatom]=atomi;\
	} // Converting atoms array to libint's std::vector<libint2::Atom>.

#define __nBasis_From_OBS__\
	int nbasis=0;\
	for (const auto& shell:obs)\
		nbasis+=shell.size(); // Size of basis set directly derived from libint2::BasisSet.

#define __Basis_From_Atoms__\
	__Libint2_Atoms__\
	libint2::BasisSet obs(basisset,libint2atoms);\
	obs.set_pure(1);

#define __Delete_Matrices__(matrices,size)\
	for (int imatrix=0;imatrix<size;imatrix++)\
		matrices.resize(0,0);\
	delete [] matrices;
