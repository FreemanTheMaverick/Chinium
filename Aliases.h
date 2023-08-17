#define __angstrom2bohr__ 1.8897259886

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
		"h","he","li","be","b","c","n","o","f","ne",\
		"na","mg","al","si","p","s","cl","ar","k","ca"\
	};
