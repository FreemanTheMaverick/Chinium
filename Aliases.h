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
		matrices[imatrix].resize(0,0);\
	delete [] matrices;

#define __Delete_Vectors__(vectors,size)\
	for (int ivector=0;ivector<size;ivector++)\
		vectors[ivector].resize(0);\
	delete [] vectors;

#define __Begin_KS__(aos,ao1xs,ao1ys,ao1zs,ao2ls,e)\
	double * ds=nullptr;\
	double * d1xs=nullptr;\
	double * d1ys=nullptr;\
	double * d1zs=nullptr;\
	double * cgs=nullptr;\
	double * d2s=nullptr;\
	double * ts=nullptr;\
	double * vrxcs=nullptr;\
	double * vsxcs=nullptr;\
	double * vlxcs=nullptr;\
	double * vtxcs=nullptr;\
	double * excs=nullptr;\
	double * vrcs=nullptr;\
	double * vscs=nullptr;\
	double * vlcs=nullptr;\
	double * vtcs=nullptr;\
	double * ecs=nullptr;\
	int xkind,ckind,xfamily,cfamily;\
	xkind=ckind=xfamily=cfamily=114514;\
	double kscale=1;\
	if (dfxid){\
		if (aos) ds=new double[ngrids]();\
		if (ao1xs){\
			d1xs=new double[ngrids]();\
			d1ys=new double[ngrids]();\
			d1zs=new double[ngrids]();\
			cgs=new double[ngrids]();\
		}\
		if (ao2ls){\
			d2s=new double[ngrids]();\
			ts=new double[ngrids]();\
		}\
		char rubbish[64];\
		if (dfcid && dfxid!=dfcid){\
			XCInfo(dfcid,rubbish,ckind,cfamily,kscale);\
			if (aos) vrcs=new double[ngrids]();\
			if (ao1xs) vscs=new double[ngrids]();\
			if (ao2ls){\
				vlcs=new double[ngrids]();\
				vtcs=new double[ngrids]();\
			}\
			if (e) ecs=new double[ngrids]();\
		}\
		XCInfo(dfxid,rubbish,xkind,xfamily,kscale);\
		if (aos) vrxcs=new double[ngrids]();\
		if (ao1xs) vsxcs=new double[ngrids]();\
		if (ao2ls){\
			vlxcs=new double[ngrids]();\
			vtxcs=new double[ngrids]();\
		}\
		if (e) excs=new double[ngrids]();\
		GetDensity(\
				aos,\
				ao1xs,ao1ys,ao1zs,\
				ao2ls,\
				ngrids,2*D,\
				ds,\
				d1xs,d1ys,d1zs,cgs,\
				d2s,ts);\
		if (e) getEVxc(dfxid,ds,cgs,d2s,ts,ngrids,excs,vrxcs,vsxcs,vlxcs,vtxcs);\
		else getVxc(dfxid,ds,cgs,d2s,ts,ngrids,vrxcs,vsxcs,vlxcs,vtxcs);\
		if (dfcid && dfxid!=dfcid){\
			if (e) getEVxc(dfcid,ds,cgs,d2s,ts,ngrids,ecs,vrcs,vscs,vlcs,vtcs);\
			else getVxc(dfcid,ds,cgs,d2s,ts,ngrids,vrcs,vscs,vlcs,vtcs);\
			VectorAddition(vrxcs,vrcs,ngrids);\
			VectorAddition(vsxcs,vscs,ngrids);\
			VectorAddition(vlxcs,vlcs,ngrids);\
			VectorAddition(vtxcs,vtcs,ngrids);\
			if (e) VectorAddition(excs,ecs,ngrids);\
		}\
	}

#define __Finalize_KS__\
	if (ds) delete [] ds;\
	if (d1xs) delete [] d1xs;\
	if (d1ys) delete [] d1ys;\
	if (d1zs) delete [] d1zs;\
	if (cgs) delete [] cgs;\
	if (d2s) delete [] d2s;\
	if (ts) delete [] ts;\
	if (vrxcs) delete [] vrxcs;\
	if (vsxcs) delete [] vsxcs;\
	if (vlxcs) delete [] vlxcs;\
	if (vtxcs) delete [] vtxcs;\
	if (excs) delete [] excs;\
	if (vrcs) delete [] vrcs;\
	if (vscs) delete [] vscs;\
	if (vlcs) delete [] vlcs;\
	if (vtcs) delete [] vtcs;\
	if (ecs) delete [] ecs;

