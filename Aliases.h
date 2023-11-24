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

#define __Initialize_KS__(aos,ao1xs,ao1ys,ao1zs,ao2ls,u,e)\
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
	double * vrrxcs=nullptr;\
	double * vrsxcs=nullptr;\
	double * vssxcs=nullptr;\
	double * vrlxcs=nullptr;\
	double * vrtxcs=nullptr;\
	double * vslxcs=nullptr;\
	double * vstxcs=nullptr;\
	double * vllxcs=nullptr;\
	double * vltxcs=nullptr;\
	double * vttxcs=nullptr;\
	\
	double * excs=nullptr;\
	double * vrcs=nullptr;\
	double * vscs=nullptr;\
	double * vlcs=nullptr;\
	double * vtcs=nullptr;\
	double * vrrcs=nullptr;\
	double * vrscs=nullptr;\
	double * vsscs=nullptr;\
	double * vrlcs=nullptr;\
	double * vrtcs=nullptr;\
	double * vslcs=nullptr;\
	double * vstcs=nullptr;\
	double * vllcs=nullptr;\
	double * vltcs=nullptr;\
	double * vttcs=nullptr;\
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
		char rubbish[64]="1919810";\
		if (dfcid && dfxid!=dfcid){\
			XCInfo(dfcid,rubbish,ckind,cfamily,kscale);\
			if (! u && aos) vrcs=new double[ngrids]();\
			if (! u && ao1xs) vscs=new double[ngrids]();\
			if (! u && ao2ls){\
				vlcs=new double[ngrids]();\
				vtcs=new double[ngrids]();\
			}\
			if (! u && e) ecs=new double[ngrids]();\
			if (u && aos) vrrcs=new double[ngrids]();\
			if (u && ao1xs){\
				vrcs=new double[ngrids]();\
				vscs=new double[ngrids]();\
				vrscs=new double[ngrids]();\
				vsscs=new double[ngrids]();\
			}\
		}\
		XCInfo(dfxid,rubbish,xkind,xfamily,kscale);\
		if (! u && aos) vrxcs=new double[ngrids]();\
		if (! u && ao1xs) vsxcs=new double[ngrids]();\
		if (! u && ao2ls){\
			vlxcs=new double[ngrids]();\
			vtxcs=new double[ngrids]();\
		}\
		if (! u && e) excs=new double[ngrids]();\
		if (u && aos) vrrxcs=new double[ngrids]();\
		if (u && ao1xs){\
			vrxcs=new double[ngrids]();\
			vsxcs=new double[ngrids]();\
			vrsxcs=new double[ngrids]();\
			vssxcs=new double[ngrids]();\
		}\
	}

#define __KS_Potential__(u,e)\
	if (dfxid){\
		if (! u && e) getEVxc(dfxid,ds,cgs,d2s,ts,ngrids,excs,vrxcs,vsxcs,vlxcs,vtxcs);\
		if (! u && ! e) getVxc(dfxid,ds,cgs,d2s,ts,ngrids,vrxcs,vsxcs,vlxcs,vtxcs);\
		if (u){\
			getVxc(dfxid,ds,cgs,d2s,ts,ngrids,vrxcs,vsxcs,vlxcs,vtxcs);\
			getFxc(\
				dfxid,ngrids,\
				ds,\
				cgs,\
				d2s,ts,\
				vrrxcs,\
				vrsxcs,vssxcs,\
				vrlxcs,vrtxcs,vslxcs,vstxcs,vllxcs,vltxcs,vttxcs);\
		}\
		if (dfcid && dfxid!=dfcid){\
			if (! u && e) getEVxc(dfcid,ds,cgs,d2s,ts,ngrids,ecs,vrcs,vscs,vlcs,vtcs);\
			if (! u && ! e) getVxc(dfcid,ds,cgs,d2s,ts,ngrids,vrcs,vscs,vlcs,vtcs);\
			if (u){\
				getVxc(dfcid,ds,cgs,d2s,ts,ngrids,vrcs,vscs,vlcs,vtcs);\
				getFxc(\
					dfcid,ngrids,\
					ds,\
					cgs,\
					d2s,ts,\
					vrrcs,\
					vrscs,vsscs,\
					vrlcs,vrtcs,vslcs,vstcs,vllcs,vltcs,vttcs);\
			}\
			if (e) VectorAddition(excs,excs,ecs,ngrids);\
			if (vrxcs) VectorAddition(vrxcs,vrxcs,vrcs,ngrids);\
			if (vsxcs) VectorAddition(vsxcs,vsxcs,vscs,ngrids);\
			if (vlxcs) VectorAddition(vlxcs,vlxcs,vlcs,ngrids);\
			if (vtxcs) VectorAddition(vtxcs,vtxcs,vtcs,ngrids);\
			if (vrrxcs) VectorAddition(vrrxcs,vrrxcs,vrrcs,ngrids);\
			if (vrsxcs) VectorAddition(vrsxcs,vrsxcs,vrscs,ngrids);\
			if (vssxcs) VectorAddition(vssxcs,vssxcs,vsscs,ngrids);\
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
	if (excs) delete [] excs;\
	if (vrxcs) delete [] vrxcs;\
	if (vsxcs) delete [] vsxcs;\
	if (vlxcs) delete [] vlxcs;\
	if (vtxcs) delete [] vtxcs;\
	if (vrrxcs) delete [] vrrxcs;\
	if (vrsxcs) delete [] vrsxcs;\
	if (vssxcs) delete [] vssxcs;\
	if (vrlxcs) delete [] vrlxcs;\
	if (vrtxcs) delete [] vrtxcs;\
	if (vslxcs) delete [] vslxcs;\
	if (vstxcs) delete [] vstxcs;\
	if (vllxcs) delete [] vllxcs;\
	if (vltxcs) delete [] vltxcs;\
	if (vttxcs) delete [] vttxcs;\
	if (ecs) delete [] ecs;\
	if (vrcs) delete [] vrcs;\
	if (vscs) delete [] vscs;\
	if (vlcs) delete [] vlcs;\
	if (vtcs) delete [] vtcs;\
	if (vrrcs) delete [] vrrcs;\
	if (vrscs) delete [] vrscs;\
	if (vsscs) delete [] vsscs;\
	if (vrlcs) delete [] vrlcs;\
	if (vrtcs) delete [] vrtcs;\
	if (vslcs) delete [] vslcs;\
	if (vstcs) delete [] vstcs;\
	if (vllcs) delete [] vllcs;\
	if (vltcs) delete [] vltcs;\
	if (vttcs) delete [] vttcs;

