#include <libint2.hpp>
#include <Eigen/Dense> // Eigen::Matrix.
#include "PointGroups.h"
#include <string>
#include <vector>
#include <cmath> // Trigonometric functions in rotation.
#include <iostream>

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;
typedef Eigen::Matrix<double,3,1> EigenVector;

double AtomicMass(int atomic_number){ // Mapping atomic numbers to atomic masses.
	double atomicmass;
	switch(atomic_number){
		case 1:atomicmass=1.0079;break;
		case 2:atomicmass=4.0026;break;
		case 3:atomicmass=6.9412;break;
		case 4:atomicmass=9.0121;break;
		case 5:atomicmass=10.811;break;
		case 6:atomicmass=12.010;break;
		case 7:atomicmass=14.006;break;
		case 8:atomicmass=15.999;break;
		case 9:atomicmass=18.998;break;
		case 10:atomicmass=20.179;break;
		case 11:atomicmass=22.989;break;
		case 12:atomicmass=24.305;break;
		case 13:atomicmass=26.981;break;
		case 14:atomicmass=28.085;break;
		case 15:atomicmass=30.973;break;
		case 16:atomicmass=32.065;break;
		case 17:atomicmass=35.453;break;
		case 18:atomicmass=39.948;break;
	}
	return atomicmass;
}

EigenMatrix Vector2Matrix(std::vector<libint2::Atom> atoms){ // libint2 stores molecular geometries in the format of std::vector<libint2::Atom>, which is not suitable for matrix manipulation, so it is necessary to convert them from std::vector<libint2::Atom> to EigenMatrix.
	int natoms=atoms.size();
	EigenMatrix coordinates(3,natoms);
	for (int iatom=0;iatom<natoms;iatom++){
		libint2::Atom atom=atoms[iatom];
		coordinates(0,iatom)=atom.x; // Atomic coordinates are stored in the format: [[x1,x2,x3,...],[y1,y2,y3,...],[z1,z2,z3,...]].
		coordinates(1,iatom)=atom.y;
		coordinates(2,iatom)=atom.z;
	}
	return coordinates;
}

std::vector<libint2::Atom> Matrix2Vector(std::vector<libint2::Atom> atoms,EigenMatrix coordinates){ // Converting molecular geometries from EigenMatrix back to std::vector<libint2::Atom>, so that libint2 can read them. Since EigenMatrix does not store atomic numbers, the original std::vector<libint2::Atom> is needed here for atomic numbers.
	int natoms=atoms.size();
	std::vector<libint2::Atom> newatoms=atoms;
	for (int iatom=0;iatom<natoms;iatom++){
		newatoms[iatom].x=coordinates(0,iatom);
		newatoms[iatom].y=coordinates(1,iatom);
		newatoms[iatom].z=coordinates(2,iatom);
	}
	return newatoms;
}

int FindImage(libint2::Atom atomi,std::vector<libint2::Atom> projections,double tolerance){ // Finding the image of atom i among atom projections corresponding to some symmetry operation.
	int Zi=atomi.atomic_number;
	double xi=atomi.x;
	double yi=atomi.y;
	double zi=atomi.z;
	EigenVector coordinatesi(xi,yi,zi); // Location of atom i.
	int natoms=projections.size();
	int nearestjatom;
	double nearestxj=114514; // The closest atom j to atom i's projection. Any arbitarily large numbers as initial values.
	double nearestyj=1919810;
	double nearestzj=889464;
	double nearestdeviationsquared=364364; // Deviation between the closest atom j and atom i's projection.
	EigenVector nearestcoordinatesj;
	for (int jatom=0;jatom<natoms;jatom++){ // Finding the closest atom j to atom i's projection.
		libint2::Atom atomj=projections[jatom];
		int Zj=atomj.atomic_number;
		if (Zi==Zj){ // Atom i and atom j must share the same atomic number if their locations are to be compared.
			double xj=atomj.x;
			double yj=atomj.y;
			double zj=atomj.z;
			EigenVector coordinatesj(xj,yj,zj); // Location of atom j.
			EigenVector deviationvector=coordinatesi-coordinatesj; // Displacement vector between atom i's projection and the closest atom j.
			double deviationsquared=deviationvector.dot(deviationvector); // Square of distance between atom i's projection and the closest atom j.
			if (nearestdeviationsquared>deviationsquared){ // Whether this atom j is closer than last atom j to atom i.
				nearestjatom=jatom;
				nearestcoordinatesj=coordinatesj; // If so, update the closest atom j and the corresponding deviation.
				nearestdeviationsquared=deviationsquared;
			}
		}
	}
	EigenVector interatomicdistancevector=coordinatesi-nearestcoordinatesj; // Displacement vector between atom i and the closest atom j.
	double interatomicdistancesquared=interatomicdistancevector.dot(interatomicdistancevector); // Square of distance between atom i and the closest atom j.
	if (nearestdeviationsquared/(interatomicdistancesquared+tolerance*tolerance)<tolerance*tolerance){ // Deviation is scaled by the distance between atom i and atom j. More deviation is tolerated considering the large interatomic distance. If the reduced deviation is larger than tolerance, the boolean will be toggled to false and the loop will be broken.
		return nearestjatom;
	}else{
		return -1; // If the nearest atom j is not within tolerance, return -1.
	}
}

bool InversionCentre(std::vector<libint2::Atom> atoms,double tolerance){ // Testing whether the origin is the inversion centre. For a molecule on its standard orientation, the origin is the only point that may be its inversion centre.
	int natoms=atoms.size();
	bool inversioncentre=true;
	EigenMatrix coordinates=Vector2Matrix(atoms);
	std::vector<libint2::Atom> projections=Matrix2Vector(atoms,-coordinates);
	for (int iatom=0;iatom<natoms && inversioncentre;iatom++){ // Once one atom cannot find its projection, the loop is over.
		libint2::Atom atomi=atoms[iatom];
		inversioncentre=FindImage(atomi,projections,tolerance)!=-1;
	}
	return inversioncentre;
}

bool Mirror(std::vector<libint2::Atom> atoms,EigenVector normal,double tolerance){ // Checking if the plane with the given normal vector is one of the mirrors of the molecule.
	int natoms=atoms.size();
	EigenMatrix coordinates=Vector2Matrix(atoms);
	double mirrorx=normal(0);
	double mirrory=normal(1);
	double mirrorz=normal(2);
	bool mirror=true;
	EigenMatrix reflection{{1-2*mirrorx*mirrorx,-2*mirrorx*mirrory,-2*mirrorx*mirrorz},
	                       {-2*mirrorx*mirrory,1-2*mirrory*mirrory,-2*mirrory*mirrorz},
	                       {-2*mirrorx*mirrorz,-2*mirrory*mirrorz,1-2*mirrorz*mirrorz}};	
	std::vector<libint2::Atom> projections=Matrix2Vector(atoms,reflection*coordinates); // Projection of molecule with respective to reflection.
	for (int iatom=0;iatom<natoms && mirror;iatom++){ // Once one of the atoms cannot find its projection, the loop is over.
		libint2::Atom atomi=atoms[iatom];
		mirror=FindImage(atomi,projections,tolerance)!=-1;
	}
	return mirror;
}

bool ProperAxis(std::vector<libint2::Atom> atoms,EigenVector axis,int manifold,double tolerance){ // Determining if the given axis is one of the molecule's proper rotation axes.
	int natoms=atoms.size();
	bool properaxis=true;
	double angle=360/(double)manifold/180*3.1415926536; // Converting degrees to radii.
	EigenMatrix coordinates=Vector2Matrix(atoms);
	double axisx=axis(0); // Three components of axial vector. The vector must be a unit vector.
	double axisy=axis(1);
	double axisz=axis(2);
	for (int i=1;i<manifold && properaxis;i++){ // Looping over all possible rotation angles. For example, for a C_3 axis, the possible possible angles are 2pi/3, 4pi/3 and 2pi.
		EigenMatrix properrotation{{axisx*axisx*(1-cos(i*angle))+cos(i*angle),axisx*axisy*(1-cos(i*angle))-axisz*sin(i*angle),axisx*axisz*(1-cos(i*angle))+axisy*sin(i*angle)},
		                           {axisx*axisy*(1-cos(i*angle))+axisz*sin(i*angle),axisy*axisy*(1-cos(i*angle))+cos(i*angle),axisy*axisz*(1-cos(i*angle))-axisx*sin(i*angle)},
		                           {axisx*axisz*(1-cos(i*angle))-axisy*sin(i*angle),axisy*axisz*(1-cos(i*angle))+axisx*sin(i*angle),axisz*axisz*(1-cos(i*angle))+cos(i*angle)}}; // Proper rotation matrix.
		std::vector<libint2::Atom> projections=Matrix2Vector(atoms,properrotation*coordinates); // Projection of molecule with respective to proper rotation.
		for (int iatom=0;iatom<natoms && properaxis;iatom++){
			libint2::Atom atomi=atoms[iatom];
			properaxis=FindImage(atomi,projections,tolerance)!=-1;
		}
	}
	return properaxis;
}

bool ImproperAxis(std::vector<libint2::Atom> atoms,EigenVector axis,int manifold,double tolerance){ // Determining if the given axis is one of the molecule's proper rotation axes.
	int natoms=atoms.size();
	bool improperaxis=true;
	double angle=360/(double)manifold/180*3.1415926536; // Converting degrees to radii.
	EigenMatrix coordinates=Vector2Matrix(atoms);
	double axisx=axis(0); // Three components of axial vector. The vector must be a unit vector.
	double axisy=axis(1);
	double axisz=axis(2);
	for (int i=1;i<manifold && improperaxis;i++){ // Looping over all possible rotation angles. For example, for a C_3 axis, the possible possible angles are 2pi/3, 4pi/3 and 2pi.
		EigenMatrix properrotation{{axisx*axisx*(1-cos(i*angle))+cos(i*angle),axisx*axisy*(1-cos(i*angle))-axisz*sin(i*angle),axisx*axisz*(1-cos(i*angle))+axisy*sin(i*angle)},
		                     {axisx*axisy*(1-cos(i*angle))+axisz*sin(i*angle),axisy*axisy*(1-cos(i*angle))+cos(i*angle),axisy*axisz*(1-cos(i*angle))-axisx*sin(i*angle)},
		                     {axisx*axisz*(1-cos(i*angle))-axisy*sin(i*angle),axisy*axisz*(1-cos(i*angle))+axisx*sin(i*angle),axisz*axisz*(1-cos(i*angle))+cos(i*angle)}}; // Proper rotation matrix.
		EigenMatrix reflection{{1-2*axisx*axisx,-2*axisx*axisy,-2*axisx*axisz},
		                       {-2*axisx*axisy,1-2*axisy*axisy,-2*axisy*axisz},
		                       {-2*axisx*axisz,-2*axisy*axisz,1-2*axisz*axisz}};
		EigenMatrix improperrotation=i%2==1?properrotation*reflection:properrotation;
		std::vector<libint2::Atom> projections=Matrix2Vector(atoms,improperrotation*coordinates); // Projection of molecule with respective to proper rotation.
		for (int iatom=0;iatom<natoms && improperaxis;iatom++){
			libint2::Atom atomi=atoms[iatom];
			improperaxis=FindImage(atomi,projections,tolerance)!=-1;
		}
	}
	return improperaxis;
}

EigenMatrix HorizontalC2Axis(std::vector<libint2::Atom>& atoms,double tolerance){ // Finding one of the molecule's horizontal C2 axes, if any.
	EigenVector horizontalc2axis(0,0,0);
	int natoms=atoms.size();
	for (int iatom=0;iatom<natoms;iatom++){
                libint2::Atom atomi=atoms[iatom];
                int Zi=atomi.atomic_number;
                double xi=atomi.x;
                double yi=atomi.y;
                double zi=atomi.z;
		if (xi*xi+yi*yi<tolerance*tolerance){
			continue;
		}
		for (int jatom=iatom;jatom<natoms;jatom++){
	                libint2::Atom atomj=atoms[jatom];
			int Zj=atomj.atomic_number;
			double xj=atomj.x;
			double yj=atomj.y;
			double zj=atomj.z;
			if (xj*xj+yj*yj<tolerance*tolerance || (xi+xj)*(xi+xj)/4+(yi+yj)*(yi+yj)/4<tolerance*tolerance){
				continue;
			}
			if (Zi==Zj && (zi+zj)*(zi+zj)<tolerance*tolerance){ // The horizontal C2 axis must pass the origin and an atom in xy plane, or the origin and the midpoint of two like atoms seperated by xy plane.
				EigenVector potentialc2axis((xi+xj),(yi+yj),0);
				potentialc2axis=potentialc2axis/sqrt(potentialc2axis.dot(potentialc2axis));
				if (ProperAxis(atoms,potentialc2axis,2,tolerance)){ // Whether the axis is a C2 axis of the molecule.
					horizontalc2axis=potentialc2axis;
					return horizontalc2axis;
				}
			}
		}
	}
	return horizontalc2axis;
}

EigenMatrix HorizontalMirror(std::vector<libint2::Atom>& atoms,double tolerance){ // Finding one of the molecule's horizontal mirror, if any.
	EigenVector horizontalmirror(0,0,0);
	int natoms=atoms.size();
	for (int iatom=0;iatom<natoms;iatom++){ // First checking planes that bisect atoms.
                libint2::Atom atomi=atoms[iatom];
                int Zi=atomi.atomic_number;
                double xi=atomi.x;
                double yi=atomi.y;
                double zi=atomi.z;
		if (xi*xi+yi*yi<tolerance*tolerance){
			continue;
		}
		EigenVector potentialmirror(yi,-xi,0);
		potentialmirror=potentialmirror/sqrt(potentialmirror.dot(potentialmirror));
		if (Mirror(atoms,potentialmirror,tolerance)){ // Whether the plane is a mirror of the molecule.
			horizontalmirror=potentialmirror;
			return horizontalmirror;
		}
	}
	for (int iatom=0;iatom<natoms;iatom++){ // Then checking planes that lie between like atoms.
		libint2::Atom atomi=atoms[iatom];
		int Zi=atomi.atomic_number;
		double xi=atomi.x;
		double yi=atomi.y;
		double zi=atomi.z;
		if (xi*xi+yi*yi<tolerance*tolerance){
			continue;
		}
		for (int jatom=iatom+1;jatom<natoms;jatom++){
	                libint2::Atom atomj=atoms[jatom];
			int Zj=atomj.atomic_number;
			double xj=atomj.x;
			double yj=atomj.y;
			double zj=atomj.z;
			if (xj*xj+yj*yj<tolerance*tolerance || (xi+xj)*(xi+xj)/4+(yi+yj)*(yi+yj)/4<tolerance*tolerance){
				continue;
			}
			if (Zi==Zj && (zi-zj)*(zi-zj)<tolerance*tolerance){
				EigenVector potentialmirror((yi+yj),-(xi+xj),0);
				potentialmirror=potentialmirror/sqrt(potentialmirror.dot(potentialmirror));
				if (Mirror(atoms,potentialmirror,tolerance)){
					horizontalmirror=potentialmirror;
					return horizontalmirror;
				}
			}
		}
	}
	return horizontalmirror;
}

PointGroup GetPointGroup(std::vector<libint2::Atom>& atoms,double tolerance){ // Moving the molecule to the standard orientation.
	int natoms=atoms.size();
	double mtotal,mxtotal,mytotal,mztotal; // Determining the centre of mass.
	for (int iatom=0;iatom<natoms;iatom++){
		libint2::Atom atom=atoms[iatom];
		int Z=atom.atomic_number;
		double x=atom.x;
		double y=atom.y;
		double z=atom.z;
		mtotal=mtotal+AtomicMass(Z);
		mxtotal=mxtotal+AtomicMass(Z)*x;
		mytotal=mytotal+AtomicMass(Z)*y;
		mztotal=mztotal+AtomicMass(Z)*z;
	}
	EigenMatrix masscentre(3,1);
	masscentre(0,0)=mxtotal/mtotal;
	masscentre(1,0)=mytotal/mtotal;
	masscentre(2,0)=mztotal/mtotal;

	double ixx=0; // Calculating the inertial tensor.
	double ixy=0;
	double ixz=0;
	double iyy=0;
	double iyz=0;
	double izz=0;
	for (int iatom=0;iatom<natoms;iatom++){
		libint2::Atom atom=atoms[iatom];
		int Z=atom.atomic_number;
		double xprime=atom.x-masscentre(0,0);
		double yprime=atom.y-masscentre(1,0);
		double zprime=atom.z-masscentre(2,0);
		ixx=ixx+(yprime*yprime+zprime*zprime)*AtomicMass(Z);
		ixy=ixy-xprime*yprime*AtomicMass(Z);
		ixz=ixz-xprime*zprime*AtomicMass(Z);
		iyy=iyy+(xprime*xprime+zprime*zprime)*AtomicMass(Z);
		iyz=iyz-yprime*zprime*AtomicMass(Z);
		izz=izz+(xprime*xprime+yprime*yprime)*AtomicMass(Z);
	}
	EigenMatrix inertialtensor(3,3);
	inertialtensor(0,0)=ixx;inertialtensor(0,1)=ixy;inertialtensor(0,2)=ixz;
	inertialtensor(1,0)=ixy;inertialtensor(1,1)=iyy;inertialtensor(1,2)=iyz;
	inertialtensor(2,0)=ixz;inertialtensor(2,1)=iyz;inertialtensor(2,2)=izz;

	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver; // Diagonalizing the inertial tensor.
	eigensolver.compute(inertialtensor); // Diagonalizing the inertial tensor. The eigenvector matrix is the rotation matrix that rotates the molecule to the standard orientation.
	EigenMatrix I=eigensolver.eigenvalues();
	EigenMatrix U=eigensolver.eigenvectors();
	EigenMatrix coordinates=Vector2Matrix(atoms);
	EigenMatrix newcoordinates=coordinates;
	for (int iatom=0;iatom<natoms;iatom++){ // Translating the centre of mass of the molecule to the origin.
		newcoordinates(0,iatom)=coordinates(0,iatom)-masscentre(0,0);
		newcoordinates(1,iatom)=coordinates(1,iatom)-masscentre(1,0);
		newcoordinates(2,iatom)=coordinates(2,iatom)-masscentre(2,0);
	}
	EigenMatrix standardcoordinates=U.transpose()*newcoordinates; // Rotating the molecule, so that the principal axes coincide with cartesian coordinates.
	std::vector<libint2::Atom> newatoms=Matrix2Vector(atoms,standardcoordinates);
	atoms=newatoms;

	ixx=I(0,0);
	iyy=I(1,0);
	izz=I(2,0);
	EigenMatrix mainx2mainz(3,3);mainx2mainz<<0,0,1, // Matrix that reorients a molecule so that its main axis is moved from x axis to z axis.
	                                          1,0,0,
	                                          0,1,0;
	EigenMatrix mainy2mainz(3,3);mainy2mainz<<0,1,0, // Matrix that reorients a molecule so that its main axis is moved from y axis to z axis.
	                                          0,0,1,
	                                          1,0,0;
	PointGroup pointgroup;
	if (ixx*ixx<tolerance*tolerance || iyy*iyy<tolerance*tolerance || izz*izz<tolerance*tolerance){ // Molecules of point groups Cv and Dh have one zero inertial moment.
		bool inversioncentre=InversionCentre(atoms,tolerance);
		pointgroup.inversion_centre=inversioncentre;
		if (inversioncentre){ // Dh has an inversion centre while Cv does not.
//			pointgroup.point_group_name="Dh";
			atoms=Matrix2Vector(atoms,mainx2mainz.transpose()*Vector2Matrix(atoms)); // Reorienting the molecule so that the main axis lies on z axis.
		}else{ // Dh has an inversion centre while Cv does not.
//			pointgroup.point_group_name="Cv";
			atoms=Matrix2Vector(atoms,mainx2mainz.transpose()*Vector2Matrix(atoms)); // Reorienting the molecule so that the main axis lies on z axis.
		}
	}else if ((ixx-iyy)*(ixx-iyy)>tolerance*tolerance && (ixx-izz)*(ixx-izz)>tolerance*tolerance && (iyy-izz)*(iyy-izz)>tolerance*tolerance){ // Molecules of point groups C1, Cs, Ci, C2, C2v, C2h, D2 and D2h have three unequivalent eigen inertial moments.
		EigenVector xaxis(1,0,0);
		EigenVector yaxis(0,1,0);
		EigenVector zaxis(0,0,1);
		if (Mirror(atoms,xaxis,tolerance) || Mirror(atoms,yaxis,tolerance) || Mirror(atoms,zaxis,tolerance)){ // Cs, C2v, C2h and D2h have at least one mirror respectively, while  C1, Ci, C2 and D2 do not.
			if (InversionCentre(atoms,tolerance)){ // C2h and D2h have an inversion centre respectively, while Cs and C2v do not.
				if (Mirror(atoms,xaxis,tolerance) && Mirror(atoms,yaxis,tolerance) && Mirror(atoms,zaxis,tolerance)){ // D2h has another three mirrors, while C2h does not.
					pointgroup=D2h;
				}else{ // D2h has three mirrors, while C2h does not.
					pointgroup=C2h;
					if (Mirror(atoms,xaxis,tolerance)){
						atoms=Matrix2Vector(atoms,mainx2mainz.transpose()*Vector2Matrix(atoms));
					}else if (Mirror(atoms,yaxis,tolerance)){
						atoms=Matrix2Vector(atoms,mainy2mainz.transpose()*Vector2Matrix(atoms));
					}
				}
			}else{ // C2h and D2h have an inversion centre respectively, while Cs and C2v do not.
				if (ProperAxis(atoms,xaxis,2,tolerance) || ProperAxis(atoms,yaxis,2,tolerance) || ProperAxis(atoms,zaxis,2,tolerance)){ // C2v has a proper axis, while Cs does not.
					pointgroup=C2v;
					if (ProperAxis(atoms,xaxis,2,tolerance)){
						atoms=Matrix2Vector(atoms,mainx2mainz.transpose()*Vector2Matrix(atoms));
					}else if (ProperAxis(atoms,yaxis,2,tolerance)){
						atoms=Matrix2Vector(atoms,mainy2mainz.transpose()*Vector2Matrix(atoms));
					}
				}else{ //  C2v has a proper axis, while Cs does not.
					pointgroup=Cs;
					if (Mirror(atoms,xaxis,tolerance)){
						atoms=Matrix2Vector(atoms,mainx2mainz.transpose()*Vector2Matrix(atoms));
					}else if (Mirror(atoms,yaxis,tolerance)){
						atoms=Matrix2Vector(atoms,mainy2mainz.transpose()*Vector2Matrix(atoms));
					}
				}
			}
		}else{ // Cs, C2v, C2h and D2h have at least one mirror respectively, while  C1, Ci, C2 and D2 do not.
			if (InversionCentre(atoms,tolerance)){ // Ci has an inversion centre, while others do not.
				pointgroup=Ci;
			}else if (ProperAxis(atoms,xaxis,2,tolerance) || ProperAxis(atoms,yaxis,2,tolerance) || ProperAxis(atoms,zaxis,2,tolerance)){ // C2 and D2 have at least one C2 axis, while C1 does not.
				if (ProperAxis(atoms,xaxis,2,tolerance) && ProperAxis(atoms,yaxis,2,tolerance) && ProperAxis(atoms,zaxis,2,tolerance)){ // D2 has three C2 axes along three principal axes, while C2 does not.
					pointgroup=D2;
				}else{ // D2 has three C2 axes along three principal axes, while C2 does not.
					pointgroup=C2;
					if (ProperAxis(atoms,xaxis,2,tolerance)){
						atoms=Matrix2Vector(atoms,mainx2mainz.transpose()*Vector2Matrix(atoms));
					}else if (ProperAxis(atoms,yaxis,2,tolerance)){
						atoms=Matrix2Vector(atoms,mainy2mainz.transpose()*Vector2Matrix(atoms));
					}
				}
			}else{ // C1 has no inversion centres, mirrors or proper axes.
				pointgroup=C1;
				atoms=Matrix2Vector(atoms,U*Vector2Matrix(atoms)); // If the molecular is asymmetric, putting it back to its original orientation.
			}
		}
	}else if ((ixx-iyy)*(ixx-iyy)<tolerance&&(ixx-izz)*(ixx-izz)<tolerance&&(iyy-izz)*(iyy-izz)<tolerance){ // Molecules of point groups Td, Oh and Ih have three equivalent eigen inertial moments.
		bool inversioncentre=InversionCentre(atoms,tolerance);
		if (inversioncentre){ // Among Td, Oh and Ih, Oh and Ih have an inversion centre respectively while Td does not.
			
		}else{
//			pointgroup.point_group_name="Td";
		}
	}else{ // Cn, Cnv, Cnh, Dn, Dnh and Sn with n>2 and Dnd with n>=2.
		EigenVector zaxis(0,0,1);
		if ((iyy-izz)*(iyy-izz)<tolerance){
			atoms=Matrix2Vector(atoms,mainx2mainz.transpose()*Vector2Matrix(atoms));
		}else if ((izz-ixx)*(izz-ixx)<tolerance){
			atoms=Matrix2Vector(atoms,mainy2mainz.transpose()*Vector2Matrix(atoms));
		}
		int manifolds[15]={2,3,4,5,6,8,9,10,12,15,18,20,24,30,36};
		int maxpropermanifold=0;
		int maximpropermanifold=0;
		for (int imanifold=0;imanifold<15;imanifold++){
			int manifold=manifolds[imanifold];
			if (ProperAxis(atoms,zaxis,manifold,tolerance)){
				maxpropermanifold=manifold;
			}
			if (ImproperAxis(atoms,zaxis,manifold,tolerance)){
				maximpropermanifold=manifold;
			}
		}
		if (maximpropermanifold==0){ // Cn, Cnv and Dn have no main improper axes.
			EigenVector horizontalc2axis=HorizontalC2Axis(atoms,tolerance);
			if (horizontalc2axis(0)!=0 || horizontalc2axis(1)!=0){ // Dn has horizontal C2 axes, while Cn and Cnv do not.
				switch(maxpropermanifold){
					case 3:pointgroup=D3;break;
//					case 4:pointgroup=D4;break;
//					case 5:pointgroup=D5;break;
//					case 6:pointgroup=D6;break;
//					case 8:pointgroup=D8;break;
//					case 9:pointgroup=D9;break;
//					case 10:pointgroup=D10;break;
//					case 12:pointgroup=D12;break;
//					case 15:pointgroup=D15;break;
//					case 18:pointgroup=D18;break;
//					case 20:pointgroup=D20;break;
//					case 24:pointgroup=D24;break;
//					case 30:pointgroup=D30;break;
//					case 36:pointgroup=D36;break;
				}
				EigenMatrix rotation{{horizontalc2axis(0),horizontalc2axis(1),0},
				                     {-horizontalc2axis(1),horizontalc2axis(0),0},
				                     {0,0,1}};
				atoms=Matrix2Vector(atoms,rotation*Vector2Matrix(atoms)); // Rotating the molecule so that x axis coincides with the first horizontal C2 axis.
			}else{  // Dn has horizontal C2 axes, while Cn and Cnv do not.
				EigenVector horizontalmirror=HorizontalMirror(atoms,tolerance);
				if (horizontalmirror(0)!=0 || horizontalmirror(1)!=0){ // Cnv has horizontal mirrors, while Cn does not.
					switch(maxpropermanifold){
						case 3:pointgroup=C3v;break;
						case 4:pointgroup=C4v;break;
//						case 5:pointgroup=C5v;break;
//						case 6:pointgroup=C6v;break;
//						case 8:pointgroup=C8v;break;
//						case 9:pointgroup=C9v;break;
//						case 10:pointgroup=C10v;break;
//						case 12:pointgroup=C12v;break;
//						case 15:pointgroup=C15v;break;
//						case 18:pointgroup=C18v;break;
//						case 20:pointgroup=C20v;break;
//						case 24:pointgroup=C24v;break;
//						case 30:pointgroup=C30v;break;
//						case 36:pointgroup=C36v;break;
					}
					EigenMatrix rotation{{-horizontalmirror(1),horizontalmirror(0),0},
				                             {-horizontalmirror(0),-horizontalmirror(1),0},
				                             {0,0,1}};
					atoms=Matrix2Vector(atoms,rotation*Vector2Matrix(atoms)); // Rotating the molecule so that x axis coincides with the first mirror.
				}else{ // Cnv has horizontal mirrors, while Cn does not.
					switch(maxpropermanifold){
						case 3:pointgroup=C3;break;
//						case 4:pointgroup=C4;break;
//						case 5:pointgroup=C5;break;
//						case 6:pointgroup=C6;break;
//						case 8:pointgroup=C8;break;
//						case 9:pointgroup=C9;break;
//						case 10:pointgroup=C10;break;
//						case 12:pointgroup=C12;break;
//						case 15:pointgroup=C15;break;
//						case 18:pointgroup=C18;break;
//						case 20:pointgroup=C20;break;
//						case 24:pointgroup=C24;break;
//						case 30:pointgroup=C30;break;
//						case 36:pointgroup=C36;break;
					}
				}
			}
		}else if (maxpropermanifold==maximpropermanifold){ // Max proper manifold and max improper manifold are the same for the main axis of Cnh and Dnh.
			EigenVector horizontalc2axis=HorizontalC2Axis(atoms,tolerance);
			if (horizontalc2axis(0)!=0 || horizontalc2axis(1)!=0){ // Dnh has horizontal C2 axes, while Cnh does not.
				switch(maxpropermanifold){
					case 3:pointgroup=D3h;break;
					case 4:pointgroup=D4h;break;
//					case 5:pointgroup=D5h;break;
//					case 6:pointgroup=D6h;break;
//					case 8:pointgroup=D8h;break;
//					case 9:pointgroup=D9h;break;
//					case 10:pointgroup=D10h;break;
//					case 12:pointgroup=D12h;break;
//					case 15:pointgroup=D15h;break;
//					case 18:pointgroup=D18h;break;
//					case 20:pointgroup=D20h;break;
//					case 24:pointgroup=D24h;break;
//					case 30:pointgroup=D30h;break;
//					case 36:pointgroup=D36h;break;
				}
				EigenMatrix rotation{{horizontalc2axis(0),horizontalc2axis(1),0},
				                     {-horizontalc2axis(1),horizontalc2axis(0),0},
				                     {0,0,1}};
				atoms=Matrix2Vector(atoms,rotation*Vector2Matrix(atoms)); // Rotating the molecule so that x axis coincides with the first horizontal C2 axis.
			}else{ // Dnh has horizontal C2 axes, while Cnh does not.
				switch(maxpropermanifold){
					case 3:pointgroup=C3h;break;
//					case 4:pointgroup=C4h;break;
//					case 5:pointgroup=C5h;break;
//					case 6:pointgroup=C6h;break;
//					case 8:pointgroup=C8h;break;
//					case 9:pointgroup=C9h;break;
//					case 10:pointgroup=C10h;break;
//					case 12:pointgroup=C12h;break;
//					case 15:pointgroup=C15h;break;
//					case 18:pointgroup=C18h;break;
//					case 20:pointgroup=C20h;break;
//					case 24:pointgroup=C24h;break;
//					case 30:pointgroup=C30h;break;
//					case 36:pointgroup=C36h;break;
				}
			}
		}else if (maxpropermanifold<maximpropermanifold){ // Max proper manifold is smaller that max improper manifold for the maxin axis of Dnd and Sn.
			EigenVector horizontalc2axis=HorizontalC2Axis(atoms,tolerance);
			if (horizontalc2axis(0)!=0 || horizontalc2axis(1)!=0){ // Dnd has horizontal C2 axes, while Sn does not.
				switch(maxpropermanifold){
					case 2:pointgroup=D2d;break;
					case 3:pointgroup=D3d;break;
					case 4:pointgroup=D4d;break;
//					case 5:pointgroup=D5d;break;
//					case 6:pointgroup=D6d;break;
//					case 8:pointgroup=D8d;break;
//					case 9:pointgroup=D9d;break;
//					case 10:pointgroup=D10d;break;
//					case 12:pointgroup=D12d;break;
//					case 15:pointgroup=D15d;break;
//					case 18:pointgroup=D18d;break;
//					case 20:pointgroup=D20d;break;
//					case 24:pointgroup=D24d;break;
//					case 30:pointgroup=D30d;break;
//					case 36:pointgroup=D36d;break;
				}
				EigenMatrix rotation{{horizontalc2axis(0),horizontalc2axis(1),0},
				                     {-horizontalc2axis(1),horizontalc2axis(0),0},
				                     {0,0,1}};
				atoms=Matrix2Vector(atoms,rotation*Vector2Matrix(atoms)); // Rotating the molecule so that x axis coincides with the first horizontal C2 axis.
			}else{ // Dnd has horizontal C2 axes, while Sn does not.
				switch(maximpropermanifold){
					case 4:pointgroup=S4;break;
//					case 6:pointgroup=S6;break;
//					case 8:pointgroup=S8;break;
//					case 10:pointgroup=S10;break;
//					case 12:pointgroup=S12;break;
//					case 18:pointgroup=S18;break;
//					case 20:pointgroup=S20;break;
//					case 24:pointgroup=S24;break;
//					case 30:pointgroup=S30;break;
//					case 36:pointgroup=S36;break;
				}
			}
		}
	}
	return pointgroup;
}

EigenMatrix Images(std::vector<libint2::Atom> atoms,PointGroup pointgroup,double tolerance){ // Image of each symmetry operation applied to each atom. For example, the image matrix of a three-atom molecule belonging to a point group with three symmetry operation may write as [[0,1,2],[1,2,0,],[2,0,1]]. The first row means that atom 0 is moved to itself by identity operator, to atom 1 by the second operator and to atom 2 by the third operator. Order of symmetry operations: identity, inversion, reflection, proper rotation, improper rotation; main axis first, then secondary ones.
	int natoms=atoms.size();
	int ninversioncentres=pointgroup.inversion_centre;
	int nmirrors=pointgroup.mirrors.rows();
	int npropers=pointgroup.proper_axes.rows();
	int nimpropers=pointgroup.improper_axes.rows();
	int noperations=1+ninversioncentres+nmirrors+npropers+nimpropers; // The total number of symmetry operations, include identity.
	EigenMatrix coordinates=Vector2Matrix(atoms);
	EigenMatrix images(natoms,noperations);

	for (int iatom=0;iatom<natoms;iatom++){ // Idensity.
		images(iatom,0)=iatom;
	}

	std::vector<libint2::Atom> projections=Matrix2Vector(atoms,-coordinates);
	for (int iatom=0;iatom<natoms;iatom++){ // Inversion.
		if (!ninversioncentres){
			continue;
		}
		libint2::Atom atomi=atoms[iatom];
		images(iatom,1)=FindImage(atomi,projections,tolerance);
	}

	int ioperation=ninversioncentres?2:1; // The index of operation is two if the point group has an inversion operator, one otherwise.

	for (int imirror=0;imirror<nmirrors;imirror++,ioperation++){ // Reflection.
		double mirrorx=pointgroup.mirrors(imirror,0);
		double mirrory=pointgroup.mirrors(imirror,1);
		double mirrorz=pointgroup.mirrors(imirror,2);
		EigenMatrix reflection{{1-2*mirrorx*mirrorx,-2*mirrorx*mirrory,-2*mirrorx*mirrorz},
		                       {-2*mirrorx*mirrory,1-2*mirrory*mirrory,-2*mirrory*mirrorz},
		                       {-2*mirrorx*mirrorz,-2*mirrory*mirrorz,1-2*mirrorz*mirrorz}}; // Reflection matrix.
		std::vector<libint2::Atom> projections=Matrix2Vector(atoms,reflection*coordinates);
		for (int iatom=0;iatom<natoms;iatom++){
			libint2::Atom atomi=atoms[iatom];
			images(iatom,ioperation)=FindImage(atomi,projections,tolerance);
		}
	}

	for (int iproper=0;iproper<npropers;iproper++,ioperation++){ // Proper rotation.
		int manifold=pointgroup.proper_manifolds(iproper);
		double angle=360/(double)manifold/180*3.1415926536; // Converting degrees to radii.
		double axisx=pointgroup.proper_axes(iproper,0); // Three components of axial vector. The vector must be a unit vector.
		double axisy=pointgroup.proper_axes(iproper,1);
		double axisz=pointgroup.proper_axes(iproper,2);
		EigenMatrix properrotation{{axisx*axisx*(1-cos(angle))+cos(angle),axisx*axisy*(1-cos(angle))-axisz*sin(angle),axisx*axisz*(1-cos(angle))+axisy*sin(angle)},
		                           {axisx*axisy*(1-cos(angle))+axisz*sin(angle),axisy*axisy*(1-cos(angle))+cos(angle),axisy*axisz*(1-cos(angle))-axisx*sin(angle)},
		                           {axisx*axisz*(1-cos(angle))-axisy*sin(angle),axisy*axisz*(1-cos(angle))+axisx*sin(angle),axisz*axisz*(1-cos(angle))+cos(angle)}}; // Proper rotation matrix.
		std::vector<libint2::Atom> projections=Matrix2Vector(atoms,properrotation*coordinates);
		for (int iatom=0;iatom<natoms;iatom++){
			libint2::Atom atomi=atoms[iatom];
			images(iatom,ioperation)=FindImage(atomi,projections,tolerance);		
		}
	}

	for (int iimproper=0;iimproper<nimpropers;iimproper++,ioperation++){ // Improper rotation.
		int manifold=pointgroup.improper_manifolds(iimproper);
		double angle=360/(double)manifold/180*3.1415926536; // Converting degrees to radii.
		double axisx=pointgroup.improper_axes(iimproper,0); // Three components of axial vector. The vector must be a unit vector.
		double axisy=pointgroup.improper_axes(iimproper,1);
		double axisz=pointgroup.improper_axes(iimproper,2);
		EigenMatrix properrotation{{axisx*axisx*(1-cos(angle))+cos(angle),axisx*axisy*(1-cos(angle))-axisz*sin(angle),axisx*axisz*(1-cos(angle))+axisy*sin(angle)},
		                           {axisx*axisy*(1-cos(angle))+axisz*sin(angle),axisy*axisy*(1-cos(angle))+cos(angle),axisy*axisz*(1-cos(angle))-axisx*sin(angle)},
		                           {axisx*axisz*(1-cos(angle))-axisy*sin(angle),axisy*axisz*(1-cos(angle))+axisx*sin(angle),axisz*axisz*(1-cos(angle))+cos(angle)}}; // Proper rotation matrix.
		EigenMatrix reflection{{1-2*axisx*axisx,-2*axisx*axisy,-2*axisx*axisz},
		                       {-2*axisx*axisy,1-2*axisy*axisy,-2*axisy*axisz},
		                       {-2*axisx*axisz,-2*axisy*axisz,1-2*axisz*axisz}}; // Reflection matrix.
		EigenMatrix improperrotation=manifold%2==1?properrotation*reflection:properrotation;
		std::vector<libint2::Atom> projections=Matrix2Vector(atoms,improperrotation*coordinates); // Projection of molecule with respective to proper rotation.
		for (int iatom=0;iatom<natoms;iatom++){
			libint2::Atom atomi=atoms[iatom];
			images(iatom,ioperation)=FindImage(atomi,projections,tolerance);
		}
	}

	return images;
}


int main(){
	libint2::initialize();
	std::string xyz;
	std::cin>>xyz;
	std::ifstream input(xyz);
	std::vector<libint2::Atom> atoms=libint2::read_dotxyz(input);
	double tolerance=0.1;
	PointGroup pointgroup=GetPointGroup(atoms,tolerance);
	for (int i=0;i<atoms.size();i++){
		libint2::Atom atom=atoms[i];
		std::cout<<atom.atomic_number<<" "<<atom.x<<" "<<atom.y<<" "<<atom.z<<std::endl;
	}
	PrintPointGroup(pointgroup);
	EigenMatrix images=Images(atoms,pointgroup,tolerance);
	std::cout<<"Symmetry operation image list:"<<std::endl;
	std::cout<<images<<std::endl;
	libint2::finalize();
}
