#include <libint2.hpp>
#include <Eigen/Dense> // Eigen::Matrix.
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

struct MolecularSymmetry{ // A structure to represent the symmetry information of a molecule.
	std::string point_group_name;
	bool inversion_centre; // Whether the origin is the inversion centre.
	EigenMatrix mirrors; // Normalized normal vector of mirrors, stored as [[nx1,ny1,nz1],[[nx2,ny2,nz2],...].
	EigenMatrix proper_axes; // Normalized vector of proper axes, stored as [[lx1,ly1,lz1],[[lx2,ly2,lz2],...].
	EigenMatrix proper_manifolds; // Manifolds of proper axes, stored as [m1,m2,...].
	EigenMatrix improper_axes; // Similar to proper_axes.
	EigenMatrix improper_manifolds; // Similar to proper_manifolds.
};

void PrintMolecularSymmetry(MolecularSymmetry molecularsymmetry){
	std::cout<<"Point group: "<<molecularsymmetry.point_group_name<<std::endl;
	std::cout<<"Inversion Centre: "<<molecularsymmetry.inversion_centre<<std::endl;
	std::cout<<"Mirrors: "<<molecularsymmetry.mirrors<<std::endl;
	std::cout<<"Proper axes: "<<molecularsymmetry.proper_axes<<std::endl;
	std::cout<<"Proper manifolds: "<<molecularsymmetry.proper_manifolds<<std::endl;
	std::cout<<"Improper axes: "<<molecularsymmetry.improper_axes<<std::endl;
	std::cout<<"Improper manifolds: "<<molecularsymmetry.improper_manifolds<<std::endl;
}

bool InversionCentre(std::vector<libint2::Atom> atoms,double tolerance){ // Testing whether the origin is the inversion centre. For a molecule on its standard orientation, the origin is the only point that may be its inversion centre.
	int natoms=atoms.size();
	bool inversioncentre=true;
	for (int iatom=0;iatom<natoms && inversioncentre;iatom++){ // Once one atom cannot find its projection, the loop is over.
		libint2::Atom atomi=atoms[iatom];
                int Zi=atomi.atomic_number;
		double xi=atomi.x;
		double yi=atomi.y;
		double zi=atomi.z;
		EigenVector coordinatesi(xi,yi,zi); // Location of atom i.
		if (coordinatesi.dot(coordinatesi)<tolerance*tolerance){ // If atom i is at the origin, skipping it.
			continue;
		}
		EigenVector coordinatesiprojection=-coordinatesi; // The projection of atom i, which is on its opposite with respect to the origin.
		double nearestxj=114514; // The closest atom j to atom i's projection. Any arbitarily large numbers as initial values.
		double nearestyj=1919810;
		double nearestzj=889464;
		double nearestdeviationsquared=364364; // Deviation between the closest atom j and atom i's projection.
		EigenVector nearestcoordinatesj;
		for (int jatom=0;jatom<natoms;jatom++){ // Finding the closest atom j to atom i's projection.
			libint2::Atom atomj=atoms[jatom];
                	int Zj=atomj.atomic_number;
			if (Zi==Zj){ // Atom i and atom j must share the same atomic number if their locations are to be compared.
				double xj=atomj.x;
				double yj=atomj.y;
				double zj=atomj.z;
				EigenVector coordinatesj(xj,yj,zj); // Location of atom j.
				EigenVector deviationvector=coordinatesiprojection-coordinatesj; // Displacement vector between atom i's projection and the closest atom j.
				double deviationsquared=deviationvector.dot(deviationvector); // Square of distance between atom i's projection and the closest atom j.
				if (nearestdeviationsquared>deviationsquared){ // Whether this atom j is closer than last atom j to atom i.
					nearestcoordinatesj=coordinatesj; // If so, update the closest atom j and the corresponding deviation.
					nearestdeviationsquared=deviationsquared;
				}
			}
		}
		EigenVector interatomicdistancevector=coordinatesi-nearestcoordinatesj; // Displacement vector between atom i and the closest atom j.
		double interatomicdistancesquared=interatomicdistancevector.dot(interatomicdistancevector); // Square of distance between atom i and the closest atom j.
		inversioncentre=nearestdeviationsquared/interatomicdistancesquared<tolerance*tolerance; // Deviation is scaled by the distance between atom i and atom j. More deviation is tolerated considering the large interatomic distance. If the reduced deviation is larger than tolerance, the boolean will be toggled to false and the loop will be broken.
	}
	return inversioncentre;
}

bool Mirror(std::vector<libint2::Atom> atoms,EigenVector normal,double tolerance){ // Checking if the plane with the given normal vector is one of the mirrors of the molecule.
	int natoms=atoms.size();
	bool mirror=true;
	for (int iatom=0;iatom<natoms && mirror;iatom++){ // Once one of the atoms cannot find its projection, the loop is over.
		libint2::Atom atomi=atoms[iatom];
		int Zi=atomi.atomic_number;
		double xi=atomi.x;
		double yi=atomi.y;
		double zi=atomi.z;
		EigenVector coordinatesi(xi,yi,zi); // Location of atom i.
		double pointplanedistance=coordinatesi.dot(normal); // Distance between atom i and plane.
		if (pointplanedistance*pointplanedistance<tolerance*tolerance){ // If the atom is in the plane, this atom will be skipped.
			continue;
		}
		EigenVector coordinatesiprojection=coordinatesi-2*pointplanedistance*normal; // The projection of atom i, which is on its opposite with respect to the plane.
		double nearestxj=114514; // The closest atom j to atom i's projection. Any arbitarily large numbers as initial values.
		double nearestyj=1919810;
		double nearestzj=889464;
		double nearestdeviationsquared=364364; // Deviation between the closest atom j and atom i's projection.
		EigenVector nearestcoordinatesj;
		for (int jatom=0;jatom<natoms;jatom++){ // Finding the closest atom j to atom i's projection.
			libint2::Atom atomj=atoms[jatom];
			int Zj=atomj.atomic_number;
			if (Zi==Zj){
				double xj=atomj.x;
				double yj=atomj.y;
				double zj=atomj.z;
				EigenVector coordinatesj(xj,yj,zj); // Location of atom j.
				EigenVector deviationvector=coordinatesiprojection-coordinatesj; // Displacement vector between atom i's projection and the closest atom j.
				double deviationsquared=deviationvector.dot(deviationvector); // Square of distance between atom i's projection and the closest atom j.
				if (nearestdeviationsquared>deviationsquared){ // Whether this atom j is closer than last atom j to atom i.
					nearestcoordinatesj=coordinatesj; // If so, update the closest atom j and the corresponding deviation.
					nearestdeviationsquared=deviationsquared;
				}
			}
		}
		EigenVector interatomicdistancevector=coordinatesi-nearestcoordinatesj; // Displacement vector between atom i and the closest atom j.
		double interatomicdistancesquared=interatomicdistancevector.dot(interatomicdistancevector); // Square of distance between atom i and the closest atom j.
		mirror=nearestdeviationsquared/interatomicdistancesquared<tolerance*tolerance; // Deviation is scaled by the distance between atom i and atom j. More deviation is tolerated considering the large interatomic distance. If the reduced deviation is larger than tolerance, the boolean will be toggled to false and the loop will be broken.
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
		EigenMatrix rotation(3,3);
		rotation<<axisx*axisx*(1-cos(i*angle))+cos(i*angle),axisx*axisy*(1-cos(i*angle))-axisz*sin(i*angle),axisx*axisz*(1-cos(i*angle))+axisy*sin(i*angle),
		          axisx*axisy*(1-cos(i*angle))+axisz*sin(i*angle),axisy*axisy*(1-cos(i*angle))+cos(i*angle),axisy*axisz*(1-cos(i*angle))-axisx*sin(i*angle),
		          axisx*axisz*(1-cos(i*angle))-axisy*sin(i*angle),axisy*axisz*(1-cos(i*angle))+axisx*sin(i*angle),axisz*axisz*(1-cos(i*angle))+cos(i*angle); // Rotation matrix.
		EigenMatrix atomprojection=rotation*coordinates; // Projection of molecule with respective to rotation.
		for (int iatom=0;iatom<natoms && properaxis;iatom++){
			libint2::Atom atomi=atoms[iatom];
			int Zi=atomi.atomic_number;
			double xi=atomi.x;
			double yi=atomi.y;
			double zi=atomi.z;
			EigenVector coordinatesi(xi,yi,zi); // Location of atom i.
			double pointaxisdistancesquared=axis.cross(coordinatesi).dot(axis.cross(coordinatesi))/(axis.dot(axis)*axis.dot(axis)); // Calculating the distance between atom i and the axis. If the distance is small, the atom can be invariant to the rotation, and will not be considered in validity evaluation of this symmetry operation.
			if (pointaxisdistancesquared<tolerance*tolerance){
				continue;
			}
			double xiprojection=atomprojection(0,iatom);
			double yiprojection=atomprojection(1,iatom);
			double ziprojection=atomprojection(2,iatom);
			EigenVector coordinatesiprojection(xiprojection,yiprojection,ziprojection); // The projection of atom i, which is rendered by rotating atom i around axis by i*angle.
			double nearestxj=114514; // The closest atom j to atom i's projection. Any arbitarily large numbers as initial values.
			double nearestyj=1919810;
			double nearestzj=889464;
			double nearestdeviationsquared=364364; // Deviation between the closest atom j and atom i's projection.
			EigenVector nearestcoordinatesj;
			for (int jatom=0;jatom<natoms;jatom++){ // Finding the closest atom to the projection of atom i.
				libint2::Atom atomj=atoms[jatom];
                		int Zj=atomj.atomic_number;
				if (Zi==Zj){
					double xj=atomj.x;
					double yj=atomj.y;
					double zj=atomj.z;
					EigenVector coordinatesj(xj,yj,zj); // Location of atom j.
					EigenVector deviationvector=coordinatesiprojection-coordinatesj; // Displacement vector between atom i's projection and the closest atom j.
					double deviationsquared=deviationvector.dot(deviationvector); // Square of distance between atom i's projection and the closest atom j.
					if (deviationsquared<nearestdeviationsquared){ // Whether this atom j is closer than last atom j to atom i.
						nearestcoordinatesj=coordinatesj; // If so, update the closest atom j and the corresponding deviation.
						nearestdeviationsquared=deviationsquared;
					}
				}
			}
			EigenVector interatomicdistancevector=coordinatesi-nearestcoordinatesj; // Displacement vector between atom i and the closest atom j.
			double interatomicdistancesquared=interatomicdistancevector.dot(interatomicdistancevector); // Square of distance between atom i and the closest atom j.
			properaxis=nearestdeviationsquared/interatomicdistancesquared<tolerance*tolerance; // Deviation is scaled by the distance between atom i and atom j. More deviation is tolerated considering the large interatomic distance.
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
		EigenMatrix rotation(3,3);
		rotation<<axisx*axisx*(1-cos(i*angle))+cos(i*angle),axisx*axisy*(1-cos(i*angle))-axisz*sin(i*angle),axisx*axisz*(1-cos(i*angle))+axisy*sin(i*angle),
		          axisx*axisy*(1-cos(i*angle))+axisz*sin(i*angle),axisy*axisy*(1-cos(i*angle))+cos(i*angle),axisy*axisz*(1-cos(i*angle))-axisx*sin(i*angle),
		          axisx*axisz*(1-cos(i*angle))-axisy*sin(i*angle),axisy*axisz*(1-cos(i*angle))+axisx*sin(i*angle),axisz*axisz*(1-cos(i*angle))+cos(i*angle); // Rotation matrix.
		EigenMatrix atomprojection=rotation*coordinates; // Projection of molecule with respective to rotation.
		for (int iatom=0;iatom<natoms && improperaxis;iatom++){
			libint2::Atom atomi=atoms[iatom];
			int Zi=atomi.atomic_number;
			double xi=atomi.x;
			double yi=atomi.y;
			double zi=atomi.z;
			EigenVector coordinatesi(xi,yi,zi); // Location of atom i.
			double pointorigindistancesquared=coordinatesi.dot(coordinatesi); // Calculating the distance between atom i and the origin. If the distance is small, the atom can be invariant to the improper rotation, and will not be considered in validity evaluation of this symmetry operation.
			if (pointorigindistancesquared<tolerance*tolerance){
				continue;
			}
			double xiprojection=atomprojection(0,iatom);
			double yiprojection=atomprojection(1,iatom);
			double ziprojection=atomprojection(2,iatom);
			EigenVector coordinatesiprojection(xiprojection,yiprojection,ziprojection); // The projection of atom i, which is rendered by rotating atom i around axis by i*angle.
			if (i%2==1){ // An atom's mirror image of the rotation image is its true improper image if the index is odd. Otherwise, its proper image and improper image are identical.
		                double pointplanedistance=coordinatesi.dot(axis); // Distance between atom i and plane.
				coordinatesiprojection=coordinatesiprojection-2*pointplanedistance*axis;
			}
			double nearestxj=114514; // The closest atom j to atom i's projection. Any arbitarily large numbers as initial values.
			double nearestyj=1919810;
			double nearestzj=889464;
			double nearestdeviationsquared=364364; // Deviation between the closest atom j and atom i's projection.
			EigenVector nearestcoordinatesj;
			for (int jatom=0;jatom<natoms;jatom++){ // Finding the closest atom to the projection of atom i.
				libint2::Atom atomj=atoms[jatom];
                		int Zj=atomj.atomic_number;
				if (Zi==Zj&&iatom!=jatom){
					double xj=atomj.x;
					double yj=atomj.y;
					double zj=atomj.z;
					EigenVector coordinatesj(xj,yj,zj); // Location of atom j.
					EigenVector deviationvector=coordinatesiprojection-coordinatesj; // Displacement vector between atom i's projection and the closest atom j.
					double deviationsquared=deviationvector.dot(deviationvector); // Square of distance between atom i's projection and the closest atom j.
					if (deviationsquared<nearestdeviationsquared){ // Whether this atom j is closer than last atom j to atom i.
                                                nearestcoordinatesj=coordinatesj; // If so, update the closest atom j and the corresponding deviation.
                                                nearestdeviationsquared=deviationsquared;
					}
				}
			}
                        EigenVector interatomicdistancevector=coordinatesi-nearestcoordinatesj; // Displacement vector between atom i and the closest atom j.
                        double interatomicdistancesquared=interatomicdistancevector.dot(interatomicdistancevector); // Square of distance between atom i and the closest atom j.
                        improperaxis=nearestdeviationsquared/interatomicdistancesquared<tolerance*tolerance; // Deviation is scaled by the distance between atom i and atom j. More deviation is tolerated considering the large interatomic distance.
		}
	}
	return improperaxis;
}

MolecularSymmetry PointGroup(std::vector<libint2::Atom>& atoms,double tolerance){ // Moving the molecule to the standard orientation.
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
	MolecularSymmetry molecularsymmetry;
	if (ixx*ixx<tolerance*tolerance || iyy*iyy<tolerance*tolerance || izz*izz<tolerance*tolerance){ // Molecules of point groups Cv and Dh have one zero inertial moment.
		bool inversioncentre=InversionCentre(atoms,tolerance);
		molecularsymmetry.inversion_centre=inversioncentre;
		if (inversioncentre){ // Dh has an inversion centre while Cv does not.
			molecularsymmetry.point_group_name="Dh";
		}else{ // Dh has an inversion centre while Cv does not.
			molecularsymmetry.point_group_name="Cv";
		}
	}else if ((ixx-iyy)*(ixx-iyy)>tolerance*tolerance && (ixx-izz)*(ixx-izz)>tolerance*tolerance && (iyy-izz)*(iyy-izz)>tolerance*tolerance){ // Molecules of point groups C1, Cs, Ci, C2, C2v, C2h, D2h, D2d have three unequivalent eigen inertial moments.
		EigenVector xaxis(1,0,0);
		EigenVector yaxis(0,1,0);
		EigenVector zaxis(0,0,1); // The standard orientation procedure will align the molecule in the way that its main mirror (on which the most atoms lie among all mirrors, if any) lies on xy plane (except for D2d), so (0,0,1) is the normal vector of its main mirror.
		if (Mirror(atoms,xaxis,tolerance) || Mirror(atoms,yaxis,tolerance) || Mirror(atoms,zaxis,tolerance)){ // Cs, C2v, C2h and D2h have one mirror respectively on xy plane, while  C1, Ci, C2 and D2 do not.
			if (InversionCentre(atoms,tolerance)){ // C2h and D2h have an inversion centre, while Cs and C2v do not.
				if (Mirror(atoms,xaxis,tolerance) && Mirror(atoms,yaxis,tolerance) && Mirror(atoms,zaxis,tolerance)){ // D2h has another two mirrors on yz and zx plane, while C2h does not.
					molecularsymmetry.point_group_name="D2h";
std::cout<<"fuck1"<<std::endl;
					molecularsymmetry.inversion_centre=true;
					EigenMatrix mirrorsandpropers(3,3);mirrorsandpropers<<1,0,0,
					                                                      0,1,0,
					                                                      0,0,1;
					molecularsymmetry.mirrors=mirrorsandpropers;
					molecularsymmetry.proper_axes=mirrorsandpropers;
					EigenMatrix propermanifolds(1,3);propermanifolds<<2,2,2;molecularsymmetry.proper_manifolds=propermanifolds;
				}else{ // D2h has another two mirrors on yz and zx planes, while C2h does not.
					molecularsymmetry.point_group_name="C2h";
std::cout<<"fuck2"<<std::endl;
					molecularsymmetry.inversion_centre=true;
					EigenMatrix mirrorandproper(1,3);
					if (Mirror(atoms,xaxis,tolerance)){
						mirrorandproper<<1,0,0;
					}else if (Mirror(atoms,yaxis,tolerance)){
						mirrorandproper<<0,1,0;
					}else if (Mirror(atoms,zaxis,tolerance)){
						mirrorandproper<<0,0,1;
					}
					molecularsymmetry.mirrors=mirrorandproper;
					molecularsymmetry.proper_axes=mirrorandproper;
					EigenMatrix propermanifold(1,1);propermanifold<<2;molecularsymmetry.proper_manifolds=propermanifold;
				}
			}else{ // C2h and D2h have an inversion centre, while Cs and C2v do not.
				if (ProperAxis(atoms,xaxis,2,tolerance) || ProperAxis(atoms,yaxis,2,tolerance) || ProperAxis(atoms,zaxis,2,tolerance)){ // C2v has a proper axis, while Cs does not.
					molecularsymmetry.point_group_name="C2v";
std::cout<<"fuck3"<<std::endl;
					molecularsymmetry.inversion_centre=false;
					EigenMatrix mirrors(2,3);
					EigenMatrix proper(1,3);
					if (ProperAxis(atoms,xaxis,2,tolerance)){
						mirrors<<0,1,0,
						         0,0,1;
						proper<<1,0,0;
					}else if (ProperAxis(atoms,yaxis,2,tolerance)){
						mirrors<<0,0,1,
                                                         1,0,0;
                                                proper<<0,1,0;
                                        }else if (ProperAxis(atoms,zaxis,2,tolerance)){
                                                mirrors<<1,0,0,
                                                         0,1,0;
                                                proper<<0,0,1;
					}
					molecularsymmetry.mirrors=mirrors;
                                        molecularsymmetry.proper_axes=proper;
					EigenMatrix propermanifold(1,1);propermanifold<<2;molecularsymmetry.proper_manifolds=propermanifold;
				}else{ //  C2v has a proper axis, while Cs does not.
					molecularsymmetry.point_group_name="Cs";
std::cout<<"fuck4"<<std::endl;
					molecularsymmetry.inversion_centre=false;
					EigenMatrix mirror(1,3);
					if (Mirror(atoms,xaxis,tolerance)){
						mirror<<1,0,0;
					}else if (Mirror(atoms,yaxis,tolerance)){
						mirror<<0,1,0;
					}else if (Mirror(atoms,zaxis,tolerance)){
						mirror<<0,0,1;
					}
					molecularsymmetry.mirrors=mirror;
				}
			}
		}else{ // Cs, C2v, C2h and D2h have one mirror respectively on the xy plane, while  C1, Ci, C2 and D2 do not.
			if (InversionCentre(atoms,tolerance)){ // Ci has an inversion centre, while others do not.
				molecularsymmetry.point_group_name="Ci";
std::cout<<"fuck5"<<std::endl;
				molecularsymmetry.inversion_centre=true;
			}else if (ProperAxis(atoms,xaxis,2,tolerance) || ProperAxis(atoms,yaxis,2,tolerance) || ProperAxis(atoms,zaxis,2,tolerance)){ // C2 and D2 has at least one C2 axis, while C1 does not.
				if (ProperAxis(atoms,xaxis,2,tolerance) && ProperAxis(atoms,yaxis,2,tolerance) && ProperAxis(atoms,zaxis,2,tolerance)){ // D2 has three C2 axes along three principal axes, while C2 does not.
					molecularsymmetry.point_group_name="D2";
std::cout<<"fuck6"<<std::endl;
					molecularsymmetry.inversion_centre=false;
					EigenMatrix propers(3,3);propers<<1,0,0,
					                                  0,1,0,
					                                  0,0,1;
                                        molecularsymmetry.proper_axes=propers;
					EigenMatrix propermanifolds(1,3);propermanifolds<<2,2,2;molecularsymmetry.proper_manifolds=propermanifolds;
				}else{ // D2 has three C2 axes along three principal axes, while C2 does not.
					molecularsymmetry.point_group_name="C2";
std::cout<<"fuck7"<<std::endl;
					molecularsymmetry.inversion_centre=false;
					EigenMatrix proper(1,3);
					if (ProperAxis(atoms,xaxis,2,tolerance)){
						proper<<1,0,0;
					}else if (ProperAxis(atoms,yaxis,2,tolerance)){
						proper<<0,1,0;
					}else if (ProperAxis(atoms,zaxis,2,tolerance)){
						proper<<0,0,1;
					}
					molecularsymmetry.proper_axes=proper;
					EigenMatrix propermanifold(1,1);propermanifold<<2;molecularsymmetry.proper_manifolds=propermanifold;
				}
			}else{ // C1 has no inversion centres, mirrors or proper axes.
				molecularsymmetry.point_group_name="C1";
std::cout<<"fuck8"<<std::endl;
				molecularsymmetry.inversion_centre=false;
			}
		}
	}else if ((ixx-iyy)*(ixx-iyy)<tolerance&&(ixx-izz)*(ixx-izz)<tolerance&&(iyy-izz)*(iyy-izz)<tolerance){ // Molecules of point groups Td, Oh and Ih have three equivalent eigen inertial moments.
		bool inversioncentre=InversionCentre(atoms,tolerance);
		if (inversioncentre){ // Among Td, Oh and Ih, Oh and Ih have an inversion centre respectively while Td does not.
			
		}else{
			molecularsymmetry.point_group_name="Td";
		}
	}else{ // Cn, Cnv, Cnh, Dn, Dnh and Sn with n>2 and Dnd with n>=2.
	}
	return molecularsymmetry;
}

int main(){
	libint2::initialize();
	std::string xyz;
	std::cin>>xyz;
	std::ifstream input(xyz);
	std::vector<libint2::Atom> atoms=libint2::read_dotxyz(input);
	MolecularSymmetry a=PointGroup(atoms,0.01);
	for (int i=0;i<atoms.size();i++){
		libint2::Atom atom=atoms[i];
		std::cout<<atom.atomic_number<<" "<<atom.x<<" "<<atom.y<<" "<<atom.z<<std::endl;
	}
	PrintMolecularSymmetry(a);
	libint2::finalize();
}
