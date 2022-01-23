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

struct MolecularSymmetry{
	std::string point_group_name;
	bool inversion_centre;
	EigenMatrix mirrors;
	EigenMatrix proper_axes;
	EigenMatrix proper_manifords;
	EigenMatrix improper_axes;
	EigenMatrix improper_manifords;
};

bool InversionCentre(std::vector<libint2::Atom> atoms,double tolerance){ // Testing whether the origin is the inversion centre. For a molecule on its standard original, the origin is the only point that may be its inversion centre.
	double deviation=0;
	int natoms=atoms.size();
	for (int iatom=0;iatom<natoms;iatom++){
		libint2::Atom atomi=atoms[iatom];
                int Zi=atomi.atomic_number;
		double xi=atomi.x;
		double yi=atomi.y;
		double zi=atomi.z;
		double nearestxj=114514;
		double nearestyj=1919810;
		double nearestzj=889464;
		double nearestdeviationsquared=364364;
		for (int jatom=0;jatom<natoms;jatom++){ // Finding the closest atom to the projection of atom i.
			libint2::Atom atomj=atoms[jatom];
                	int Zj=atomj.atomic_number;
			if (Zi==Zj){
				double xj=atomj.x;
				double yj=atomj.y;
				double zj=atomj.z;
				double nearestdeviationsquaredtrial=(xi+xj)*(xi+xj)*+(yi+yj)*(yi+yj)+(zi+zj)*(zi+zj);
				if (nearestdeviationsquared>nearestdeviationsquaredtrial){ // Whether this atom j is closer than last atom j to atom i.
					nearestxj=xj;
					nearestyj=yj;
					nearestzj=zj;
					nearestdeviationsquared=nearestdeviationsquaredtrial;
				}
			}
		}
		double distancesquared=(xi-nearestxj)*(xi-nearestxj)*+(yi-nearestyj)*(yi-nearestyj)+(zi-nearestzj)*(zi-nearestzj);
		deviation=deviation+nearestdeviationsquared/(distancesquared+1e-10); // Deviation is scaled by the distance between atom i and atom j. More deviation is tolerated considering the large interatomic distance. The 1e-10 is for preventing the denominator from dropping to zero in the case of an atom lying at the centre of mass.
	}
	deviation=deviation/natoms;
	bool inversioncentre;
	if (deviation<tolerance){
		inversioncentre=true;
	}else{
		inversioncentre=false;
	}
	return inversioncentre;
}

bool Mirror(std::vector<libint2::Atom> atoms,EigenVector normal,double tolerance){ // Checking if the plane with the given normal vector is one of the mirrors of the molecule.
	tolerance=tolerance*tolerance;
	int natoms=atoms.size();
	bool mirror=true;
	for (int iatom=0;iatom<natoms && mirror;iatom++){ // Once one of the atoms cannot find its mirror image, the loop is over.
		int Zi=atoms[iatom].atomic_number;
		double xi=atoms[iatom].x;
		double yi=atoms[iatom].y;
		double zi=atoms[iatom].z;
		EigenVector coordinatesi(xi,yi,zi);
		double nearestxj=114514;
		double nearestyj=1919810;
		double nearestzj=889464;
		EigenVector nearestcoordinatesj(nearestxj,nearestyj,nearestzj);
		double nearestdeviationsquared=364364;
		double distance=coordinatesi.dot(normal); // Distance between atom and plane.
		if (distance*distance<tolerance){ // If the atom is in the plane, this atom will be skipped.
			continue;
		}
		EigenVector coordinatesiprojection=coordinatesi-2*distance*normal; // Mirror image of atom.
		for (int jatom=0;jatom<natoms;jatom++){
			int Zj=atoms[jatom].atomic_number;
			if (Zi==Zj){
				double xj=atoms[jatom].x;
				double yj=atoms[jatom].y;
				double zj=atoms[jatom].z;
				EigenVector coordinatesj(xj,yj,zj);
				EigenVector deviationvector=coordinatesiprojection-coordinatesj;
				double deviationsquared=deviationvector.dot(deviationvector); // Distance between atom i's mirror image and atom j.
				if (deviationsquared<nearestdeviationsquared){ // Once a nearer neighbour to atom i's mirror image is found, update the nearest neighbour to it.
					nearestdeviationsquared=deviationsquared;
					nearestcoordinatesj=coordinatesj;
				}
			}
		}
		EigenVector interatomicdistancevector=coordinatesi-nearestcoordinatesj;
		double interatomicdistancesquared=interatomicdistancevector.dot(interatomicdistancevector); // Distance between atom i and atom j.
		mirror=nearestdeviationsquared/interatomicdistancesquared<tolerance;
	}
	return mirror;
}

bool ProperAxis(std::vector<libint2::Atom> atoms,EigenVector axis,int manifold,double tolerance){ // Determining if the given axis is one of the molecule's proper rotation axes.
	bool properaxis=true;
	int natoms=atoms.size();
	double angle=360/(double)manifold/180*3.1415926536; // Converting degrees to radii.
	EigenMatrix coordinates=Vector2Matrix(atoms);
	double axisx=axis(0,0); // Three components of axial vector. The vector must be a unit vector.
	double axisy=axis(1,0);
	double axisz=axis(2,0);
	for (int i=1;i<manifold && properaxis;i++){ // Looping over all possible rotation angles. For example, for a C_3 axis, the possible possible angles are 2pi/3, 4pi/3 and 2pi.
		EigenMatrix rotation(3,3);
		rotation<<axisx*axisx*(1-cos(i*angle))+cos(i*angle),axisx*axisy*(1-cos(i*angle))-axisz*sin(i*angle),axisx*axisz*(1-cos(i*angle))+axisy*sin(i*angle),
		          axisx*axisy*(1-cos(i*angle))+axisz*sin(i*angle),axisy*axisy*(1-cos(i*angle))+cos(i*angle),axisy*axisz*(1-cos(i*angle))-axisx*sin(i*angle),
		          axisx*axisz*(1-cos(i*angle))-axisy*sin(i*angle),axisy*axisz*(1-cos(i*angle))+axisx*sin(i*angle),axisz*axisz*(1-cos(i*angle))+cos(i*angle); // Rotation matrix.
		EigenMatrix atomprojection=rotation*coordinates; // Projection of molecule with respective to rotation.
		for (int iatom=0;iatom<natoms && properaxis;iatom++){
			double xiprojection=atomprojection(0,iatom);
			double yiprojection=atomprojection(1,iatom);
			double ziprojection=atomprojection(2,iatom);
			libint2::Atom atomi=atoms[iatom];
			int Zi=atomi.atomic_number;
			double xi=atomi.x;
			double yi=atomi.y;
			double zi=atomi.z;
			EigenVector coordinatesi(xi,yi,zi); // EigenMatrix is not suitable for vector multiplication. EigenVector must be used.
			double pointaxisdistancesquared=axis.cross(coordinatesi).dot(axis.cross(coordinatesi))/(axis.dot(axis)*axis.dot(axis)); // Calculating the distance between atom i and the axis. If the distance is small, the atom can be invariant to the rotation, and will not be considered in validity evaluation of this symmetry operation.
			if (pointaxisdistancesquared<tolerance){
				continue;
			}
			double nearestxj=114514;
			double nearestyj=1919810;
			double nearestzj=889464;
			double nearestdeviationsquared=364364;
			for (int jatom=0;jatom<natoms;jatom++){ // Finding the closest atom to the projection of atom i.
				libint2::Atom atomj=atoms[jatom];
                		int Zj=atomj.atomic_number;
				if (Zi==Zj&&iatom!=jatom){
					double xj=atomj.x;
					double yj=atomj.y;
					double zj=atomj.z;
					double deviationsquared=(xiprojection-xj)*(xiprojection-xj)+(yiprojection-yj)*(yiprojection-yj)+(ziprojection-zj)*(ziprojection-zj);
					if (deviationsquared<nearestdeviationsquared){
						nearestxj=xj;
						nearestyj=yj;
						nearestzj=zj;
						nearestdeviationsquared=deviationsquared;
					}
				}
			}
			double distancesquared=(xi-nearestxj)*(xi-nearestxj)+(yi-nearestyj)*(yi-nearestyj)+(zi-nearestzj)*(zi-nearestzj);
			properaxis=nearestdeviationsquared/(distancesquared+1e-10)<tolerance*tolerance; // Deviation is scaled by the distance between atom i and atom j. More deviation is tolerated considering the large interatomic distance. The 1e-10 is for preventing the denominator from dropping to zero in the case of an atom lying at the centre of mass.
		}
	}
	return properaxis;
}

MolecularSymmetry PointGroup(std::vector<libint2::Atom>& atoms,double tolerance){ // Moving the molecule to the standard orientation.
	int natoms=atoms.size();
	tolerance=tolerance*tolerance;
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

	tolerance=tolerance*tolerance;
	ixx=I(0,0);
	iyy=I(0,1);
	izz=I(0,2);
	MolecularSymmetry molecularsymmetry;
	if (ixx*ixx<tolerance||iyy*iyy<tolerance||izz*izz<tolerance){ // Molecules of point groups Cv and Dh have one zero inertial moment.
		bool inversioncentre=InversionCentre(atoms,tolerance);
		if (inversioncentre){
			molecularsymmetry.point_group_name="Dh";
		}else{
			molecularsymmetry.point_group_name="Cv";
		}
		molecularsymmetry.inversion_centre=inversioncentre;
	}else if ((ixx-iyy)*(ixx-iyy)<tolerance&&(ixx-izz)*(ixx-izz)<tolerance&&(iyy-izz)*(iyy-izz)<tolerance){ // Molecules of point groups Td, Oh and Ih have three equivalent eigen inertial moments.
		bool inversioncentre=InversionCentre(atoms,tolerance);
	}
	return molecularsymmetry;
}

int main(){
	libint2::initialize();
	std::string xyz;
	std::cin>>xyz;
	std::ifstream input(xyz);
	std::vector<libint2::Atom> atoms=libint2::read_dotxyz(input);
	//std::vector<libint2::Atom> newatoms=PreStandardOrientation(atoms);
	for (int i=0;i<atoms.size();i++){
		libint2::Atom atom=atoms[i];
		std::cout<<atom.atomic_number<<" "<<atom.x<<" "<<atom.y<<" "<<atom.z<<std::endl;
	}
	//MolecularSymmetry a=PointGroup(atoms);
	EigenVector axis(0,0,1);
//	axis<<0,0.707106781,0.707106781;
	bool proper=ProperAxis(atoms,axis,3,0.01);
	bool mirror=Mirror(atoms,axis,0.01);
	std::cout<<proper<<" "<<mirror;
	libint2::finalize();
}
