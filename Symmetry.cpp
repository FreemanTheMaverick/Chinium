#include <libint2.hpp>
#include <Eigen/Dense> // Eigen::Matrix.
#include <string>
#include <vector>
#include <iostream>

typedef Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;

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

EigenMatrix Vector2Matrix(std::vector<libint2::Atom> atoms){ // libint2 stores molecular geometries in the format of std::vector<libint2::Atom>, which is not suitable for matrix manipulation, so we need to convert them from std::vector<libint2::Atom> to EigenMatrix.
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

std::vector<libint2::Atom> StandardOrientation(std::vector<libint2::Atom> atoms){ // Moving the molecule to the standard orientation.
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
	EigenMatrix s=eigensolver.eigenvalues();
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
	return newatoms;
}

int main(){
	libint2::initialize();
	std::string xyz="f.xyz";
	std::ifstream input(xyz);
	std::vector<libint2::Atom> atoms=libint2::read_dotxyz(input);
	std::vector<libint2::Atom> newatoms=StandardOrientation(atoms);
	for (int i=0;i<atoms.size();i++){
		libint2::Atom atom=newatoms[i];
		std::cout<<atom.atomic_number<<" "<<atom.x<<" "<<atom.y<<" "<<atom.z<<std::endl;
	}
	libint2::finalize();
}
