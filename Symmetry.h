#include <libint2.hpp>
#include <Eigen/Dense> // Eigen::Matrix.
#include <vector>
#include "PointGroups.h"

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;
typedef Eigen::Matrix<double,3,1> EigenVector;

double AtomicMass(int atomic_number); // Mapping atomic numbers to atomic masses.
EigenMatrix Vector2Matrix(std::vector<libint2::Atom> atoms); // libint2 stores molecular geometries in the format of std::vector<libint2::Atom>, which is not suitable for matrix manipulation, so it is necessary to convert them from std::vector<libint2::Atom> to EigenMatrix.
std::vector<libint2::Atom> Matrix2Vector(std::vector<libint2::Atom> atoms,EigenMatrix coordinates); // Converting molecular geometries from EigenMatrix back to std::vector<libint2::Atom>, so that libint2 can read them. Since EigenMatrix does not store atomic numbers, the original std::vector<libint2::Atom> is needed here for atomic numbers.

int FindImage(libint2::Atom atomi,std::vector<libint2::Atom> projections,double tolerance); // Finding the image of atom i among atom projections corresponding to some symmetry operation.

bool InversionCentre(std::vector<libint2::Atom> atoms,double tolerance); // Testing whether the origin is the inversion centre. For a molecule on its standard orientation, the origin is the only point that may be its inversion centre.
bool Mirror(std::vector<libint2::Atom> atoms,EigenVector normal,double tolerance); // Checking if the plane with the given normal vector is one of the mirrors of the molecule.
bool ProperAxis(std::vector<libint2::Atom> atoms,EigenVector axis,int manifold,double tolerance); // Determining if the given axis is one of the molecule's proper rotation axes.
bool ImproperAxis(std::vector<libint2::Atom> atoms,EigenVector axis,int manifold,double tolerance); // Determining if the given axis is one of the molecule's proper rotation axes.

EigenMatrix HorizontalC2Axis(std::vector<libint2::Atom>& atoms,double tolerance); // Finding one of the molecule's horizontal C2 axes, if any.
EigenMatrix HorizontalMirror(std::vector<libint2::Atom>& atoms,double tolerance); // Finding one of the molecule's horizontal mirror, if any.

PointGroup GetPointGroup(std::vector<libint2::Atom>& atoms,double tolerance); // Moving the molecule to the standard orientation.

EigenMatrix EquivalentAtoms(std::vector<libint2::Atom> atoms,PointGroup pointgroup,double tolerance); // Image of each symmetry operation applied to each atom. For example, the image matrix of a three-atom molecule belonging to a point group with three symmetry operation may write as [[0,1,2],[1,2,0,],[2,0,1]]. The first row means that atom 0 is moved to itself by identity operator, to atom 1 by the second operator and to atom 2 by the third operator. Order of symmetry operations: identity, inversion, reflection, proper rotation, improper rotation; main axis first, then secondary ones; main axis exponent 1 first, then increasing by one (for proper rotations) or two (for improper rotations) at a time.

EigenMatrix ShellCentres(libint2::BasisSet obs, std::vector<libint2::Atom> atoms); // Getting the centre of each shell. For example, [5,3,1,1,...] means that the first shell is centred at atom 5, the second atom 3 and the third and the fourth atom 1, etc.
EigenMatrix EquivalentShells(libint2::BasisSet obs,std::vector<libint2::Atom> atoms,PointGroup pointgroup,double tolerance); // Finding equivalent shells of the molecule. A set of shells are equivalent if their centres can be projected to each other by symmetry operations and their angular momenta are equal.
