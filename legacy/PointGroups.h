#include <Eigen/Dense> // Eigen::Matrix.

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;

struct PointGroup{ // A structure to represent the symmetry information of a point group.
	std::string name;
	bool inversion_centre; // Whether the origin is the inversion centre.
	EigenMatrix mirrors; // Normalized normal vector of mirrors, stored as [[nx1,ny1,nz1],[[nx2,ny2,nz2],...].
	EigenMatrix proper_axes; // Normalized vector of proper axes, stored as [[lx1,ly1,lz1],[[lx2,ly2,lz2],...].
	EigenMatrix proper_manifolds; // Manifolds of proper axes, stored as [m1,m2,...].
	EigenMatrix proper_exponents; // Exponents of proper axes, stored as [e1,e2,...].
	EigenMatrix improper_axes; // Similar to proper_axes.
	EigenMatrix improper_manifolds; // Similar to proper_manifolds.
	EigenMatrix improper_exponents; // Similar to proper_exponents.
};

extern PointGroup C1;
extern PointGroup Cs;
extern PointGroup Ci;
extern PointGroup C2;
extern PointGroup C2v;
extern PointGroup C2h;
extern PointGroup D2;
extern PointGroup D2h;
extern PointGroup D3;
extern PointGroup C3;
extern PointGroup C3h;
extern PointGroup C3v;
extern PointGroup C4v;
extern PointGroup D3h;
extern PointGroup D4h;
extern PointGroup D6h;
extern PointGroup D9h;
extern PointGroup D2d;
extern PointGroup D3d;
extern PointGroup D4d;
extern PointGroup S4;

void PrintPointGroup(PointGroup pointgroup); // Printing molecular symmetry information.

