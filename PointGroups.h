#include <Eigen/Dense> // Eigen::Matrix.

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> EigenMatrix;

struct PointGroup{ // A structure to represent the symmetry information of a point group.
	std::string name;
	bool inversion_centre; // Whether the origin is the inversion centre.
	EigenMatrix mirrors; // Normalized normal vector of mirrors, stored as [[nx1,ny1,nz1],[[nx2,ny2,nz2],...].
	EigenMatrix proper_axes; // Normalized vector of proper axes, stored as [[lx1,ly1,lz1],[[lx2,ly2,lz2],...].
	EigenMatrix proper_manifolds; // Manifolds of proper axes, stored as [m1,m2,...].
	EigenMatrix improper_axes; // Similar to proper_axes.
	EigenMatrix improper_manifolds; // Similar to proper_manifolds.
};

PointGroup C1{
	name:"C1",
	inversion_centre:false
};

EigenMatrix csmirror{{0,0,1}};
PointGroup Cs{
	name:"Cs",
	inversion_centre:false,
	mirrors:csmirror
};

PointGroup Ci{
	name:"Ci",
	inversion_centre:true
};

EigenMatrix c2proper{{0,0,1}};
EigenMatrix c2propermanifold{{2}};
PointGroup C2{
	name:"C2",
	inversion_centre:false,
	proper_axes:c2proper, // Intel compiler reports a bug in this line, but GNU compiler does not. Maybe it is one of Intel compiler's bugs.
	proper_manifolds:c2propermanifold // Intel compiler reports a bug in this line, but GNU compiler does not. Maybe it is one of Intel compiler's bugs.
};

EigenMatrix c2vmirrors{{1,0,0},{0,1,0}};
EigenMatrix c2vproper{{0,0,1}};
EigenMatrix c2vpropermanifold{{2}};
PointGroup C2v{
	name:"C2v",
	inversion_centre:false,
	mirrors:c2vmirrors,
	proper_axes:c2vproper,
	proper_manifolds:c2vpropermanifold
};

EigenMatrix c2hmirrorandproper{{0,0,1}};
EigenMatrix c2hpropermanifold{{2}};
PointGroup C2h{
	name:"C2h",
	inversion_centre:true,
	mirrors:c2hmirrorandproper,
	proper_axes:c2hmirrorandproper,
	proper_manifolds:c2hpropermanifold
};

EigenMatrix d2propers{{0,0,1},{1,0,0},{0,1,0}};
EigenMatrix d2propermanifolds{{2,2,2}};
PointGroup D2{
	name:"D2",
	inversion_centre:false,
	proper_axes:d2propers, // Intel compiler reports a bug in this line, but GNU compiler does not. Maybe it is one of Intel compiler's bugs.
	proper_manifolds:d2propermanifolds // Intel compiler reports a bug in this line, but GNU compiler does not. Maybe it is one of Intel compiler's bugs.
};

EigenMatrix d2hmirrorsandpropers{{0,0,1},{1,0,0},{0,1,0}};
EigenMatrix d2hpropermanifolds{{2,2,2}};
PointGroup D2h{
	name:"D2h",
	inversion_centre:true,
	mirrors:d2hmirrorsandpropers,
	proper_axes:d2hmirrorsandpropers,
	proper_manifolds:d2hpropermanifolds
};

PointGroup D3{
	name:"D3",
	inversion_centre:false
};

PointGroup C3{
	name:"C3",
	inversion_centre:false
};

PointGroup C3h{
	name:"C3h",
	inversion_centre:false
};

PointGroup C3v{
	name:"C3v",
	inversion_centre:false
};

PointGroup C4v{
	name:"C4v",
	inversion_centre:false
};

PointGroup D3h{
	name:"D3h",
	inversion_centre:false
};

PointGroup D4h{
	name:"D4h",
	inversion_centre:true
};

PointGroup D2d{
	name:"D2d",
	inversion_centre:false
};

PointGroup D3d{
	name:"D3d",
	inversion_centre:true
};

PointGroup D4d{
	name:"D4d",
	inversion_centre:false
};

PointGroup S4{
	name:"S4",
	inversion_centre:false
};

void PrintPointGroup(PointGroup pointgroup){ // Printing molecular symmetry information.
	std::cout<<"Point group: "<<pointgroup.name<<std::endl;
	std::cout<<"Inversion Centre: "<<pointgroup.inversion_centre<<std::endl;
	std::cout<<"Mirrors: "<<pointgroup.mirrors<<std::endl;
	std::cout<<"Proper axes: "<<pointgroup.proper_axes<<std::endl;
	std::cout<<"Proper manifolds: "<<pointgroup.proper_manifolds<<std::endl;
	std::cout<<"Improper axes: "<<pointgroup.improper_axes<<std::endl;
	std::cout<<"Improper manifolds: "<<pointgroup.improper_manifolds<<std::endl;
}

