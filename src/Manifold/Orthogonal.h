#include "Manifold.h"

class Orthogonal: public Manifold{ public:
	Orthogonal(EigenMatrix p);
	int getDimension();
	double Inner(EigenMatrix X, EigenMatrix Y);
	double Distance(EigenMatrix q);
	EigenMatrix Exponential(EigenMatrix X);
	EigenMatrix Logarithm(EigenMatrix q);
	EigenMatrix TangentProjection(EigenMatrix A);
	EigenMatrix TangentPurification(EigenMatrix A);
	void ManifoldPurification();
	void getGradient();
	void getHessian();
};
