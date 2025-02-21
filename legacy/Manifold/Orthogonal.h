#include "Manifold.h"

class Orthogonal: public Manifold{ public:
	Orthogonal(EigenMatrix p);

	int getDimension();
	double Inner(EigenMatrix X, EigenMatrix Y);
	std::function<double (EigenMatrix, EigenMatrix)> getInner();
	double Distance(EigenMatrix q);

	EigenMatrix Exponential(EigenMatrix X);
	EigenMatrix Logarithm(EigenMatrix q);

	EigenMatrix TangentProjection(EigenMatrix A);
	EigenMatrix TangentPurification(EigenMatrix A);

	EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y);
	EigenMatrix TransportManifold(EigenMatrix X, EigenMatrix q);

	void Update(EigenMatrix p, bool purify);
	void getGradient();
	void getHessian();
};
