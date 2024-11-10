class Manifold{ public:
	std::string Name;
	EigenMatrix P;
	EigenMatrix Aux;
	EigenMatrix Ge;
	EigenMatrix Gr;
	std::function<EigenMatrix (EigenMatrix)> He;
	std::function<EigenMatrix (EigenMatrix)> Hr;

	virtual int getDimension() = 0;
	virtual double Inner(EigenMatrix X, EigenMatrix Y) = 0;
	virtual std::function<double (EigenMatrix, EigenMatrix)> getInner() = 0;
	virtual double Distance(EigenMatrix q) = 0;

	virtual EigenMatrix Exponential(EigenMatrix X) = 0;
	virtual EigenMatrix Logarithm(EigenMatrix q) = 0;

	virtual EigenMatrix TangentProjection(EigenMatrix A) = 0;
	virtual EigenMatrix TangentPurification(EigenMatrix A) = 0;

	virtual EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) = 0;
	virtual EigenMatrix TransportManifold(EigenMatrix X, EigenMatrix q) = 0;

	virtual void Update(EigenMatrix p, bool purify) = 0;
	virtual void getGradient() = 0;
	virtual void getHessian() = 0;
};
