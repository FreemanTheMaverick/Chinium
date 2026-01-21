class QuadraticObj: public Maniverse::Objective{ public:
	EigenMatrix A, B;

	QuadraticObj(EigenMatrix A, EigenMatrix B): A(A), B(B){};

	void Calculate(std::vector<EigenMatrix> X, int /*derivative*/) override{
		const EigenMatrix& x = X[0];
		Value = Dot(x, A + 0.5 * B * x);
		Gradient = { A + B * x };
	};

	std::vector<std::vector<EigenMatrix>> Hessian(std::vector<EigenMatrix> V) const override{
		return std::vector<std::vector<EigenMatrix>>{{ B * V[0] }};
	};
};
