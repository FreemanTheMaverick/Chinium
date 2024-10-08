class MwfnShell{ public:
	int Type = -114;
	std::vector<double> Exponents = {};
	std::vector<double> Coefficients = {};
	std::vector<double> NormalizedCoefficients = {};
	int getSize();
	int getNumPrims();
	void Print();
};
