class Int2C1E{ public:
	Mwfn* MWFN;

	// Zeroth order
	EigenMatrix Overlap;
	EigenMatrix Kinetic;
	EigenMatrix Nuclear;
	EigenMatrix DipoleX;
	EigenMatrix DipoleY;
	EigenMatrix DipoleZ;
	EigenMatrix QuadrapoleXX;
	EigenMatrix QuadrapoleXY;
	EigenMatrix QuadrapoleXZ;
	EigenMatrix QuadrapoleYY;
	EigenMatrix QuadrapoleYZ;
	EigenMatrix QuadrapoleZZ;

	// First order
	std::vector<EigenMatrix> OverlapGrads;
	std::vector<EigenMatrix> KineticGrads;
	std::vector<EigenMatrix> NuclearGrads;

	// Second order
	std::vector<std::vector<EigenMatrix>> OverlapHesss;
	std::vector<std::vector<EigenMatrix>> KineticHesss;
	std::vector<std::vector<EigenMatrix>> NuclearHesss;

	Int2C1E(Mwfn& mwfn);
	void CalculateIntegrals(int order, int output);
	std::tuple<
		std::vector<double>,
		std::vector<double>,
		std::vector<double>
	> ContractGrads(EigenMatrix D, EigenMatrix W, int output);
	std::tuple<
		std::vector<std::vector<double>>,
		std::vector<std::vector<double>>,
		std::vector<std::vector<double>>
	> ContractGrads(std::vector<EigenMatrix>& Ds, std::vector<EigenMatrix>& Ws, int output);
	std::tuple<
		std::vector<std::vector<double>>,
		std::vector<std::vector<double>>,
		std::vector<std::vector<double>>
	> ContractHesss(EigenMatrix D, EigenMatrix W, int output);
};
