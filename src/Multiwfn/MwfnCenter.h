class MwfnCenter{ public:
	int Index = -114;
	double Nuclear_charge = -114;
	std::vector<double> Coordinates = {114, 514, 1919810};
	std::vector<MwfnShell> Shells = {};
	int getNumShells();
	int getNumBasis();
	std::string getSymbol();
	void Print();
};
