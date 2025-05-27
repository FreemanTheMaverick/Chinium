class ExchangeCorrelation{ public:
	std::vector<int> Codes;
	double EXX = 1;
	int Spin = 1; // 1 - Unpolarized, 2 - Polarized
	std::string Family;

	void Read(std::string df, bool output);
	void Evaluate(std::string order, Grid& grid);
	explicit operator bool() const{ return this->Codes.size() > 0; }
};
