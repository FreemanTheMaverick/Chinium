class Environment{ public:
	double Temperature = 0;
	double ChemicalPotential = 0;
	Environment(double t, double mu){
		this->Temperature = t;
		this->ChemicalPotential = mu;
	}
};
