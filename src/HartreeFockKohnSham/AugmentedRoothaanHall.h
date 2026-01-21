class AugmentedRoothaanHall{ public:
	int MaxSize;
	bool Verbose;
	std::deque<EigenMatrix> Ps, Gs;
	std::vector<EigenMatrix> Pdiffs, Gdiffs;
	EigenMatrix Tinv;

	AugmentedRoothaanHall(int max_size, bool verbose);
	void Append(EigenMatrix P, EigenMatrix G);
	EigenMatrix Hessian(EigenMatrix v) const;
};
