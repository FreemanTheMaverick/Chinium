EigenMatrix CDIIS(std::deque<EigenMatrix>& Gs, std::deque<EigenMatrix>& Fs);

EigenMatrix ADIIS(EigenVector A, EigenMatrix B, std::deque<EigenMatrix>& Fs, int output);

static std::function<
			std::vector<EigenMatrix> // Multiple variables
			(
				std::vector<EigenVector>, // DIIS weights
				std::deque<std::vector<EigenMatrix>>&
			)
		> DefaultDiisUpdate = [](
				std::vector<EigenVector> W,
				std::deque<std::vector<EigenMatrix>>& After){
			const int diis_size = After.size();
			const int nmatrices = After[0].size();
			std::vector<EigenMatrix> Updated; Updated.reserve(nmatrices);
			for ( int i = 0; i < nmatrices; i++ ){
				const int nrows = After[0][i].rows();
				const int ncols = After[0][i].cols();
				EigenMatrix matrix = EigenZero(nrows, ncols);
				for ( int j = 0; j < diis_size; j++ )
					matrix += W[i][j] * After[j][i];
				Updated.push_back(matrix);
			}
			return Updated;
		};

bool GeneralizedDIIS(
		std::function<
			std::vector<std::tuple< // Multiple variables
					double, EigenMatrix, EigenMatrix // Objective, Residual, Next step
			>>
			(std::vector<EigenMatrix>&)
		>& RawUpdate,
		double tolerance, int max_size, int max_iter,
		double& E, std::vector<EigenMatrix>& M, int output,
		std::function<
			std::vector<EigenMatrix> // Multiple variables
			(
				std::vector<EigenVector>, // DIIS weights
				std::deque<std::vector<EigenMatrix>>&
			)
		>& DiisUpdate = DefaultDiisUpdate);
