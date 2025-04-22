std::vector<std::vector<EigenArray>> Matrices2Arrays(std::vector<EigenMatrix>& matrices, int nmatrices_redun);

std::vector<std::vector<EigenArray>> Matrices2Arrays(std::vector<EigenMatrix>& matrices);

std::vector<EigenMatrix> Arrays2Matrices(std::vector<std::vector<EigenArray>>& arrays, int nmatrices_noredun);

std::vector<EigenMatrix> Arrays2Matrices(std::vector<std::vector<EigenArray>>& arrays);

void MultipleMatrixReduction(
		std::vector<std::vector<EigenArray>>& omp_out,
		std::vector<std::vector<EigenArray>>& omp_in);

std::vector<std::vector<EigenArray>> MultipleMatrixInitialization(int nmatrices, int nrows, int ncols);
