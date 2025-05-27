void GuessSCF(Multiwfn& mwfn, Int2C1E& int2c1e, Grid& grid, std::string guess, const bool output);
void HartreeFockKohnSham(Multiwfn& mwfn, Int2C1E& int2c1e, Int4C2E& int4c2e, ExchangeCorrelation& xc, Grid& grid, std::string scf, int output, int nthreads);
std::tuple<EigenMatrix, EigenMatrix> HFKSDerivative(Multiwfn& mwfn, Int2C1E& int2c1e, Int4C2E& int4c2e, ExchangeCorrelation& xc, Grid& grid, int derivative, int output, int nthreads);
