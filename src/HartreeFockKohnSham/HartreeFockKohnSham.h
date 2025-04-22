void GuessSCF(Multiwfn& mwfn, Int2C1E& int2c1e, std::string guess, const bool output);
void HartreeFockKohnSham(Multiwfn& mwfn, Int2C1E& int2c1e, Int4C2E& int4c2e, std::string scf, int output, int nthreads);
std::tuple<EigenMatrix, EigenMatrix> HFKSDerivative(Multiwfn& mwfn, Int2C1E& int2c1e, Int4C2E& int4c2e, int derivative, int output, int nthreads);
