EigenMatrix CDIIS(std::deque<EigenMatrix>& Gs, std::deque<EigenMatrix>& Fs);

EigenMatrix ADIIS(EigenVector A, EigenMatrix B, std::deque<EigenMatrix>& Fs);
