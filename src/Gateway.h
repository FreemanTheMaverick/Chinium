std::vector<std::vector<double>> ReadXYZ(std::string inp);

std::string ReadBasisSet(std::string inp);

bool isInt(double x);

std::tuple<double, double> ReadNumElectrons(std::string inp);

int ReadWfnType(std::string inp);

int ReadNumThreads(std::string inp);

std::string ReadJobType(std::string inp);

std::string ReadSCF(std::string inp);

std::string ReadGuess(std::string inp);

std::string ReadGrid(std::string inp);

std::string ReadMethod(std::string inp);

int ReadDerivative(std::string inp);

double ReadTemperature(std::string inp);

double ReadChemicalPotential(std::string inp);
