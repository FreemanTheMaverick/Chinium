#pragma once

#include <string>
#include <libmwfn.h>

#include "Integral.h"

class Representation{ public:
	Mwfn mwfn;
	int Np, Na, Nb;
	Int2C1E int2c1e;
	Representation(std::string inp);
};

class RepR: public Representation{ public:
	RepR(std::string inp);
};

class RepU: public Representation{ public:
	RepU(std::string inp);
};

class RepRO: public Representation{ public:
	RepRO(std::string inp);
};
