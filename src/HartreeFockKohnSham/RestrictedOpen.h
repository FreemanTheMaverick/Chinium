#pragma once

#include <string>

#include "../Job.h"
#include "../Representation.h"

#include "SelfConsistentField.h"

class RO_SCF: public Job, public RepRO, public SCF{ public:
	double Coupling = 0;
	RO_SCF(std::string inp);
	virtual void Calculate0() override;
	virtual void PostProcess0() override{ __PostProcess0__(energy) };
	virtual void PostProcess1() override{ __PostProcess1__ };
	virtual void PostProcess2() override{ __PostProcess2__ };
};
