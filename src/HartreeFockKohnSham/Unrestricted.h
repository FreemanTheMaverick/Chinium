#pragma once

#include <string>

#include "../Job.h"
#include "../Representation.h"

#include "SelfConsistentField.h"

class U_SCF: public Job, public RepU, public SCF{ public:
	U_SCF(std::string inp): Job(inp), RepU(inp), SCF(inp, mwfn, int2c1e){
		xc.Spin = 2;
	};
	virtual void Calculate0() override;
	virtual void PostProcess0() override{ __PostProcess0__(energy) };
	virtual void PostProcess1() override{ __PostProcess1__ };
	virtual void PostProcess2() override{ __PostProcess2__ };
};

