#pragma once

#include <string>

#include "../Job.h"
#include "../Representation.h"

#include "SelfConsistentField.h"
#include<iostream>

class RO_SCF: public Job, public RepRO, public SCF{ public:
	RO_SCF(std::string inp): Job(inp), RepRO(inp), SCF(inp, mwfn, int2c1e){
		if ( Np == Na && Np == Nb ) xc.Spin = 1;
		else xc.Spin = 2;
		Na -= Np;
		Nb -= Np;
	};
	virtual void Calculate0() override;
	virtual void PostProcess0() override{ __PostProcess0__(energy) };
	virtual void PostProcess1() override{ __PostProcess1__ };
	virtual void PostProcess2() override{ __PostProcess2__ };
};
