#pragma once

#include <Eigen/Core>
#include <vector>
#include <string>

#include "../Macro.h"
#include "../Job.h"
#include "../Representation.h"

#include "SelfConsistentField.h"

class R_SCF: public Job, public RepR, public SCF{ public:
	std::vector<EigenVector> dEs;
	std::vector<EigenMatrix> dFs; // For reusing the intermediate CPSCF results of R_SCF in RGC_SCF
    R_SCF(std::string inp): Job(inp), RepR(inp), SCF(inp, mwfn, int2c1e){
		xc.Spin = 1;
	};
	virtual void Calculate0() override;
	virtual void Calculate1() override;
	virtual void Calculate2() override;
	virtual void PostProcess0() override{ __PostProcess0__(energy) };
	virtual void PostProcess1() override{ __PostProcess1__ };
	virtual void PostProcess2() override{ __PostProcess2__ };
};
