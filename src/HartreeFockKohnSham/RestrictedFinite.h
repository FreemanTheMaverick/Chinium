#pragma once

#include <string>

#include "../Gateway.h"

#include "Restricted.h"

class RGC_SCF: public R_SCF{ public:
	double Temperature, ChemicalPotential;
	RGC_SCF(std::string inp): R_SCF(inp){
		Temperature = ReadTemperature(inp);
		ChemicalPotential = ReadChemicalPotential(inp);
		xc.Spin = 1;
	};
	void Calculate0() override;
	// Calculate1() of R_SCF is reused.
	void Calculate2() override;
	void PostProcess0() override{ __PostProcess0__(grand potential) };
};
