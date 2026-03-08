#pragma once

#include <string>

#include "../Gateway.h"

#include "Unrestricted.h"

class UGC_SCF: public U_SCF{ public:
	double Temperature, ChemicalPotential;
	UGC_SCF(std::string inp): U_SCF(inp){
		Temperature = ReadTemperature(inp);
		ChemicalPotential = ReadChemicalPotential(inp);
		xc.Spin = 2;
	};
	void Calculate0() override;
	void PostProcess0() override{ __PostProcess0__(grand potential) };
};
