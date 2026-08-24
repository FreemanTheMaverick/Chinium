#pragma once

#include <string>

#include "../Macro.h"
#include "../Grid.h"

#include "Restricted.h"

class TwoDet: public R_SCF{ public:
	int TwoDetType = 1;
	std::vector<std::array<EigenMatrix, 2>> lowers2;
	Grid grid2;
	TwoDet(std::string inp);
	void Calculate0() override;
};
