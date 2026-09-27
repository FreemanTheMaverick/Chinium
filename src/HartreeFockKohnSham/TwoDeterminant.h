#pragma once

#include <string>

#include "../Macro.h"
#include "../Grid.h"

#include "Restricted.h"

class TwoDet: public R_SCF{ public:
	int TwoDetType = 1;
	Grid grid2;
	TwoDet(std::string inp);
	void Calculate0() override;
};
