#pragma once

#include <string>

#include "../Macro.h"
#include "../Grid.h"

#include "RestrictedOpen.h"

class TwoDet: public RO_SCF{ public:
	int TwoDetType = 1;
	Grid grid2;
	TwoDet(std::string inp);
	void Calculate0() override;
};
