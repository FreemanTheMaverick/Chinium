#pragma once

#include <string>
#include <stdexcept>

#include "RestrictedOpen.h"

class TwoDet: public RO_SCF{ public:
	TwoDet(std::string inp): RO_SCF(inp){
		xc.Spin = 1;
		for ( auto& subgridbatch : grid.SubGridBatches ) for ( auto& subgrid : subgridbatch ) subgrid->Spin = 1;
		if ( Na != 1 && Nb != 1 ) throw std::runtime_error("Two-determinant ROKS requires exactly one unpaired alpha electron and one unpaired beta electron! (Keyword: spin 2 2)");
	};
	void Calculate0() override;
};
