#pragma once

#include <string>
#include <libmwfn.h>

#include "../Integral.h"
#include "../Grid.h"

void GuessSCF(Mwfn& mwfn, Int2C1E& int2c1e, Grid& grid, std::string guess, const bool output);
