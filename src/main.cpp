#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>
#include <cstdio>

#include "Macro.h"
#include "Multiwfn.h" // Requires <Eigen/Dense>, <vector>, <string>, "Macro.h".

int main(){
	
	Multiwfn mwfn=Multiwfn("chain_loc.mwfn",1);
	mwfn.getTwoCenter(0, 1);
	mwfn.getRepulsion(-1., 8, 1);
	mwfn.GenerateGrid("SG-1",0,1);
	mwfn.GuessSCF("sap");
	mwfn.HartreeFockKohnSham(0,0,1,8);
	mwfn.Localize("fock", "occ", 1);
	mwfn.Export("chain.mwfn", 1);
	return 0;
}

