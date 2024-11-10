#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>
#include <cstdio>

#include "Macro.h"
#include "Multiwfn.h" // Requires <Eigen/Dense>, <vector>, <string>, "Macro.h".

int main(){
	
	Multiwfn mwfn=Multiwfn("h2o.mwfn",1);
	mwfn.getTwoCenter(0, 1);
	mwfn.getRepulsion(-1., 8, 1);
	mwfn.GenerateGrid("SG-1",0,1);
	mwfn.GuessSCF("sap");
	mwfn.HartreeFockKohnSham(0,0,3,8);
	//mwfn.Localize("Foster", "both", 2);
	//mwfn.Export("chainene_fb.mwfn", 1);
	mwfn.Localize("Pipek", "both", 2);
	mwfn.Export("h2o.mwfn", 1);
	return 0;
}

