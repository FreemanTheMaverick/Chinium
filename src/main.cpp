#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>
#include <cstdio>

#include "Macro.h"
#include "Multiwfn.h" // Requires <Eigen/Dense>, <vector>, <string>, "Macro.h".

int main(){
	
	Multiwfn mwfn=Multiwfn("h2o.mwfn",1);
	mwfn.E_tot = 0;
	mwfn.Gradient = EigenZero(mwfn.getNumCenters(), 3);
	mwfn.Hessian = EigenZero(3*mwfn.getNumCenters(), 3*mwfn.getNumCenters());

	mwfn.NuclearRepulsion({0}, 1);
	mwfn.getTwoCenter({0, 1}, 1);
	mwfn.getRepulsion({0}, -1., 8, 1);
	mwfn.GenerateGrid("SG-1",1);
	mwfn.getGridAO(1,1);
	mwfn.XC.Read("b3lyp", 1);
	mwfn.PrepareXC("ev",1);
	mwfn.GuessSCF("sap");
	mwfn.HartreeFockKohnSham(0,0,2,8);
	std::printf("Total energy: %17.10f\n", mwfn.E_tot);

	mwfn.NuclearRepulsion({1}, 1);
	mwfn.getRepulsion({1}, -1., 8, 1);
	mwfn.HFKSGradient();
	std::printf("Total Nuclear gradient:\n");
	for ( int iatom = 0; iatom < mwfn.getNumCenters(); iatom++ )
		std::printf("| %3d  %2s  % 10.17f  % 10.17f  % 10.17f\n", iatom, mwfn.Centers[iatom].getSymbol().c_str(), mwfn.Gradient(iatom, 0), mwfn.Gradient(iatom, 1), mwfn.Gradient(iatom, 2));
	//mwfn.Localize("Foster", "both", 2);
	//mwfn.Export("chainene_fb.mwfn", 1);
	//mwfn.Localize("Pipek", "both", 2);
	//mwfn.Export("h2o.mwfn", 1);
	return 0;
}

