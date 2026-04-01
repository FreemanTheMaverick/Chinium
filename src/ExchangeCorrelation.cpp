#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
extern "C"{
	#include <xc.h>
}
#include <sstream>
#include <fstream>
#include <cstdio>
#include <cassert>
#include <string>
#include <vector>
#include <libmwfn.h>

#include "Macro.h"
#include "Grid.h"
#include "ExchangeCorrelation.h"

void ExchangeCorrelation::Read(std::string df, bool output){

	// Parsing df file.
	std::string filename = (std::string)std::getenv("CHINIUM_PATH") + "/DensityFunctionals/" + df + ".df";
	std::ifstream file(filename);
	if (!file.good()) throw std::runtime_error("Density functional file is missing!");
	if (output) std::printf("Reading density functional file %s ...\n", filename.c_str());
	std::string thisline;

	std::getline(file, thisline);
	std::stringstream ss(thisline);
	int ifunc = 0;
	int type = 0;
	int code = 0;
	this->EXX = 0;
	if (output) std::printf("Density functional information:\n");
	while (ss >> code){
		if (ss.fail()) throw std::runtime_error("Corrupted density functional file!");
		xc_func_type func;
		xc_func_init(&func, code, this->Spin);
		std::string name = func.info->name;
		std::string kind;
		switch (func.info->kind){
			case (XC_EXCHANGE):
				kind = "Exchange"; break;
			case (XC_CORRELATION):
				kind = "Correlation"; break;
			case (XC_EXCHANGE_CORRELATION):
				kind = "Exchange-correlation"; break;
			case (XC_KINETIC):
				kind = "Kinetic"; break;
		}
		std::string family;
		switch (func.info->family){
			case (XC_FAMILY_LDA):
			case (XC_FAMILY_HYB_LDA):
				family = "LDA"; type = type > 0 ? type : 0; break;
			case (XC_FAMILY_GGA):
			case (XC_FAMILY_HYB_GGA):
				family = "GGA"; type = type > 1 ? type : 1; break;
			case (XC_FAMILY_MGGA):
			case (XC_FAMILY_HYB_MGGA):
				family = "mGGA"; type = type > 2 ? type : 2; break;
		}
		this->Codes.push_back(code);
		const double exx = xc_hyb_exx_coef(&func);
		if ( this->EXX < exx ) this->EXX = exx;
		if (output){
			std::printf("| Functional %d:\n", ifunc++);
			std::printf("| | Code: %d\n", code);
			std::printf("| | Name: %s\n", name.c_str());
			std::printf("| | Kind: %s\n", kind.c_str());
			std::printf("| | Family: %s\n", family.c_str());
			if ( func.info->kind == XC_EXCHANGE || func.info->kind == XC_EXCHANGE_CORRELATION )
				std::printf("| | Exact exchange component: %f\n", exx);
		}
		xc_func_end(&func);
	}
	switch (type){
		case (0):
			this->Family = "LDA"; break;
		case (1):
			this->Family = "GGA"; break;
		case (2):
			this->Family = "mGGA"; break;
	}
}

#define __XC_LDA__\
	if ( order == "e" ){\
		EigenVector eps(ngrids);\
		xc_lda_exc(&func, ngrids, rho.data(), eps.data());\
		Eps += eps;\
	}else if ( order == "v" ){\
		EigenMatrix eps1rho(Eps1Rho.rows(), ngrids);\
		xc_lda_vxc(&func, ngrids, rho.data(), eps1rho.data());\
		Eps1Rho += eps1rho;\
	}else if ( order == "ev" ){\
		EigenVector eps(ngrids);\
		EigenMatrix eps1rho(Eps1Rho.rows(), ngrids);\
		xc_lda_exc_vxc(&func, ngrids, rho.data(), eps.data(), eps1rho.data());\
		Eps += eps;\
		Eps1Rho += eps1rho;\
	}else if ( order == "f" ){\
		EigenMatrix eps2rho2(Eps2Rho2.rows(), ngrids);\
		xc_lda_fxc(&func, ngrids, rho.data(), eps2rho2.data());\
		Eps2Rho2 += eps2rho2;\
	}

#define __XC_GGA__\
	if ( order == "e" ){\
		EigenVector eps(ngrids);\
		xc_gga_exc(&func, ngrids, rho.data(), sigma.data(), eps.data());\
		Eps += eps;\
	}else if ( order == "v" ){\
		EigenMatrix eps1rho(Eps1Rho.rows(), ngrids);\
		EigenMatrix eps1sigma(Eps1Sigma.rows(), ngrids);\
		xc_gga_vxc(&func, ngrids, rho.data(), sigma.data(), eps1rho.data(), eps1sigma.data());\
		Eps1Rho += eps1rho;\
		Eps1Sigma += eps1sigma;\
	}else if ( order == "ev" ){\
		EigenVector eps(ngrids);\
		EigenMatrix eps1rho(Eps1Rho.rows(), ngrids);\
		EigenMatrix eps1sigma(Eps1Sigma.rows(), ngrids);\
		xc_gga_exc_vxc(&func, ngrids, rho.data(), sigma.data(), eps.data(), eps1rho.data(), eps1sigma.data());\
		Eps += eps;\
		Eps1Rho += eps1rho;\
		Eps1Sigma += eps1sigma;\
	}else if ( order == "f" ){\
		EigenMatrix eps2rho2(Eps2Rho2.rows(), ngrids);\
		EigenMatrix eps2rhosigma(Eps2RhoSigma.rows(), ngrids);\
		EigenMatrix eps2sigma2(Eps2Sigma2.rows(), ngrids);\
		xc_gga_fxc(&func, ngrids, rho.data(), sigma.data(), eps2rho2.data(), eps2rhosigma.data(), eps2sigma2.data());\
		Eps2Rho2 += eps2rho2;\
		Eps2RhoSigma += eps2rhosigma;\
		Eps2Sigma2 += eps2sigma2;\
	}

#define __XC_MGGA__\
	if ( order == "e" ){\
		EigenVector eps(ngrids);\
		xc_mgga_exc(&func, ngrids, rho.data(), sigma.data(), lapl.data(), tau.data(), eps.data());\
		Eps += eps;\
	}else if ( order == "v" ){\
		EigenMatrix eps1rho(Eps1Rho.rows(), ngrids);\
		EigenMatrix eps1sigma(Eps1Sigma.rows(), ngrids);\
		EigenMatrix eps1lapl(Eps1Lapl.rows(), ngrids);\
		EigenMatrix eps1tau(Eps1Tau.rows(), ngrids);\
		xc_mgga_vxc(&func, ngrids, rho.data(), sigma.data(), lapl.data(), tau.data(), eps1rho.data(), eps1sigma.data(), eps1lapl.data(), eps1tau.data());\
		Eps1Rho += eps1rho;\
		Eps1Sigma += eps1sigma;\
		Eps1Lapl += eps1lapl;\
		Eps1Tau += eps1tau;\
	}else if ( order == "ev" ){\
		EigenVector eps(ngrids);\
		EigenMatrix eps1rho(Eps1Rho.rows(), ngrids);\
		EigenMatrix eps1sigma(Eps1Sigma.rows(), ngrids);\
		EigenMatrix eps1lapl(Eps1Lapl.rows(), ngrids);\
		EigenMatrix eps1tau(Eps1Tau.rows(), ngrids);\
		xc_mgga_exc_vxc(&func, ngrids, rho.data(), sigma.data(), lapl.data(), tau.data(), eps.data(), eps1rho.data(), eps1sigma.data(), eps1lapl.data(), eps1tau.data());\
		Eps += eps;\
		Eps1Rho += eps1rho;\
		Eps1Sigma += eps1sigma;\
		Eps1Lapl += eps1lapl;\
		Eps1Tau += eps1tau;\
	}

void ExchangeCorrelation::Evaluate(std::string order, Grid& grid){
	#pragma omp parallel for schedule(static) num_threads(grid.getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : grid.SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){		
			const int ngrids = subgrid->NumGrids;
			const int nspins = subgrid->Spin;
			const EigenTensor<2> rho = subgrid->Rho.shuffle(Eigen::array<int, 2>{1, 0});
			const EigenTensor<2> sigma = subgrid->Sigma.shuffle(Eigen::array<int, 2>{1, 0});
			const EigenTensor<2> lapl = subgrid->Lapl.shuffle(Eigen::array<int, 2>{1, 0});
			const EigenTensor<2> tau = subgrid->Tau.shuffle(Eigen::array<int, 2>{1, 0});

			EigenVector Eps;
			if ( order == "e" || order == "ev" ){
				Eps = EigenZero(ngrids, 1);
				subgrid->Eps.resize(ngrids);
			}
			EigenMatrix Eps1Rho, Eps1Sigma, Eps1Lapl, Eps1Tau;
			if ( order == "v" || order == "ev" ){
				Eps1Rho = EigenZero(nspins, ngrids);
				Eps1Sigma = EigenZero(nspins == 1 ? 1 : 3, ngrids);
				Eps1Lapl = EigenZero(nspins, ngrids);
				Eps1Tau = EigenZero(nspins, ngrids);
				subgrid->Eps1Rho.resize(ngrids, nspins);
				subgrid->Eps1Sigma.resize(ngrids, nspins == 1 ? 1 : 3);
				subgrid->Eps1Lapl.resize(ngrids, nspins);
				subgrid->Eps1Tau.resize(ngrids, nspins);
			}
			EigenMatrix Eps2Rho2, Eps2RhoSigma, Eps2Sigma2;
			if ( order == "f" ){
				Eps2Rho2 = EigenZero(nspins == 1 ? 1 : 3, ngrids);
				Eps2RhoSigma = EigenZero(nspins == 1 ? 1 : 6, ngrids);
				Eps2Sigma2 = EigenZero(nspins == 1 ? 1 : 6, ngrids);
				subgrid->Eps2Rho2.resize(ngrids, nspins, nspins);
				subgrid->Eps2RhoSigma.resize(ngrids, nspins, nspins == 1 ? 1 : 3);
				subgrid->Eps2Sigma2.resize(ngrids, nspins == 1 ? 1 : 3, nspins == 1 ? 1 : 3);
			}

			for ( int code : this->Codes ){
				xc_func_type func;
				xc_func_init(&func, code, this->Spin);
				switch (func.info->family){
					case (XC_FAMILY_LDA):
					case (XC_FAMILY_HYB_LDA):
						__XC_LDA__ break;
					case (XC_FAMILY_GGA):
					case (XC_FAMILY_HYB_GGA):
						__XC_GGA__ break;
					case (XC_FAMILY_MGGA):
					case (XC_FAMILY_HYB_MGGA):
						__XC_MGGA__ break;
				}
				xc_func_end(&func);
			}

			if ( order == "e" || order == "ev" ){
				std::memcpy(subgrid->Eps.data(), Eps.data(), ngrids * 8);
			}
			#define CopyTensor(tensor){\
				tensor.transposeInPlace();\
				std::memcpy(subgrid->tensor.data(), tensor.data(), tensor.size() * 8);\
			}
			if ( order == "v" || order == "ev" ){
				CopyTensor(Eps1Rho);
				CopyTensor(Eps1Sigma);
				CopyTensor(Eps1Lapl);
				CopyTensor(Eps1Tau);
			}
			#define CopySymTensor3(tensor){\
				tensor.transposeInPlace();\
				for ( int i = 0, k = 0; i < subgrid->tensor.dimension(0); i++ ){\
					for ( int j = i; j < subgrid->tensor.dimension(1); j++, k++ ){\
						std::memcpy(&subgrid->tensor(0, i, j), &tensor(0, k), ngrids * 8);\
						std::memcpy(&subgrid->tensor(0, j, i), &tensor(0, k), ngrids * 8);\
					}\
				}\
			}
			if ( order == "f" ){
				CopySymTensor3(Eps2Rho2);
				CopyTensor(Eps2RhoSigma);
				CopySymTensor3(Eps2Sigma2);
			}
		}
	}
}
