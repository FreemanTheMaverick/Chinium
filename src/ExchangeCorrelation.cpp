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
#include <optional>
#include <libmwfn.h>

#include "Macro.h"
#include "Grid/Grid.h"
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
		EigenTensor<1> eps(Eps.value().dimensions()); eps.setZero();\
		xc_lda_exc(&func, ngrids, rho.value().data(), eps.data());\
		Eps.value() += eps;\
	}else if ( order == "v" ){\
		EigenTensor<2> eps1rho(Eps1Rho.value().dimensions()); eps1rho.setZero();\
		xc_lda_vxc(&func, ngrids, rho.value().data(), eps1rho.data());\
		Eps1Rho.value() += eps1rho;\
	}else if ( order == "ev" ){\
		EigenTensor<1> eps(Eps.value().dimensions()); eps.setZero();\
		EigenTensor<2> eps1rho(Eps1Rho.value().dimensions()); eps1rho.setZero();\
		xc_lda_exc_vxc(&func, ngrids, rho.value().data(), eps.data(), eps1rho.data());\
		Eps.value() += eps;\
		Eps1Rho.value() += eps1rho;\
	}else if ( order == "f" ){\
		EigenTensor<2> eps2rho2(Eps2Rho2.value().dimensions()); eps2rho2.setZero();\
		xc_lda_fxc(&func, ngrids, rho.value().data(), eps2rho2.data());\
		Eps2Rho2.value() += eps2rho2;\
	}

#define __XC_GGA__\
	if ( order == "e" ){\
		EigenTensor<1> eps(Eps.value().dimensions()); eps.setZero();\
		xc_gga_exc(&func, ngrids, rho.value().data(), sigma.value().data(), eps.data());\
		Eps.value() += eps;\
	}else if ( order == "v" ){\
		EigenTensor<2> eps1rho(Eps1Rho.value().dimensions()); eps1rho.setZero();\
		EigenTensor<2> eps1sigma(Eps1Sigma.value().dimensions()); eps1sigma.setZero();\
		xc_gga_vxc(&func, ngrids, rho.value().data(), sigma.value().data(), eps1rho.data(), eps1sigma.data());\
		Eps1Rho.value() += eps1rho;\
		Eps1Sigma.value() += eps1sigma;\
	}else if ( order == "ev" ){\
		EigenTensor<1> eps(Eps.value().dimensions()); eps.setZero();\
		EigenTensor<2> eps1rho(Eps1Rho.value().dimensions()); eps1rho.setZero();\
		EigenTensor<2> eps1sigma(Eps1Sigma.value().dimensions()); eps1sigma.setZero();\
		xc_gga_exc_vxc(&func, ngrids, rho.value().data(), sigma.value().data(), eps.data(), eps1rho.data(), eps1sigma.data());\
		Eps.value() += eps;\
		Eps1Rho.value() += eps1rho;\
		Eps1Sigma.value() += eps1sigma;\
	}else if ( order == "f" ){\
		EigenTensor<2> eps2rho2(Eps2Rho2.value().dimensions()); eps2rho2.setZero();\
		EigenTensor<2> eps2rhosigma(Eps2RhoSigma.value().dimensions()); eps2rhosigma.setZero();\
		EigenTensor<2> eps2sigma2(Eps2Sigma2.value().dimensions()); eps2sigma2.setZero();\
		xc_gga_fxc(&func, ngrids, rho.value().data(), sigma.value().data(), eps2rho2.data(), eps2rhosigma.data(), eps2sigma2.data());\
		Eps2Rho2.value() += eps2rho2;\
		Eps2RhoSigma.value() += eps2rhosigma;\
		Eps2Sigma2.value() += eps2sigma2;\
	}

#define __XC_MGGA__\
	if ( order == "e" ){\
		EigenTensor<1> eps(Eps.value().dimensions()); eps.setZero();\
		xc_mgga_exc(&func, ngrids, rho.value().data(), sigma.value().data(), lapl.value().data(), tau.value().data(), eps.data());\
		Eps.value() += eps;\
	}else if ( order == "v" ){\
		EigenTensor<2> eps1rho(Eps1Rho.value().dimensions()); eps1rho.setZero();\
		EigenTensor<2> eps1sigma(Eps1Sigma.value().dimensions()); eps1sigma.setZero();\
		EigenTensor<2> eps1lapl(Eps1Lapl.value().dimensions()); eps1lapl.setZero();\
		EigenTensor<2> eps1tau(Eps1Tau.value().dimensions()); eps1tau.setZero();\
		xc_mgga_vxc(&func, ngrids, rho.value().data(), sigma.value().data(), lapl.value().data(), tau.value().data(), eps1rho.data(), eps1sigma.data(), eps1lapl.data(), eps1tau.data());\
		Eps1Rho.value() += eps1rho;\
		Eps1Sigma.value() += eps1sigma;\
		Eps1Lapl.value() += eps1lapl;\
		Eps1Tau.value() += eps1tau;\
	}else if ( order == "ev" ){\
		EigenTensor<1> eps(Eps.value().dimensions()); eps.setZero();\
		EigenTensor<2> eps1rho(Eps1Rho.value().dimensions()); eps1rho.setZero();\
		EigenTensor<2> eps1sigma(Eps1Sigma.value().dimensions()); eps1sigma.setZero();\
		EigenTensor<2> eps1lapl(Eps1Lapl.value().dimensions()); eps1lapl.setZero();\
		EigenTensor<2> eps1tau(Eps1Tau.value().dimensions()); eps1tau.setZero();\
		xc_mgga_exc_vxc(&func, ngrids, rho.value().data(), sigma.value().data(), lapl.value().data(), tau.value().data(), eps.data(), eps1rho.data(), eps1sigma.data(), eps1lapl.data(), eps1tau.data());\
		Eps.value() += eps;\
		Eps1Rho.value() += eps1rho;\
		Eps1Sigma.value() += eps1sigma;\
		Eps1Lapl.value() += eps1lapl;\
		Eps1Tau.value() += eps1tau;\
	}

void ExchangeCorrelation::Evaluate(std::string order, Grid& grid){
	#pragma omp parallel for schedule(static) num_threads(grid.getNumThreads())
	for ( std::vector<std::unique_ptr<SubGrid>>& subgrids : grid.SubGridBatches ){
		for ( std::unique_ptr<SubGrid>& subgrid : subgrids ){		
			const int ngrids = subgrid->NumGrids;
			int spin_type = -1;
			switch (subgrid->Spin){
				case 1: spin_type = 0; break;
				case 2: spin_type = subgrid->MWFN->Wfntype == 1 ? 1 : 2; break;
				case 3: spin_type = 3; break;
			}
			EigenTensor<2> d_out(1, 1);
			if ( spin_type == 0 ){
				d_out.resize(ngrids, 4); d_out.setZero();
				if (subgrid->Rho.data()) std::memcpy(&d_out(0, 0), subgrid->Rho.data(), ngrids * 8);
				if (subgrid->Sigma.data()) std::memcpy(&d_out(0, 1), subgrid->Sigma.data(), ngrids * 8);
				if (subgrid->Lapl.data()) std::memcpy(&d_out(0, 2), subgrid->Lapl.data(), ngrids * 8);
				if (subgrid->Tau.data()) std::memcpy(&d_out(0, 3), subgrid->Tau.data(), ngrids * 8);
			}else if ( spin_type == 1 || spin_type == 2 ){
				d_out.resize(ngrids, 9); d_out.setZero();
				if (subgrid->Rho.data()) std::memcpy(&d_out(0, 0), subgrid->Rho.data(), ngrids * 2 * 8);
				if (subgrid->Sigma.data()) std::memcpy(&d_out(0, 2), subgrid->Sigma.data(), ngrids * 3 * 8);
				if (subgrid->Lapl.data()) std::memcpy(&d_out(0, 5), subgrid->Lapl.data(), ngrids * 2 * 8);
				if (subgrid->Tau.data()) std::memcpy(&d_out(0, 7), subgrid->Tau.data(), ngrids * 2 * 8);
			}else if ( spin_type == 3 ){
				d_out.resize(ngrids, 15); d_out.setZero();
				if (subgrid->Rho.data()) std::memcpy(&d_out(0, 0), subgrid->Rho.data(), ngrids * 3 * 8);
				if (subgrid->Sigma.data()) std::memcpy(&d_out(0, 3), subgrid->Sigma.data(), ngrids * 6 * 8);
				if (subgrid->Lapl.data()) std::memcpy(&d_out(0, 9), subgrid->Lapl.data(), ngrids * 3 * 8);
				if (subgrid->Tau.data()) std::memcpy(&d_out(0, 12), subgrid->Tau.data(), ngrids * 3 * 8);
			}
			EigenMatrix trans_mat;
			if ( spin_type == 0 ){
				trans_mat = EigenOne(4, 4);
			}else if ( spin_type == 1 ){
				trans_mat = EigenOne(9, 9);
			}else if ( spin_type == 2 ){
				trans_mat.resize(9, 9); trans_mat <<
					0.5 , 0.5 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
					1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
					//0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
					0.  , 0.  , 0.25, 0.25, 0.25, 0.  , 0.  , 0.  , 0.  ,
					0.  , 0.  , 1.  , 0.5 , 0.  , 0.  , 0.  , 0.  , 0.  ,
					0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
					//0.  , 0.  , 0.  , 0.5 , 1.  , 0.  , 0.  , 0.  , 0.  ,
					//0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
					//0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  ,
					0.  , 0.  , 0.  , 0.  , 0.  , 0.5 , 0.5 , 0.  , 0.  ,
					0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  ,
					//0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  ,
					0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.5 , 0.5 ,
					0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  ;
					//0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  ;
			}else if ( spin_type == 3 ){
				trans_mat.resize(15, 9); trans_mat <<
					0.5 , 0.5 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
					1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
					0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
					0.  , 0.  , 0.25, 0.25, 0.25, 0.  , 0.  , 0.  , 0.  ,
					0.  , 0.  , 1.  , 0.5 , 0.  , 0.  , 0.  , 0.  , 0.  ,
					0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
					0.  , 0.  , 0.  , 0.5 , 1.  , 0.  , 0.  , 0.  , 0.  ,
					0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
					0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  ,
					0.  , 0.  , 0.  , 0.  , 0.  , 0.5 , 0.5 , 0.  , 0.  ,
					0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  ,
					0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  ,
					0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.5 , 0.5 ,
					0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  ,
					0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  ;
			}
			const Eigen::TensorMap<EigenTensor<2>> trans(trans_mat.data(), trans_mat.rows(), trans_mat.cols());
            const EigenTensor<2> d_in = d_out.contract(trans, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)}); // d_in = d_out * trans
			std::optional<const Eigen::Map<const EigenVector>> rho, sigma, lapl, tau;
			if ( spin_type == 0 ){
				rho.emplace(&d_in(0, 0), ngrids);
				sigma.emplace(&d_in(0, 1), ngrids);
				lapl.emplace(&d_in(0, 2), ngrids);
				tau.emplace(&d_in(0, 3), ngrids);
			}else{
				rho.emplace(&d_in(0, 0), ngrids * 2);
				sigma.emplace(&d_in(0, 2), ngrids * 3);
				lapl.emplace(&d_in(0, 5), ngrids * 2);
				tau.emplace(&d_in(0, 7), ngrids * 2);
			}

			EigenTensor<1> v0;
			EigenTensor<2> v1_in;
			EigenTensor<2> v2_in;
			std::optional<Eigen::TensorMap<EigenTensor<1>>> Eps;
			std::optional<Eigen::TensorMap<EigenTensor<2>>> Eps1Rho, Eps1Sigma, Eps1Lapl, Eps1Tau;
			std::optional<Eigen::TensorMap<EigenTensor<2>>> Eps2Rho2, Eps2RhoSigma, Eps2Sigma2;
			if ( order == "e" || order == "ev" ){
				v0.resize(ngrids); v0.setZero();
				Eps.emplace(v0.data(), ngrids);
			}
			if ( order == "v" || order == "ev" ){
				if ( spin_type == 0 ){
					v1_in.resize(ngrids, 4);
					Eps1Rho.emplace(&v1_in(0, 0), ngrids, 1);
					Eps1Sigma.emplace(&v1_in(0, 1), ngrids, 1);
					Eps1Lapl.emplace(&v1_in(0, 2), ngrids, 1);
					Eps1Tau.emplace(&v1_in(0, 3), ngrids, 1);
				}else{
					v1_in.resize(ngrids, 9);
					Eps1Rho.emplace(&v1_in(0, 0), ngrids, 2);
					Eps1Sigma.emplace(&v1_in(0, 2), ngrids, 3);
					Eps1Lapl.emplace(&v1_in(0, 5), ngrids, 2);
					Eps1Tau.emplace(&v1_in(0, 7), ngrids, 2);
				}
				v1_in.setZero();
			}
			if ( order == "f" ){
				if ( spin_type == 0 ){
					v2_in.resize(ngrids, 3);
					Eps2Rho2.emplace(&v2_in(0, 0), ngrids, 1);
					Eps2RhoSigma.emplace(&v2_in(0, 1), ngrids, 1);
					Eps2Sigma2.emplace(&v2_in(0, 2), ngrids, 1);
				}else{
					v2_in.resize(ngrids, 15);
					Eps2Rho2.emplace(&v2_in(0, 0), ngrids, 3);
					Eps2RhoSigma.emplace(&v2_in(0, 3), ngrids, 6);
					Eps2Sigma2.emplace(&v2_in(0, 9), ngrids, 6);
				}
				v2_in.setZero();
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
				subgrid->Eps = v0;
			}
			if ( order == "v" || order == "ev" ){
            	const EigenTensor<2> v1_out = v1_in.contract(trans, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 1)}); // v2_out = v2_in * trans.t
				if ( spin_type == 0 ){
					subgrid->Eps1Rho = SliceTensor(v1_out, {0, 0}, {ngrids, 1});
					subgrid->Eps1Sigma = SliceTensor(v1_out, {0, 1}, {ngrids, 1});
					subgrid->Eps1Lapl = SliceTensor(v1_out, {0, 2}, {ngrids, 1});
					subgrid->Eps1Tau = SliceTensor(v1_out, {0, 3}, {ngrids, 1});
				}else if ( spin_type == 1 || spin_type == 2 ){
					subgrid->Eps1Rho = SliceTensor(v1_out, {0, 0}, {ngrids, 2});
					subgrid->Eps1Sigma = SliceTensor(v1_out, {0, 2}, {ngrids, 3});
					subgrid->Eps1Lapl = SliceTensor(v1_out, {0, 5}, {ngrids, 2});
					subgrid->Eps1Tau = SliceTensor(v1_out, {0, 7}, {ngrids, 2});
				}else if ( spin_type == 3 ){
					subgrid->Eps1Rho = SliceTensor(v1_out, {0, 0}, {ngrids, 3});
					subgrid->Eps1Sigma = SliceTensor(v1_out, {0, 3}, {ngrids, 6});
					subgrid->Eps1Lapl = SliceTensor(v1_out, {0, 9}, {ngrids, 3});
					subgrid->Eps1Tau = SliceTensor(v1_out, {0, 12}, {ngrids, 3});
				}
			}
			if ( order == "f" ){
				EigenTensor<3> v2_mid;
				if ( spin_type == 0 ){
					v2_mid.resize(ngrids, 2, 2);
					std::memcpy(&v2_mid(0, 0, 0), &v2_in(0, 0), ngrids * 8); // rho rho
					std::memcpy(&v2_mid(0, 0, 1), &v2_in(0, 1), ngrids * 8); // rho sigma
					std::memcpy(&v2_mid(0, 1, 0), &v2_in(0, 1), ngrids * 8); // sigma rho
					std::memcpy(&v2_mid(0, 1, 1), &v2_in(0, 2), ngrids * 8); // sigma sigma
				}else{
					v2_mid.resize(ngrids, 5, 5);
					// rho rho
					std::memcpy(&v2_mid(0, 0, 0), &v2_in(0, 0), ngrids * 8); // rho_alpha rho_alpha
					std::memcpy(&v2_mid(0, 0, 1), &v2_in(0, 1), ngrids * 8); // rho_alpha rho_beta
					std::memcpy(&v2_mid(0, 1, 0), &v2_in(0, 1), ngrids * 8); // rho_beta rho_alpha
					std::memcpy(&v2_mid(0, 1, 1), &v2_in(0, 2), ngrids * 8); // rho_beta rho_beta
					// rho sigma
					std::memcpy(&v2_mid(0, 0, 2), &v2_in(0, 3), ngrids * 8); // rho_alpha sigma_alpha
					std::memcpy(&v2_mid(0, 2, 0), &v2_in(0, 3), ngrids * 8); // sigma_alpha rho_alpha
					std::memcpy(&v2_mid(0, 0, 3), &v2_in(0, 4), ngrids * 8); // rho_alpha sigma_mix
					std::memcpy(&v2_mid(0, 3, 0), &v2_in(0, 4), ngrids * 8); // sigma_mix rho_alpha
					std::memcpy(&v2_mid(0, 0, 4), &v2_in(0, 5), ngrids * 8); // rho_alpha sigma_beta
					std::memcpy(&v2_mid(0, 4, 0), &v2_in(0, 5), ngrids * 8); // sigma_beta rho_alpha
					std::memcpy(&v2_mid(0, 1, 2), &v2_in(0, 6), ngrids * 8); // rho_beta sigma_alpha
					std::memcpy(&v2_mid(0, 2, 1), &v2_in(0, 6), ngrids * 8); // sigma_alpha rho_beta
					std::memcpy(&v2_mid(0, 1, 3), &v2_in(0, 7), ngrids * 8); // rho_beta sigma_mix
					std::memcpy(&v2_mid(0, 3, 1), &v2_in(0, 7), ngrids * 8); // sigma_mix rho_beta
					std::memcpy(&v2_mid(0, 1, 4), &v2_in(0, 8), ngrids * 8); // rho_beta sigma_beta
					std::memcpy(&v2_mid(0, 4, 1), &v2_in(0, 8), ngrids * 8); // sigma_beta rho_beta
					// sigma sigma
					std::memcpy(&v2_mid(0, 2, 2), &v2_in(0, 9), ngrids * 8); // sigma_alpha sigma_alpha
					std::memcpy(&v2_mid(0, 2, 3), &v2_in(0, 10), ngrids * 8); // sigma_alpha sigma_mix
					std::memcpy(&v2_mid(0, 3, 2), &v2_in(0, 10), ngrids * 8); // sigma_mix sigma_alpha
					std::memcpy(&v2_mid(0, 2, 4), &v2_in(0, 11), ngrids * 8); // sigma_alpha sigma_beta
					std::memcpy(&v2_mid(0, 4, 2), &v2_in(0, 11), ngrids * 8); // sigma_beta sigma_alpha
					std::memcpy(&v2_mid(0, 3, 3), &v2_in(0, 12), ngrids * 8); // sigma_mix sigma_mix
					std::memcpy(&v2_mid(0, 3, 4), &v2_in(0, 13), ngrids * 8); // sigma_mix sigma_beta
					std::memcpy(&v2_mid(0, 4, 3), &v2_in(0, 13), ngrids * 8); // sigma_beta sigma_mix
					std::memcpy(&v2_mid(0, 4, 4), &v2_in(0, 14), ngrids * 8); // sigma_beta sigma_beta
				}

				const EigenTensor<2> trans2 = SliceTensor(trans, {0, 0}, {v2_mid.dimension(1), v2_mid.dimension(1)});
				const EigenTensor<3> v2_out = v2_mid.contract(trans2, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 1)}).contract(trans2, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(2, 1)}); // v3_out = trans * v3_in * trans.t

				if ( spin_type == 0 ){
					subgrid->Eps2Rho2 = SliceTensor(v2_out, {0, 0, 0}, {ngrids, 1, 1});
					subgrid->Eps2RhoSigma = SliceTensor(v2_out, {0, 0, 1}, {ngrids, 1, 1});
					subgrid->Eps2Sigma2 = SliceTensor(v2_out, {0, 1, 1}, {ngrids, 1, 1});
				}else if ( spin_type == 1 || spin_type == 2 ){
					subgrid->Eps2Rho2 = SliceTensor(v2_out, {0, 0, 0}, {ngrids, 2, 2});
					subgrid->Eps2RhoSigma = SliceTensor(v2_out, {0, 0, 2}, {ngrids, 2, 3});
					subgrid->Eps2Sigma2 = SliceTensor(v2_out, {0, 2, 2}, {ngrids, 3, 3});
				}else if ( spin_type == 3 ){
					subgrid->Eps2Rho2 = SliceTensor(v2_out, {0, 0, 0}, {ngrids, 3, 3});
					subgrid->Eps2RhoSigma = SliceTensor(v2_out, {0, 0, 3}, {ngrids, 3, 6});
					subgrid->Eps2Sigma2 = SliceTensor(v2_out, {0, 3, 3}, {ngrids, 6, 6});
				}
			}
		}
	}
}
