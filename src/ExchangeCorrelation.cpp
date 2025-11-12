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
			if ( spin_type == 0 ) trans_mat = EigenOne(4, 4);
			else if ( spin_type == 1 ) trans_mat = EigenOne(9, 9);
			else if ( spin_type == 2 ){
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
			EigenMatrix rho, sigma, lapl, tau;
			if ( spin_type == 0 ){
				rho = Eigen::Map<const EigenMatrix>(&d_in(0, 0), 1, ngrids);
				sigma = Eigen::Map<const EigenMatrix>(&d_in(0, 1), 1, ngrids);
				lapl = Eigen::Map<const EigenMatrix>(&d_in(0, 2), 1, ngrids);
				tau = Eigen::Map<const EigenMatrix>(&d_in(0, 3), 1, ngrids);
			}else{
				rho = Eigen::Map<const EigenMatrix>(&d_in(0, 0), ngrids, 2).transpose();
				sigma = Eigen::Map<const EigenMatrix>(&d_in(0, 2), ngrids, 3).transpose();
				lapl = Eigen::Map<const EigenMatrix>(&d_in(0, 5), ngrids, 2).transpose();
				tau = Eigen::Map<const EigenMatrix>(&d_in(0, 7), ngrids, 2).transpose();
			}

			EigenVector Eps;
			if ( order == "e" || order == "ev" ){
				Eps.resize(ngrids); Eps.setZero();
			}
			EigenMatrix Eps1Rho, Eps1Sigma, Eps1Lapl, Eps1Tau;
			if ( order == "v" || order == "ev" ){
				if ( spin_type == 0 ){
					Eps1Rho = EigenZero(1, ngrids);
					Eps1Sigma = EigenZero(1, ngrids);
					Eps1Lapl = EigenZero(1, ngrids);
					Eps1Tau = EigenZero(1, ngrids);
				}else{
					Eps1Rho = EigenZero(2, ngrids);
					Eps1Sigma = EigenZero(3, ngrids);
					Eps1Lapl = EigenZero(2, ngrids);
					Eps1Tau = EigenZero(2, ngrids);
				}
			}
			EigenMatrix Eps2Rho2, Eps2RhoSigma, Eps2Sigma2;
			if ( order == "f" ){
				if ( spin_type == 0 ){
					Eps2Rho2 = EigenZero(1, ngrids);
					Eps2RhoSigma = EigenZero(1, ngrids);
					Eps2Sigma2 = EigenZero(1, ngrids);
				}else{
					Eps2Rho2 = EigenZero(3, ngrids);
					Eps2RhoSigma = EigenZero(6, ngrids);
					Eps2Sigma2 = EigenZero(6, ngrids);
				}
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

			EigenTensor<2> v1_in;
			if ( order == "v" || order == "ev" ){
				Eps1Rho = Eps1Rho.transpose().eval();
				Eps1Sigma = Eps1Sigma.transpose().eval();
				Eps1Lapl = Eps1Lapl.transpose().eval();
				Eps1Tau = Eps1Tau.transpose().eval();
				if ( spin_type == 0 ){
					v1_in.resize(ngrids, 4);
					std::memcpy(&v1_in(0, 0), Eps1Rho.data(), ngrids * 8);
					std::memcpy(&v1_in(0, 1), Eps1Sigma.data(), ngrids * 8);
					std::memcpy(&v1_in(0, 2), Eps1Lapl.data(), ngrids * 8);
					std::memcpy(&v1_in(0, 3), Eps1Tau.data(), ngrids * 8);
				}else{
					v1_in.resize(ngrids, 9);
					std::memcpy(&v1_in(0, 0), Eps1Rho.data(), 2 * ngrids * 8);
					std::memcpy(&v1_in(0, 2), Eps1Sigma.data(), 3 * ngrids * 8);
					std::memcpy(&v1_in(0, 5), Eps1Lapl.data(), 2 * ngrids * 8);
					std::memcpy(&v1_in(0, 7), Eps1Tau.data(), 2 * ngrids * 8);
				}
			}
			EigenTensor<2> v2_in;
			if ( order == "f" ){
				Eps2Rho2 = Eps2Rho2.transpose().eval();
				Eps2RhoSigma = Eps2RhoSigma.transpose().eval();
				Eps2Sigma2 = Eps2Sigma2.transpose().eval();
				if ( spin_type == 0 ){
					v2_in.resize(ngrids, 3);
					std::memcpy(&v2_in(0, 0), Eps2Rho2.data(), ngrids * 8);
					std::memcpy(&v2_in(0, 1), Eps2RhoSigma.data(), ngrids * 8);
					std::memcpy(&v2_in(0, 2), Eps2Sigma2.data(), ngrids * 8);
				}else{
					v2_in.resize(ngrids, 15);
					std::memcpy(&v2_in(0, 0), Eps2Rho2.data(), 3 * ngrids * 8);
					std::memcpy(&v2_in(0, 3), Eps2RhoSigma.data(), 6 * ngrids * 8);
					std::memcpy(&v2_in(0, 9), Eps2Sigma2.data(), 6 * ngrids * 8);
				}
			}

			if ( order == "e" || order == "ev" ){
				subgrid->Eps = Eigen::TensorMap<EigenTensor<1>>(Eps.data(), ngrids);
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
