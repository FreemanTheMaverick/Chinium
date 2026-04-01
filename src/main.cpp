#include <string>
#include <memory>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "Gateway.h"
#include "HartreeFockKohnSham.h"
//#include "Localization/Localize.h"

#define Round(x) (int)( isInt(x) ? std::lround(x) : std::floor(x) )
#define __Bad_Input__ throw std::runtime_error("Something wrong in the input file!")

int main(int /*argc*/, char* argv[]){
	std::printf("*** Chinium started ***\n");

	// File names
	std::string inp = argv[1];
	std::string job_name = inp;
	size_t suffix_pos = inp.find_last_of('.');
	if ( suffix_pos != std::string::npos ) job_name = job_name.substr(0, suffix_pos);

	// Determinining the job type
	const std::string jobtype = ReadJobType(inp);
	const int wfntype = ReadWfnType(inp);
	const double temperature = ReadTemperature(inp);
	std::unique_ptr<Job> job;
	if ( jobtype == "SCF" ){
		if ( wfntype == 0 || wfntype == 2 ){
			if ( temperature == 0 ) job = std::make_unique<R_SCF>(inp);
			else if ( temperature > 0 ) job = std::make_unique<RGC_SCF>(inp);
			else __Bad_Input__;
		}else if ( wfntype == 1 ){
			if ( temperature == 0 ) job = std::make_unique<U_SCF>(inp);
			else if ( temperature > 0 ) job = std::make_unique<UGC_SCF>(inp);
			else __Bad_Input__;
		}else __Bad_Input__;
	}else if ( jobtype == "TWODET" && wfntype == 2 ){
		job = std::make_unique<TwoDet>(inp);
	//}else if ( jobtype == "LOCALIZATION" ){
	//	if ( wfntype == 0 ) job = std::make_unique<R_Localization>(inp);
	//	else if ( wfntype == 1 ) job = std::make_unique<U_Localization>(inp);
	//	else __Bad_Input__;
	}else __Bad_Input__;

	// Running the job
	const int derivative = ReadDerivative(inp);
	if ( derivative >= 0 ){
		job->Calculate0();
		job->PostProcess0();
	}
	if ( derivative >= 1 ){
		job->Calculate1();
		job->PostProcess1();
	}
	if ( derivative >= 2 ){
		job->Calculate2();
		job->PostProcess2();
	}

	std::printf("*** Chinium terminated normally ***\n");
	return 0;
}
