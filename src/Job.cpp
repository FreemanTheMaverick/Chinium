#include <string>
#include <typeinfo>
#include <stdexcept>
#include <cassert>

#include "Job.h"

Job::Job(std::string inp){
	size_t suffix_pos = inp.find_last_of('.');
	if ( suffix_pos != std::string::npos ) basename = inp.substr(0, suffix_pos);
	else throw std::runtime_error("The input file must have a suffix!");
}

void Job::Calculate0(){
	assert(0 && "Job not implemented!");
}

void Job::Calculate1(){
	assert(0 && "Job not implemented!");
}

void Job::Calculate2(){
	assert(0 && "Job not implemented!");
}

void Job::PostProcess0(){
	assert(0 && "Job not implemented!");
}

void Job::PostProcess1(){
	assert(0 && "Job not implemented!");
}

void Job::PostProcess2(){
	assert(0 && "Job not implemented!");
}
