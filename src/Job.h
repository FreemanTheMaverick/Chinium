#pragma once

#include <string>

class Job{ public:
	std::string basename;
	Job(std::string inp);
	virtual void Calculate0();
	virtual void PostProcess0();
	virtual void Calculate1();
	virtual void PostProcess1();
	virtual void Calculate2();
	virtual void PostProcess2();
};
