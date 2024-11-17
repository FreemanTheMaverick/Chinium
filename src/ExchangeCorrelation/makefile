.PHONY: all

all: ExchangeCorrelation MwfnXC

ExchangeCorrelation: ExchangeCorrelation.cpp
	$(CXX) -o ../../obj/ExchangeCorrelation.o ExchangeCorrelation.cpp -c $(GeneralFlags) $(LIBXCFlags)

MwfnXC: MwfnXC.cpp
	$(CXX) -o ../../obj/MwfnXC.o MwfnXC.cpp -c $(GeneralFlags) $(EIGEN3Flags)
