Flags = $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags)

.PHONY: all

all: Normalization TwoCenter FourCenter

Normalization: Normalization.cpp
	$(CXX) -o ../../obj/Normalization.o Normalization.cpp -c $(Flags)

TwoCenter: TwoCenter.cpp
	$(CXX) -o ../../obj/OneCenter.o TwoCenter.cpp -c $(Flags)

FourCenter: FourCenter.cpp
	$(CXX) -o ../../obj/FourCenter.o FourCenter.cpp -c $(Flags)
