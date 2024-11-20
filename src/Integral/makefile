Flags = $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags)

.PHONY: all

all: NuclearRepulsion Normalization TwoCenter FourCenter

NuclearRepulsion: NuclearRepulsion.cpp
	$(CXX) -o ../../obj/NuclearRepulsion.o NuclearRepulsion.cpp -c $(GeneralFlags) $(EIGEN3Flags)

Normalization: Normalization.cpp
	$(CXX) -o ../../obj/Normalization.o Normalization.cpp -c $(Flags)

TwoCenter: TwoCenter.cpp
	$(CXX) -o ../../obj/TwoCenter.o TwoCenter.cpp -c $(Flags)

FourCenter: FourCenter.cpp
	$(CXX) -o ../../obj/FourCenter.o FourCenter.cpp -c $(Flags)
