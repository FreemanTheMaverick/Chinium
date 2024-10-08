Flags = $(GeneralFlags) $(EIGEN3Flags)

.PHONY: all

all: Fock Localize

Fock: Fock.cpp
	$(CXX) -o ../../obj/Fock.o Fock.cpp -c $(Flags)

Localize: Localize.cpp
	$(CXX) -o ../../obj/Localize.o Localize.cpp -c $(Flags)
