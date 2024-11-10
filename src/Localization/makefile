Flags = $(GeneralFlags) $(EIGEN3Flags)

.PHONY: all

all: Fock FosterBoys PipekMezey Localize

Fock: Fock.cpp
	$(CXX) -o ../../obj/Fock.o Fock.cpp -c $(Flags)

FosterBoys: FosterBoys.cpp
	$(CXX) -o ../../obj/FosterBoys.o FosterBoys.cpp -c $(Flags)

PipekMezey: PipekMezey.cpp
	$(CXX) -o ../../obj/PipekMezey.o PipekMezey.cpp -c $(Flags)

Localize: Localize.cpp
	$(CXX) -o ../../obj/Localize.o Localize.cpp -c $(Flags)
