Flags = $(GeneralFlags) $(EIGEN3Flags)

.PHONY: all

all: DIIS CDIIS ADIIS

DIIS: DIIS.cpp
	$(CXX) -o ../../obj/DIIS.o DIIS.cpp -c $(Flags)

CDIIS: CDIIS.cpp
	$(CXX) -o ../../obj/CDIIS.o CDIIS.cpp -c $(Flags)

ADIIS: ADIIS.cpp
	$(CXX) -o ../../obj/ADIIS.o ADIIS.cpp -c $(Flags) $(MANIVERSEFlags)
