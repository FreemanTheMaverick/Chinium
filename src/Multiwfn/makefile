Flags = $(GeneralFlags) $(EIGEN3Flags)

.PHONY: all

all: MwfnCenter MwfnShell MwfnIO

MwfnBasis: MwfnBasis.cpp
	$(CXX) -o ../../obj/MwfnBasis.o MwfnBasis.cpp -c $(Flags)

MwfnCenter: MwfnCenter.cpp
	$(CXX) -o ../../obj/MwfnCenter.o MwfnCenter.cpp -c $(Flags)

MwfnIO: MwfnIO.cpp
	$(CXX) -o ../../obj/MwfnIO.o MwfnIO.cpp -c $(Flags)

MwfnShell: MwfnShell.cpp
	$(CXX) -o ../../obj/MwfnShell.o MwfnShell.cpp -c $(Flags)

