Flags = $(GeneralFlags) $(EIGEN3Flags)

.PHONY: all

all: GuessSCF FockFormation SelfConsistentField

GuessSCF: GuessSCF.cpp
	$(CXX) -o ../../obj/GuessSCF.o GuessSCF.cpp -c $(Flags)

FockFormation: FockFormation.cpp
	$(CXX) -o ../../obj/FockFormation.o FockFormation.cpp -c $(Flags)

SelfConsistentField: SelfConsistentField.cpp
	$(CXX) -o ../../obj/SelfConsistentField.o SelfConsistentField.cpp -c $(Flags)


