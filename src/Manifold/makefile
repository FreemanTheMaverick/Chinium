Flags = $(GeneralFlags) $(EIGEN3Flags)

.PHONY: all

all: Simplex Orthogonal

Simplex: Simplex.cpp
	$(CXX) -o ../../obj/Simplex.o Simplex.cpp -c $(Flags)

Orthogonal: Orthogonal.cpp
	$(CXX) -o ../../obj/Orthogonal.o Orthogonal.cpp -c $(Flags)
