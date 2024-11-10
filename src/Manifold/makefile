Flags = $(GeneralFlags) $(EIGEN3Flags)

.PHONY: all

all: Simplex Orthogonal Grassmann GrassmannQLocal

Simplex: Simplex.cpp
	$(CXX) -o ../../obj/Simplex.o Simplex.cpp -c $(Flags)

Orthogonal: Orthogonal.cpp
	$(CXX) -o ../../obj/Orthogonal.o Orthogonal.cpp -c $(Flags)

Grassmann: Grassmann.cpp
	$(CXX) -o ../../obj/Grassmann.o Grassmann.cpp -c $(Flags)

GrassmannQLocal: GrassmannQLocal.cpp
	$(CXX) -o ../../obj/GrassmannQLocal.o GrassmannQLocal.cpp -c $(Flags)
