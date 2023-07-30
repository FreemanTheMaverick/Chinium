CXX=g++
CC=gcc
# The path where you can find "Eigen/", "signature_of_eigen3_matrix_library" and "unsupported/".
EIGEN3=/usr/include/eigen3
# The path where you can find "include/", "lib/" and "share/".
LIBINT2=/home/freeman/libint_2.7.2
# The path where you can find "include/" and "lib64/".
OSQP=/home/freeman/osqp/osqp-1.0.0.beta1
# The path where you can fine "bin/", "include/" and "lib/".
LIBXC=/home/freeman/libxc_6.2.2

GeneralFlags=-Wall -O3
EIGEN3Flags=-I$(EIGEN3) -mavx2
LIBINT2Flags=-I$(LIBINT2)/include -L$(LIBINT2)/lib -lint2
OSQPFlags=-I$(OSQP)/include/public -I$(OSQP)/build/include/public -L$(OSQP)/build/out -losqpstatic -lm
LIBXCFlags=-I$(LIBXC)/include -L$(LIBXC)/lib -lxc

DFLib='-D__DF_library_path__="$(PWD)/DensityFunctionals/"'

.PHONY: all

all: main Gateway InitialGuess Libint2 AtomicIntegrals HartreeFock Optimization OSQP LinearAlgebra GridIntegrals
	$(CXX) -o Chinium main.o Gateway.o InitialGuess.o Libint2.o AtomicIntegrals.o HartreeFock.o Optimization.o OSQP.o LinearAlgebra.o GridIntegrals.o -fopenmp $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags) $(OSQPFlags)

main: main.cpp
	$(CXX) main.cpp -c $(GeneralFlags) $(EIGEN3Flags)

Gateway: Gateway.cpp
	$(CXX) Gateway.cpp -c $(GeneralFlags)

InitialGuess: InitialGuess.cpp
	$(CXX) InitialGuess.cpp -c $(GeneralFlags) $(EIGEN3Flags)

Libint2: Libint2.cpp
	$(CXX) Libint2.cpp -c $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags)

AtomicIntegrals: AtomicIntegrals.cpp
	$(CXX) AtomicIntegrals.cpp -c -fopenmp $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags)

HartreeFock: HartreeFock.cpp
	$(CXX) HartreeFock.cpp -c -fopenmp $(GeneralFlags) $(EIGEN3Flags)

Optimization: Optimization.cpp
	$(CXX) Optimization.cpp -c $(GeneralFlags) $(EIGEN3Flags)

OSQP: OSQP.c
	$(CC) OSQP.c -c $(GeneralFlags) $(OSQPFlags)

LinearAlgebra: LinearAlgebra.cpp
	$(CXX) LinearAlgebra.cpp -c $(GeneralFlags) $(EIGEN3Flags)

GridIntegrals: GridIntegrals.cpp
	$(CXX) GridIntegrals.cpp -c $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags)

DensityFunctional: DensityFunctional.cpp
	$(CXX) DensityFunctional.cpp -c $(GeneralFlags) $(LIBXCFlags) $(DFLib)

LD:
	$(CXX) -o Chinium main.o Gateway.o InitialGuess.o Libint2.o AtomicIntegrals.o HartreeFock.o Optimization.o OSQP.o LinearAlgebra.o GridIntegrals.o -fopenmp $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags) $(OSQPFlags)



ld: DensityFunctional.o
	$(CXX) DensityFunctional.o $(GeneralFlags) $(LIBXCFlags)




Symmetry/PointGroups:Symmetry/PointGroups.cpp
	icpc Symmetry/PointGroups.cpp -c -lint2 -Wall -O2

Symmetry/Identification:Symmetry/Identification.cpp
	icpc Symmetry/Identification.cpp -c -lint2 -Wall -O2

Symmetry/OrbitalTransformation:Symmetry/OrbitalTransformation.cpp
	icpc Symmetry/OrbitalTransformation.cpp -c -lint2 -Wall -O2

Parser:Parser.cpp
	icpc Parser.cpp -c -Wall -O2

