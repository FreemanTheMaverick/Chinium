CXX=g++
CC=gcc
# The path where you can find "Eigen/", "signature_of_eigen3_matrix_library" and "unsupported/".
EIGEN3=/home/yzhangnn/eigen3/include/eigen3
# The path where you can find "include/", "lib/" and "share/".
LIBINT2=/home/yzhangnn/libint_2.7.1
# The path where you can find "include/" and "lib64/".
OSQP=/home/yzhangnn/osqp_0.6.3

GeneralFlags=-Wall -O2
EIGEN3Flags=-I$(EIGEN3) -mavx2
LIBINT2Flags=-I$(LIBINT2)/include -L$(LIBINT2)/lib -lint2
OSQPFlags=-I$(OSQP)/include/osqp -L$(OSQP)/lib64 -losqpstatic -lm

.PHONY: all

all: main Gateway InitialGuess AtomicIntegrals HartreeFock Optimization OSQP LinearAlgebra
	$(CXX) -o Chinium main.o Gateway.o InitialGuess.o AtomicIntegrals.o HartreeFock.o Optimization.o OSQP.o LinearAlgebra.o -fopenmp $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags) $(OSQPFlags)

main: main.cpp
	$(CXX) main.cpp -c $(GeneralFlags) $(EIGEN3Flags)

Gateway: Gateway.cpp
	$(CXX) Gateway.cpp -c $(GeneralFlags)

InitialGuess: InitialGuess.cpp
	$(CXX) InitialGuess.cpp -c $(GeneralFlags) $(EIGEN3Flags)

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

LD: main.o Gateway.o InitialGuess.o AtomicIntegrals.o HartreeFock.o Optimization.o OSQP.o LinearAlgebra.o
	$(CXX) -o Chinium main.o Gateway.o InitialGuess.o AtomicIntegrals.o HartreeFock.o Optimization.o OSQP.o LinearAlgebra.o -fopenmp $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags) $(OSQPFlags)



ld: OSQP.o Optimization.o
	$(CXX) Optimization.o OSQP.o $(GeneralFlags) $(EIGEN3Flags) $(OSQPFlags)




Symmetry/PointGroups:Symmetry/PointGroups.cpp
	icpc Symmetry/PointGroups.cpp -c -lint2 -Wall -O2

Symmetry/Identification:Symmetry/Identification.cpp
	icpc Symmetry/Identification.cpp -c -lint2 -Wall -O2

Symmetry/OrbitalTransformation:Symmetry/OrbitalTransformation.cpp
	icpc Symmetry/OrbitalTransformation.cpp -c -lint2 -Wall -O2

Parser:Parser.cpp
	icpc Parser.cpp -c -Wall -O2

