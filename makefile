CXX=g++
CC=gcc
EIGEN3=/home/yzhangnn/eigen3/include/eigen3 # The path where you can find "Eigen/", "signature_of_eigen3_matrix_library" and "unsupported/".
LIBINT2=/home/yzhangnn/libint_2.7.1/ # The path where you can find "include", "lib" and "share".
OSQP=/home/yzhangnn/osqp_0.6.3/


.PHONY: all

all: main Gateway InitialGuess AtomicIntegrals HartreeFock Optimization
	$(CXX) main.o Gateway.o InitialGuess.o AtomicIntegrals.o HartreeFock.o Optimization.o -lint2 -fopenmp -I$(EIGEN3) -I$(LIBINT2)/include -L$(LIBINT2)/lib -o Chinium

main: main.cpp
	$(CXX) main.cpp -c -Wall -O2 -I$(EIGEN3)

Gateway: Gateway.cpp
	$(CXX) Gateway.cpp -c -Wall -O2

InitialGuess: InitialGuess.cpp
	$(CXX) InitialGuess.cpp -c -Wall -I$(EIGEN3)

AtomicIntegrals: AtomicIntegrals.cpp
	$(CXX) AtomicIntegrals.cpp -c -lint2 -fopenmp -Wall -O2 -I$(EIGEN3) -I$(LIBINT2)/include

HartreeFock: HartreeFock.cpp
	$(CXX) HartreeFock.cpp -c -fopenmp -Wall -O2 -I$(EIGEN3)

Optimization: Optimization.cpp
	$(CXX) Optimization.cpp -c -Wall -O2  -I$(EIGEN3)

OSQP: OSQP.c
	$(CC) OSQP.c -c -Wall -O2 -I/home/yzhangnn/osqp_0.6.3/include/osqp -L/home/yzhangnn/osqp_0.6.3/lib64 -losqp

LD: main.o Gateway.o InitialGuess.o AtomicIntegrals.o HartreeFock.o Optimization.o
	$(CXX) main.o Gateway.o InitialGuess.o AtomicIntegrals.o HartreeFock.o Optimization.o -lint2 -fopenmp -I$(EIGEN3) -I$(LIBINT2)/include -L$(LIBINT2)/lib -o Chinium

ld: OSQP.o Optimization.o
	$(CXX) Optimization.o OSQP.o -I$(EIGEN3) -I$(OSQP)/include/osqp -L$(OSQP)/lib64 -losqp



LinearAlgebra:LinearAlgebra.cpp
	$(CXX) LinearAlgebra.cpp -c -Wall -O2

Symmetry/PointGroups:Symmetry/PointGroups.cpp
	icpc Symmetry/PointGroups.cpp -c -lint2 -Wall -O2

Symmetry/Identification:Symmetry/Identification.cpp
	icpc Symmetry/Identification.cpp -c -lint2 -Wall -O2

Symmetry/OrbitalTransformation:Symmetry/OrbitalTransformation.cpp
	icpc Symmetry/OrbitalTransformation.cpp -c -lint2 -Wall -O2

Parser:Parser.cpp
	icpc Parser.cpp -c -Wall -O2

