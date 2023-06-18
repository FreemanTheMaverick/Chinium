CXX=g++
EIGEN3=
LIBINT2=

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

LD: main.o Gateway.o InitialGuess.o AtomicIntegrals.o HartreeFock.o Optimization.o
	$(CXX) main.o Gateway.o InitialGuess.o AtomicIntegrals.o HartreeFock.o Optimization.o -lint2 -fopenmp -I$(EIGEN3) -I$(LIBINT2)/include -L$(LIBINT2)/lib -o Chinium





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

