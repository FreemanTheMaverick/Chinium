main:main.cpp
	g++ main.cpp -c -Wall -O2

Gateway:Gateway.cpp
	g++ Gateway.cpp -c -Wall -O2

InitialGuess:InitialGuess.cpp
	g++ InitialGuess.cpp -c -lint2 -Wall

AtomicIntegrals:AtomicIntegrals.cpp
	g++ AtomicIntegrals.cpp -c -lint2 -fopenmp -Wall -O2

HartreeFock:HartreeFock.cpp
	g++ HartreeFock.cpp -c -fopenmp -Wall -O2

LD:main.o Gateway.o InitialGuess.o AtomicIntegrals.o HartreeFock.o
	g++ main.o Gateway.o InitialGuess.o AtomicIntegrals.o HartreeFock.o -lint2 -fopenmp




LinearAlgebra:LinearAlgebra.cpp
	g++ LinearAlgebra.cpp -c -Wall -O2

Symmetry/PointGroups:Symmetry/PointGroups.cpp
	icpc Symmetry/PointGroups.cpp -c -lint2 -Wall -O2

Symmetry/Identification:Symmetry/Identification.cpp
	icpc Symmetry/Identification.cpp -c -lint2 -Wall -O2

Symmetry/OrbitalTransformation:Symmetry/OrbitalTransformation.cpp
	icpc Symmetry/OrbitalTransformation.cpp -c -lint2 -Wall -O2

Parser:Parser.cpp
	icpc Parser.cpp -c -Wall -O2

