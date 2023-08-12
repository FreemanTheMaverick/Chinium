CXX=g++
# The path where you can find "Eigen/", "signature_of_eigen3_matrix_library" and "unsupported/".
EIGEN3=/home/yzhangnn/eigen3/include/eigen3
# The path where you can find "include/", "lib/" and "share/".
LIBINT2=/home/yzhangnn/libint_2.7.1
# The path where you can find "include/" and "lib64/".
OSQP=/home/yzhangnn/osqp_0.6.3
# The path where you can fine "bin/", "include/" and "lib/".
LIBXC=/home/yzhangnn/libxc_6.2.2

GeneralFlags=-Wall -O3
EIGEN3Flags=-I$(EIGEN3) -mavx2
LIBINT2Flags=-I$(LIBINT2)/include -L$(LIBINT2)/lib -lint2
OSQPFlags=-I$(OSQP)/include/osqp -L$(OSQP)/lib64 -losqpstatic
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

OSQP: OSQP.cpp
	$(CXX) OSQP.cpp -c $(GeneralFlags) $(OSQPFlags)

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
