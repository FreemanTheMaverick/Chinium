CXX=g++
# The path where you can find "Eigen/", "signature_of_eigen3_matrix_library" and "unsupported/".
EIGEN3=/home/yzhangnn/eigen3/include/eigen3
# The path where you can find "include/", "lib/" and "share/".
LIBINT2=/home/yzhangnn/scratch/libint_2.7.2
# The path where you can find "include/" and "lib64/".
OSQP=/home/yzhangnn/osqp_0.6.3
# The path where you can find "bin/", "include/" and "lib/".
LIBXC=/home/yzhangnn/libxc_6.2.2

GeneralFlags=-O3 -Wall -Wextra -Wpedantic
EIGEN3Flags=-isystem $(EIGEN3) -march=native -fopenmp -DEIGEN_INITIALIZE_MATRICES_BY_ZERO
# -DEIGEN_NO_DEBUG
LIBINT2Flags=-isystem $(LIBINT2)/include -L$(LIBINT2)/lib -lint2
OSQPFlags=-isystem $(OSQP)/include/osqp -L$(OSQP)/lib64 -losqpstatic
LIBXCFlags=-isystem $(LIBXC)/include -L$(LIBXC)/lib -lxc

DFLib='-D__DF_library_path__="$(PWD)/DensityFunctionals/"'
GridLib='-D__Grid_library_path__="$(PWD)/Grids/"'
SAPLib='-D__SAP_library_path__="$(PWD)/SAP/"'

.PHONY: all

all: main Gateway NuclearRepulsion InitialGuess Libint2 AtomicIntegrals AtoIntGradients HartreeFock Optimization OSQP LinearAlgebra GridGeneration GridIntegrals DensityFunctional Lebedev HFGradient CoupledPerturbed OccupationGradient HFHessian
	$(CXX) -o Chinium main.o Gateway.o NuclearRepulsion.o InitialGuess.o Libint2.o AtomicIntegrals.o AtoIntGradients.o HartreeFock.o Optimization.o OSQP.o LinearAlgebra.o GridGeneration.o GridIntegrals.o DensityFunctional.o sphere_lebedev_rule.o HFGradient.o CoupledPerturbed.o OccupationGradient.o HFHessian.o $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags) $(OSQPFlags) $(LIBXCFlags)

main: main.cpp
	$(CXX) main.cpp -c $(GeneralFlags) $(EIGEN3Flags)

Gateway: Gateway.cpp
	$(CXX) Gateway.cpp -c $(GeneralFlags) $(DFLib) $(GridLib)

NuclearRepulsion: NuclearRepulsion.cpp
	$(CXX) NuclearRepulsion.cpp -c $(GeneralFlags) $(EIGEN3Flags)

InitialGuess: InitialGuess.cpp
	$(CXX) InitialGuess.cpp -c $(GeneralFlags) $(EIGEN3Flags) $(SAPLib)

Libint2: Libint2.cpp
	$(CXX) Libint2.cpp -c $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags)

AtomicIntegrals: AtomicIntegrals.cpp
	$(CXX) AtomicIntegrals.cpp -c $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags)

AtoIntGradients: AtoIntGradients.cpp
	$(CXX) AtoIntGradients.cpp -c $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags)

HartreeFock: HartreeFock.cpp
	$(CXX) HartreeFock.cpp -c $(GeneralFlags) $(EIGEN3Flags)

Optimization: Optimization.cpp
	$(CXX) Optimization.cpp -c $(GeneralFlags) $(EIGEN3Flags)

OSQP: OSQP.cpp
	$(CXX) OSQP.cpp -c $(GeneralFlags) $(OSQPFlags)

LinearAlgebra: LinearAlgebra.cpp
	$(CXX) LinearAlgebra.cpp -c $(GeneralFlags) $(EIGEN3Flags)

GridGeneration: GridGeneration.cpp
	$(CXX) GridGeneration.cpp -c $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags) $(GridLib)

GridIntegrals: GridIntegrals.cpp
	$(CXX) GridIntegrals.cpp -c $(GeneralFlags) $(EIGEN3Flags)

DensityFunctional: DensityFunctional.cpp
	$(CXX) DensityFunctional.cpp -c $(GeneralFlags) $(LIBXCFlags) $(DFLib)

Lebedev: sphere_lebedev_rule.cpp
	$(CXX) sphere_lebedev_rule.cpp -c $(GeneralFlags)

HFGradient: HFGradient.cpp
	$(CXX) HFGradient.cpp -c $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags)

CoupledPerturbed: CoupledPerturbed.cpp
	$(CXX) CoupledPerturbed.cpp -c $(GeneralFlags) $(EIGEN3Flags)

OccupationGradient: OccupationGradient.cpp
	$(CXX) OccupationGradient.cpp -c $(GeneralFlags) $(EIGEN3Flags)

HFHessian: HFHessian.cpp
	$(CXX) HFHessian.cpp -c $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags)

LD:
	$(CXX) -o Chinium main.o Gateway.o NuclearRepulsion.o InitialGuess.o Libint2.o AtomicIntegrals.o AtoIntGradients.o HartreeFock.o Optimization.o OSQP.o LinearAlgebra.o GridGeneration.o GridIntegrals.o DensityFunctional.o sphere_lebedev_rule.o HFGradient.o CoupledPerturbed.o OccupationGradient.o HFHessian.o $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags) $(OSQPFlags) $(LIBXCFlags)


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
