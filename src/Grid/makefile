.PHONY: all

all: Lebedev GridGeneration GridAO GridDensity GridPotential

Lebedev: sphere_lebedev_rule.cpp
	$(CXX) -o ../../obj/sphere_lebedev_rule.o sphere_lebedev_rule.cpp -c $(GeneralFlags)

GridGeneration: GridGeneration.cpp
	$(CXX) -o ../../obj/GridGeneration.o GridGeneration.cpp -c $(GeneralFlags) $(EIGEN3Flags)

GridAO: GridAO.cpp
	$(CXX) -o ../../obj/GridAO.o GridAO.cpp -c $(GeneralFlags) $(EIGEN3Flags)

GridDensity: GridDensity.cpp
	$(CXX) -o ../../obj/GridDensity.o GridDensity.cpp -c $(GeneralFlags) $(EIGEN3Flags)

GridPotential: GridPotential.cpp
	$(CXX) -o ../../obj/GridPotential.o GridPotential.cpp -c $(GeneralFlags) $(EIGEN3Flags)

