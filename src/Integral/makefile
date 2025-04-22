Flags = $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags)

.PHONY: all

all: Normalization Int2C1E Int4C2E Parallel

Normalization: Normalization.cpp
	$(CXX) -o ../../obj/Normalization.o Normalization.cpp -c $(Flags)

Int2C1E: Int2C1E.cpp
	$(CXX) -o ../../obj/Int2C1E.o Int2C1E.cpp -c $(Flags)

Int4C2E: Int4C2E.cpp
	$(CXX) -o ../../obj/Int4C2E.o Int4C2E.cpp -c $(Flags)

Parallel: Parallel.cpp
	$(CXX) -o ../../obj/Parallel.o Parallel.cpp -c $(GeneralFlags) $(EIGEN3Flags)
