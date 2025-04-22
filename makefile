export MAKE = __MAKE__
export CXX = __CXX__

export EIGEN3 = __EIGEN3__
# The path where you can find "Eigen/", "signature_of_eigen3_matrix_library" and "unsupported/".
export LIBINT2 = __LIBINT2__
# The path where you can find "include/", "lib/" and "share/".
export LIBXC = __LIBXC__
# The path where you can find "bin/", "include/" and "lib/".
export MANIVERSE = __MANIVERSE__

export GeneralFlags = -Wall -Wextra -Wpedantic -fopenmp -O3 -std=c++2a
export EIGEN3Flags = -isystem $(EIGEN3) -march=native -DEIGEN_INITIALIZE_MATRICES_BY_ZERO
export LIBINT2Flags = -isystem $(LIBINT2)/include/ -L$(LIBINT2)/lib/ -lint2
export LIBXCFlags = -isystem $(LIBXC)/include/ -L$(LIBXC)/lib/ -lxc
export MANIVERSEFlags = -isystem $(MANIVERSE)/include/ -L$(MANIVERSE)/lib/ -l:libmaniverse.a

.PHONY: all

all:
	mkdir -p obj/
	$(MAKE) -C src/
	cd obj/ && $(CXX) -o ../Chinium *.o $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags) $(LIBXCFlags) $(MANIVERSEFlags)

ld:
	cd obj/ && $(CXX) -o ../Chinium *.o $(GeneralFlags) $(EIGEN3Flags) $(LIBINT2Flags) $(LIBXCFlags) $(MANIVERSEFlags)

clean:
	rm -rf obj/*.o
