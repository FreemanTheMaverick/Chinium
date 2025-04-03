export MAKE = make
export CXX = g++

export EIGEN3 = /home/yzhangnn/eigen3/include/eigen3/
# The path where you can find "Eigen/", "signature_of_eigen3_matrix_library" and "unsupported/".
export LIBINT2 = /home/yzhangnn/scratch/libint_2.8.0/
# The path where you can find "include/", "lib/" and "share/".
export LIBXC = /home/yzhangnn/libxc_6.2.2/
# The path where you can find "bin/", "include/" and "lib/".
export MANIVERSE = /home/yzhangnn/Maniverse/

export GeneralFlags = -Wall -Wextra -Wpedantic -fopenmp -O3 -std=c++20
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
