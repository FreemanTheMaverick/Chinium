.PHONY: all

all : MAIN HARTREEFOCKKOHNSHAM
#all: MAIN GATEWAY DIIS_ GRID HARTREEFOCKKOHNSHAM MULTIWFN INTEGRAL LOCALIZATION EXCHANGECORRELATION

MAIN: main.cpp
	$(CXX) -o ../obj/main.o main.cpp -c $(GeneralFlags) $(EIGEN3Flags)

GATEWAY: Gateway.cpp
	$(CXX) -o ../obj/Gateway.o Gateway.cpp -c $(GeneralFlags) $(EIGEN3Flags)

DIIS_:
	$(MAKE) -C DIIS

EXCHANGECORRELATION:
	$(MAKE) -C ExchangeCorrelation

HARTREEFOCKKOHNSHAM:
	$(MAKE) -C HartreeFockKohnSham

MULTIWFN:
	$(MAKE) -C Multiwfn

INTEGRAL:
	$(MAKE) -C Integral

GRID:
	$(MAKE) -C Grid

LOCALIZATION:
	$(MAKE) -C Localization
