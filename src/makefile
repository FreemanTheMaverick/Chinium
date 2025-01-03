.PHONY: all

#all: MAIN GATEWAY GRID OPTIMIZATION HARTREEFOCKKOHNSHAM MULTIWFN INTEGRAL LOCALIZATION MANIFOLD EXCHANGECORRELATION
all: HARTREEFOCKKOHNSHAM GRID

MAIN: main.cpp
	$(CXX) -o ../obj/main.o main.cpp -c $(GeneralFlags) $(EIGEN3Flags)

GATEWAY: Gateway.cpp
	$(CXX) -o ../obj/Gateway.o Gateway.cpp -c $(GeneralFlags) $(EIGEN3Flags)
	
EXCHANGECORRELATION:
	$(MAKE) -C ExchangeCorrelation

OPTIMIZATION:
	$(MAKE) -C Optimization

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

MANIFOLD:
	$(MAKE) -C Manifold
