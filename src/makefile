.PHONY: all

all: MAIN GRID OPTIMIZATION HARTREEFOCKKOHNSHAM MULTIWFN INTEGRAL LOCALIZATION
#all: HARTREEFOCKKOHNSHAM

MAIN: main.cpp
	$(CXX) -o ../obj/main.o main.cpp -c $(GeneralFlags) $(EIGEN3Flags)
	
#EXCHANGECORRELATION:
#	$(MAKE) -C ExchangeCorrelation

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
