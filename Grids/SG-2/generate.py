from scipy.optimize import fsolve
import sys

e=2.7182818284
da2=2.6
da3=2.7
rmin=1e-7

# Expectation value of orbital radius of the outermost orbital of atom (ATOMIC DATA AND NUCLEAR DATA TABLES 53, 113-162 (1993)) and alphas (Theor. Chem. Acc. (2011) 130:645-669)
ras=   [[  1, "H",1.500000,2.6,2.7],\
        [  2,"He",0.927272,da2,da3],\
        [  3,"Li",3.873661,3.2,3.0],\
        [  4,"Be",2.649396,2.4,2.4],\
        [  5, "B",2.204757,2.4,2.4],\
        [  6, "C",1.714495,2.2,2.4],\
        [  7, "N",1.409631,2.2,2.4],\
        [  8, "O",1.232198,2.2,2.6],\
        [  9, "F",1.084786,2.2,2.1],\
        [ 10,"Ne",0.965273,da2,da3],\
        [ 11,"Na",4.208762,3.2,3.2],\
        [ 12,"Mg",3.252938,2.4,2.6],\
        [ 13,"Al",3.433889,2.5,2.6],\
        [ 14,"Si",2.752216,2.3,2.8],\
        [ 15, "P",2.322712,2.5,2.4],\
        [ 16, "S",2.060717,2.5,2.4],\
        [ 17,"Cl",1.842024,2.5,2.6],\
        [ 18,"Ar",1.662954,da2,da3],\
        [ 19, "K",5.243652,da2,da3],\
        [ 20,"Ca",4.218469,da2,da3],\
        [ 21,"Sc",3.959716,da2,da3],\
	[ 22,"Ti",3.778855,da2,da3],\
	[ 23, "V",3.626388,da2,da3],\
	[ 24,"Cr",3.675012,da2,da3],\
	[ 25,"Mn",3.381917,da2,da3],\
	[ 26,"Fe",3.258487,da2,da3],\
	[ 27,"Co",3.153572,da2,da3],\
	[ 28,"Ni",3.059109,da2,da3],\
	[ 29,"Cu",3.330979,da2,da3],\
	[ 30,"Zn",2.897648,da2,da3]\
        ]

def SolvePrint(index,test):
    for ra in ras:
        z=ra[0]
        n=ra[1]
        r=ra[2]
        a=ra[index+1]
        funcmin=lambda x:e**(a*x-e**(-x))-rmin
        xmin=fsolve(funcmin,-2)
        funcmax=lambda x:e**(a*x-e**(-x))-r*10
        xmax=fsolve(funcmax,1)
        if test:
            print(z,n,a,xmin[0],xmax[0],funcmin(xmin)[0],funcmax(xmax)[0])
        else:
            print(z,n,a,xmin[0],xmax[0])

if __name__=='__main__':
    index=eval(sys.argv[1])
    test=eval(sys.argv[2])
    SolvePrint(index,test)
