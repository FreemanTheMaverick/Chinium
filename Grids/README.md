Taking `sg-2/h.grid` as an example:
```
5 75 # There are 5 groups of consecutive shells with the same numbers of spherical points and 75 shells in total.
de2 2.6 -2.312968699567745 1.1619004544598117 # Radial-formula-dependent line.
6 35 # Each of the innermost 35 shells has 6 spherical points.
110 12 # ...
302 16 # ...
86 7 # ...
26 5 # Each of the outmost 5 shells have 26 spherical points.
```
The second line is radial-formula-dependent. For the second version of double exponential (DE2), this line has four tokens: "de2", alpha, minimal xi and maximal xi, as described in [*Theor. Chem. Acc.* (2011) 130:645-669](https://link.springer.com/article/10.1007/s00214-011-0985-x).
