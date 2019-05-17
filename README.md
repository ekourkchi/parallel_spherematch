## Parallel Spherematch

![screenshot from 2019-01-13 20-52-25](https://user-images.githubusercontent.com/13570487/51099472-fc6a9780-1774-11e9-90b4-68edbdac8938.png)

k-d trees are useful data structures to store multidimensional points in such a way to preserve the relative position of the data points. Using this structure, we find all possible pairs of data points, distributed on a sphere, that are within a particular angular separation. We use the Astrometry.net spherematch package which is originally developed in C with a Python interface for astronomers. We introduce new functions in the level of Python-C extension of the package that enables us to use the power of OpenMP. We find that however adding more threads increases the efficiency of spherematch, the total run-time is dominated by other operations that need to be done in Python. Therefore, we implement the other intensive operations in C where we can also use OpenMP. This together with breaking the entire job into several smaller sub-tasks help of to reduce the run-time of individual sub-tasks, and therefore the total queue waiting time, that can dominate the total run-time.

The full report of this project is available in [ehsan.ICS632.project.pdf](https://github.com/ekourkchi/parallel_spherematch/blob/master/ehsan.ICS632.project.pdf).

Downloading Astrometry.net package we modified the following files and recompile the entire package:

URL: [http://astrometry.net/downloads/](http://astrometry.net/downloads/)

This is the directory where all the following files exist: astrometry.net-0.64/libkd

1) spherematch.py --> The higher level Python wrapper. It needs to import "spherematch_c" shared library

2) pyspherematch.c --> The wrapper Python-C extension

3) setup.py --> modified to include OMP flag, when compiling "pyspherematch.c" and producing "spherematch_c.so"

****************************************************************

All other auxiliary Python codes to run tests and measure the performance:

1) ultra_test.py --> compares the original "spherematch.match" and Python implemented histogram code with "spherematch.match_sdr_omp" which performs all the necessary calculations in C.
2) pr.test1.py --> To measure the performance results for "spherematch.match_esn_omp"
3) pr.test2.py --> To measure the performance results for "spherematch.match_sss_omp"

Plotting the results: 
4) test1.plot.py  


- - - -
 * Copyright 2015
 * Author: Ehsan Kourkchi <ekourkchi@gmail.com>
5) test2.plot.desktop.py 

****************************************************************
