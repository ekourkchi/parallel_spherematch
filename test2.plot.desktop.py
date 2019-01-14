#!/usr/bin/python
from time import time
import sys
import os
import numpy as np
from math import *
import glob
from astropy.table import Table, Column 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pylab as py


###################################################################################
## Figure Confuguration
###################################################################################

fig = py.figure(figsize=(6, 5), dpi=100)
fig.subplots_adjust(hspace=0.15, top=0.95, bottom=0.15, left=0.15, right=0.95)

ax = fig.add_subplot(111)
plt.minorticks_on()
plt.tick_params(which='major', length=7, width=1.5)
plt.tick_params(which='minor', length=4, color='#000033', width=1.0) 
###################################################################################





processes = [1,2,3,4,5,6,8]
original_time = []
omp_time = []

for process in processes:
  
  filename = 'out.pr.test2.p'+str(int(process))+'.txt'
  
  table = np.genfromtxt(filename , delimiter=',', filling_values="-100000", names=True, dtype=None )
  T_orig =  table['T_orig']
  T_omp =  table['T_omp']
  
  original_time.append(np.median(T_orig))
  omp_time.append(np.median(T_omp))
  


  

original_time = np.median(original_time)

ax.plot([0, 10], [original_time, original_time], '--', markersize = 4, color='black', label="Original Time",  lw=2) 
ax.plot(processes, omp_time, 'o', markersize = 5, color='blue', label="OMP",  lw=2)

##plt.xscale('log')
###plt.yscale('log')
plt.xlim(0,10)
plt.ylim(0,50)

ax.annotate('Desktop PC',(6,45), fontsize=14, color='green')
ax.set_xlabel('# of threads', fontsize=14)
ax.set_ylabel('Run Time (s)', fontsize=14)
ax.legend( loc=2 )
 
plt.show()



