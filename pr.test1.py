from time import time
import sys
import os
import numpy as np
from math import *

# spherematch related libraries
from astrometry.util.starutil_numpy import * 
import spherematch as spherematch

from astropy.table import Table, Column 
from astropy.io import ascii, fits

import random 

import pyfits
from numpy import radians, degrees, sin, cos, arctan2, hypot
#################################################################


#################################################################
def my_shuffle(array):
        random.seed(0)
        random.shuffle(array)
        return array
#################################################################

def angleto3Dradius(angle, isDegree=True):
  
  if isDegree:
    angle = angle*pi/180.
  
  
  return sqrt((sin(angle))**2 + (1-cos(angle))**2)

#################################################################

### export OMP_NUM_THREADS=2

if __name__ == '__main__':
  


  
  fits_random = 'SORTED.randoms_radeczVIs_pixmap_2.2000070_2.8000197_SCORE_PSDi_Ebv_pixcov_652012.fits'
  
  
  indir = 'INPUTS.1/'
  
  
###################################################
  time0 = time()
###################################################
  # RANDOM
  
  filename = indir + fits_random
  
  a = pyfits.open(filename)
  hdr = pyfits.getheader(filename, 1) # header
  d = a[1].data
  
  ra = d.field('ra')
  dec = d.field('dec')
  s = d.field('s')
###################################################
  print "I/O time = ", time()-time0




  #indices = np.arange(len(ra))
  #indices = my_shuffle(indices)
  #ra = ra[indices]
  #dec = dec[indices]


   
  radius_in_deg = 75.  # degrees
  r = angleto3Dradius(radius_in_deg)
  
  size = len(ra)
  p = 100
  
  #n =  int(sys.argv[1])
  
  n_array = []
  T_orig = []
  T_omp = []
  
  iter = 1
  for n in range(0,7000,500):
      print "N:", n
      n_array.append(n)
    
      doJob = True
      low = p*n
      up = p*(n+1)
      if low <= size and up > size:
	  up = size
      if low >= size:
	  doJob = False
      if up<low or up<0 or low<0:
	  doJob = False
      
      
      if doJob:

	  ra_small = ra[low:up]
	  dec_small = dec[low:up]
	  xyz_small = radectoxyz(ra_small, dec_small)
	  
	  
	  xyz = radectoxyz(ra, dec)

	
	  #print "No.: ", n, "Go ... "
	  #print "# of data points: ", size
	  #print "[low:up]: ", low, up
	  
	  powmin = 0.1   # from 10**0.1 Mpc/h
	  powmax = 3.25  # to 10**3.25 Mpc/h
	  n_bins = 38
	  

#######################################################################
	  #sum = 0
	  #for i in range(iter):
	    #time0 = time()
	    #(inds,dists) = spherematch.match(xyz_small, xyz, r)
	    #sum += time()-time0
	  #T_orig.append(sum/iter)
#######################################################################
	  ## Ehsan OMP
	  sum = 0
	  for i in range(iter):
	    time0 = time()
	    (inds_omp,dists_omp) = spherematch.match_esn_omp(xyz_small, xyz, r, 100)
	    sum += time()-time0
	    print "# of match: ", len(inds_omp[:,0])
	  T_omp.append(sum/iter)
	  
#######################################################################


  #myTable = Table()
  #myTable.add_column(Column(data=n_array, name='N'))
  #myTable.add_column(Column(data=T_orig, name='T_orig'))
  #myTable.add_column(Column(data=T_omp, name='T_omp'))
  #myTable.write('out.pr.test1.txt', format='ascii.fixed_width',delimiter=',', bookend=False)
          
          #test = True
          #if test:
	      #print  'checksum (0/OK): ', sum(inds) - sum(inds_omp), " | ", len(dists) - len(dists_omp)
 
 
  #I/O time =  0.00744104385376
  #N: 0
  ## of match:  57062498
  #N: 500
  ## of match:  57356457
  #N: 1000
  ## of match:  57881907
  #N: 1500
  ## of match:  56802316
  #N: 2000
  ## of match:  55562024
  #N: 2500
  ## of match:  56216600
  #N: 3000
  ## of match:  55856268
  #N: 3500
  ## of match:  57112424
  #N: 4000
  ## of match:  56762397
  #N: 4500
  ## of match:  55985962
  #N: 5000
  ## of match:  56946478
  #N: 5500
  ## of match:  57359282
  #N: 6000
  ## of match:  56518878
  #N: 6500
  ## of match:  57329420
