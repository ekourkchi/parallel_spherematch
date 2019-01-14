from time import time
import sys
import os
import numpy as np
from math import *

# spherematch related libraries
from astrometry.util.starutil_numpy import * 
import spherematch as spherematch


import random 

import pyfits
from pyfits import *
from numpy import radians, degrees, sin, cos, arctan2, hypot
from astropy.io import fits
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
  


  
  prefix = 'DR'
  fits_data = 'SORTED.CORE_radeczVIs_pixmap_2.2000995_2.7999725_SCORE_PSDi_Ebv_pixcov_69626.fits'

  fits_random = 'SORTED.randoms_radeczVIs_pixmap_2.2000070_2.8000197_SCORE_PSDi_Ebv_pixcov_652012.fits'
  
  
  outdir = 'OUTPUTS.1/'+prefix+'_hist_OMP/'
  indir = 'INPUTS.1/'
  
  
###################################################
   # DATA

  filename = indir + fits_data
  
  a = pyfits.open(filename)
  hdr = getheader(filename, 1) # header
  d = a[1].data
  
  ra = d.field('ra')
  dec = d.field('dec')
  s = d.field('s')
###################################################
  # RANDOM
  
  filename = indir + fits_random
  
  a = pyfits.open(filename)
  hdr = getheader(filename, 1) # header
  d = a[1].data
  
  ra_rand = d.field('ra')
  dec_rand = d.field('dec')
  s_rand = d.field('s')
###################################################





  #indices = np.arange(len(ra))
  #indices = my_shuffle(indices)
  #ra = ra[indices]
  #dec = dec[indices]


   
  radius_in_deg = 75.  # degrees
  r = angleto3Dradius(radius_in_deg)
  
  size = len(ra)
  p = 100
  
  n =  int(sys.argv[1])


  if True:
    
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
	  
	  
	  xyz = radectoxyz(ra_rand, dec_rand)

	
	  print "No.: ", n, "Go ... "
	  print "size all: ", size
	  print "[low:up]: ", low, up
	  
	  powmin = 0.1   # from 10**0.1 Mpc/h
	  powmax = 3.25  # to 10**3.25 Mpc/h
	  n_bins = 38
	  

#######################################################################
	  
	  time0 = time()
	  #(inds,dists) = spherematch.match_esn_omp(xyz_small, xyz, r, 100)
	  (inds,dists) = spherematch.match(xyz_small, xyz, r)

	  m1, m2 = inds[:,0], inds[:,1]
	  dist_in_deg = dist2deg(dists)
	  
	  d = dist_in_deg[:,0]
  	  w=np.where(d > 0.)

	  m1 = m1[w]
	  m2 = m2[w]
	  d = d[w]
	  m1 += low
	  
	  s1 = s[m1]
	  s2 = s_rand[m2]
	  pp = abs(s1-s2)
	  rp = (s1+s2)*d*np.pi/180./2.
	  ss = sqrt(pp**2.+rp**2.)
	  
	  
	  ### This is an easier alternative
	  #edges = np.logspace(powmin, powmax, num=39)
	  
	  #delta = 0.082732
	  delta = (powmax-powmin)/n_bins
	  edges = 10.**(np.arange(powmin, powmax+delta, delta))
	  

	  #mid_point = 10.**(np.arange(powmin+0.5*delta, powmax, delta))
	  
	  hist_s = np.histogram(ss, bins = edges)
	  hist_s = hist_s[0]

	  print "run_time Original = ", time()-time0
	  
#######################################################################
	  ## Ehsan OMP
	  time0 = time()
	  (c_hists, c_edges) = spherematch.match_sdr_omp(xyz_small, xyz, r, s, s_rand, low, powmin, powmax, n_bins, 100)
	  c_hists = c_hists[:,0]
	  c_hists = c_hists.astype(np.int64)
	  c_edges = c_edges[:,0]
	  print "run_time Ehsan = ", time()-time0
#######################################################################
          
          test = True
          if test:
	      print  'Hist(checksum): ', sum(hist_s), sum(c_hists), " | ", len(hist_s), len(c_hists)
	      print  'Edge(checksum): ', sum(edges), sum(c_edges),  " | ", len(edges), len(c_edges)
              
              ok = 0
	      for i in range(len(hist_s)):
		ok =+ ((hist_s[i] - c_hists[i]) + (edges[i] - c_edges[i]))
		#print hist_s[i], c_hists[i],  " | ", edges[i], c_edges[i]
	      ok += (edges[len(edges)-1] - c_edges[len(edges)-1])
	      if ok < 1.E-5:
		print "\n OK ... :)\n"
	      else:
		print "\n Something is wrong ... :(\n"
	    