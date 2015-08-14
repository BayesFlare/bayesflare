#!/usr/bin/env python

import bayesflare as bf
import os
import sys
import random
import numpy as np

"""
A script to find "quiet" stars in Kepler data. For all stars will first remove the best fit
quadratic from the data, so as not to worry about slow changes.

We will define "quiet" by the following criteria:
 - The standard deviation of the data before and after detrending it (using a 4th order polynomial
and a running median filter) are within 25% of each other.
 - The maximum and minimum values in the data are within 7.5 sigma of each other.
 - There are no "spectral lines" in the data with *amplitude* more than 7.5 times the median
amplitude.
"""
# number of quiet stars to find
Nquiet = 1

# directory of Kepler Quarter 1 data
keplerdir = '/home/holly/data/001429589'

# list directory contents and randomise
kfs = os.listdir(keplerdir)
random.shuffle(kfs)

# go through the list and make sure only long cadence data is included
kfs = [os.path.join(keplerdir, x) for x in kfs if 'llc' in x]

nq = 0 # number of quiet stars found
quietstars = []

print len(kfs)

outfile = 'quietstarlist.txt'

# loop through files
for i, kf in enumerate(kfs):
  print "Loop %d" % (i+1)
  
  # read in lightcurve
  lc = bf.Lightcurve(curve=kf)
  
  # remove a quadratic from the light curve to get any very slow variations that are still "quiet"
  # enough
  ts = lc.cts - lc.cts[0]
  z = np.polyfit(ts, lc.clc, 2)
  f = np.poly1d(z)
  lc.clc = lc.clc - f(ts)
  
  # get the "normal" standard deviation of the data
  stdpre = np.std(lc.clc)
  lc.detrend(method='runningmedian', nbins=55)
  # get the standard deviation of the detrended the lightcurve
  stdpost = np.std(lc.clc)

  # check detrended standard deviation is within 25% of pre-detrended
  if stdpost < 0.75*stdpre or stdpost > 1.25*stdpre:
    print "%s fails standard deviation test (%f vs %f)" % (kf, stdpre, stdpost)
    del lc
    continue
  
  # check maximum and minimum lightcurve value are within 7.5 sigma of each other
  if np.amax(lc.clc)-np.amin(lc.clc) > 7.5*stdpre :
    print "%s fails min/max test (%f sigma difference)" % (kf,
(np.amax(lc.clc)-np.amin(lc.clc)/stdpre))
    del lc
    continue
  
  # check for spectral lines
  sk, f = lc.psd()
  sk = np.sqrt(sk) # convert power spectrum into amplitude spectrum
  msk = np.median(sk)
  foundline = False
  for s in sk:
    if s > 7.5*msk:
      sline = s/msk
      foundline = True
      break
      
  if foundline:
    print "%s fails to the line test (line height %f)" % (kf, sline)
    del lc
    continue
  
  # We've found a quiet star!
  print "%s contains a quiet star" % kf
  nq = nq+1
  quietstars.append(kf)
  
  # output to file
  f = open(outfile, 'a')
  f.write(kf+'\n')
  f.close()
  
  del lc
  
  # stopping criterion
  if nq > Nquiet-1:
    break

  print "We found %d quiet stars" % nq

