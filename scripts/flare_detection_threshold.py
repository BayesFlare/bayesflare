#!/usr/bin/env python

"""
Script to calculate the threshold of the flare detection based on simulated Gaussian
noise.

The script can be run with purely white noise, or noise including a random
low-frequency sinusoidal variation, or on a set of real Kepler light curves.
"""

import numpy as np
import matplotlib

#matplotlib.use('Agg')

import matplotlib.pyplot as pl
import bayesflare as bf
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import sys
import os
import gzip

from optparse import OptionParser

__version__= "1.0"

# main function
if __name__=='__main__':

  usage = "Usage: %prog [options]"

  description = \
  """
     This script will get the threshold for detecting a flare (and a polynomial background) versus
     Gaussian noise (and a polynomial background) (or other noise sources) for a given false alarm
     rate.
  """

  parser = OptionParser( usage = usage, description = description, version = __version__ )

  parser.add_option("-N", "--Nsims", dest="Nsims",
                    help="The number of simulated noise realisations to use [default: %default].",
                    type="int", default=1000)

  parser.add_option("-L", "--file-list", dest="filelist",
                    help="If a file list is given it is assumed to be an ascii file containing \
one Kepler light curve file per line. These are then used instead of simulated data to calculate \
a threshold.")

  parser.add_option("-y", "--far", dest="far",
                    help="The required false alarm rate per time series (percent) [default: %default].",
                    type="float", default=0.1)

  parser.add_option("-s", "--std", dest="nstd",
                    help="The Gaussian noise standard deviation [default: %default].",
                    type="float", default=1.0)

  parser.add_option("-a", "--add-sinusoid", dest="dosinusoids",
                    help="Add a sinusoid to the Gaussian noise [default: %default].",
                    action="store_true", default=False)

  parser.add_option("-m", "--sinusoid-amp-min", dest="ampmin",
                    help="Minimum amplitude (in units of the noise standard deviation) range from \
which the sinusoid amplitudes will be drawn [default: %default].",
                    type="float", default=0.)

  parser.add_option("-M", "--sinusoid-amp-max", dest="ampmax",
                    help="Maximum amplitude (in units of the noise standard deviation) range from \
which the sinusoid amplitudes will be drawn [default: %default].", type="float", default=100.)

  parser.add_option("-f", "--sinusoid-freq-min", dest="freqmin",
                    help="Minimum frequency (Hz) range from which the sinusoid frequencies will \
be drawn [default: %default (=1/month)].", type="float", default=(1./(31.*86400.)))

  parser.add_option("-F", "--sinusoid-freq-max", dest="freqmax",
                    help="Maximum frequency (Hz) range from which the sinusoid frequencies will \
be drawn [default: %default (=0.5/day)].", type="float", default=(1./(2.*86400.)))

  parser.add_option("-l", "--length", dest="tlength",
                    help="The length (in seconds) for each simulated time series [default: %default \
(the length of Kepler Q1 data)].", type="float", default=2893536.)

  parser.add_option("-p", "--time-step", dest="tstep",
                    help="The time interval between points (in seconds) for the time series \
[default: %default (the interval for Kepler Q1 data)].", type="float", default=1765.55929)

  parser.add_option("-b", "--background-length", dest="bglen",
                    help="The length (in number of data) of the running background window used \
in the Bayes factor calculation (must be an odd number) [default: %default (about 27 hours \
with the default time step)].", type="int", default=55)

  parser.add_option("-o", "--background-order", dest="bgorder",
                    help="The polynomial order of the fitted background variability [default: %default].",
                    type="int", default=4)

  parser.add_option("-x", "--output-data", dest="outdata",
                    help="Set an output file name for the odds ratio data. This file name \
will be stripped of it's extension and also used to output a .png image of the cumulative \
distribution. If the file name ends in \".gz\" then the output will be gzipped.", default=None)

  parser.add_option("-z", "--log-far-plot", dest="logplot",
                    help="If outputting the data set this to plot the FAR axis on a log-scale [default: %default].",
                    action="store_true", default=False)

  # read in arguments
  (opts, args) = parser.parse_args()

  dosinusoids = opts.dosinusoids # say whether to add sinusoids

  # check if there's a list of Kepler light curve files
  if not opts.__dict__['filelist']: # no list so simulate the data
    kl = False # no Kepler file list

    # number of simulations
    Nsims = opts.Nsims

    nstd = opts.nstd # the noise standard deviation

    if dosinusoids:
      ampranges = [opts.ampmin*nstd, opts.ampmax**nstd] # range of amplitudes of sinusoid (as a factor of the noise standard
      freqranges = [opts.freqmin, opts.freqmax] # frequencies ranging from 1/day to 1/month

      amps = ampranges[0] + np.random.rand(Nsims)*np.diff(ampranges) # uniform between ranges
      freqs = freqranges[0] + np.random.rand(Nsims)*np.diff(freqranges)

      # random initial phase between 0 and 2pi
      phase = 2.*np.pi*np.random.rand(Nsims)

    # create times stamps for light curves (the same length as Kepler Q1 data)
    ts = np.arange(0., opts.tlength, opts.tstep, dtype='float32')
  else: # there is a file list
    kl = True

    try:
      f = open(opts.filelist, "r")
    except:
      print >> sys.stderr, "Error... cannot open Kepler file list %s." % opts.filelist

    kfiles = f.readlines()
    f.close()

    Nsims = len(kfiles) # number of files

  # false alarm rate (%)
  far = opts.far

  bglen=opts.bglen
  bgorder=opts.bgorder

  if opts.outdata is not None:
    base = os.path.splitext(opts.outdata)
    gzipped = False

    if '.gz' in opts.outdata: # check for gzip extension
      gzipped = True
      basename = os.path.splitext(base[0])[0] # get base extension before the .gz

      try:
        f = gzip.open(opts.outdata, "w")
      except:
        print >> sys.stderr, "Error... could not open output file %s." % opts.outdata
        sys.exit(-1)
    else:
      basename = base[0]

      try:
        f = open(opts.outdata, "w")
      except:
        print >> sys.stderr, "Error... could not open output file %s." % opts.outdata
        sys.exit(-1)

    f.close()

  if bglen % 2 == 0:
    print >> sys.stderr, "Error... background length (bglen) must be an odd number"
    sys.exit(-1)

  Bfs = [] # list to hold Bayes factors

  # perform loop
  for i in range(Nsims):
    # create data containing a flare and white noise
    if kl:
      if os.path.isfile(kfiles[i].strip()):
        flarelc = bf.Lightcurve(curve=kfiles[i].strip())
      else:
        print >> sys.stderr, "Error... file in list (%s) does not exist." % kfiles[i].strip()
        sys.exit(-1)
    else:
      flarelc = bf.Lightcurve()

      flarelc.clc = nstd*np.random.randn(len(ts))
      flarelc.cts = np.copy(ts)
      flarelc.cle = np.zeros(len(ts))
      flarelc.cadence = 'long'

      if dosinusoids: # add sinusoid
        flarelc.clc = flarelc.clc + amps[i]*np.sin(2.*np.pi*freqs[i]*ts + phase[i])

    #pl.plot(flarelc.cts, flarelc.clc)
    #pl.show()

    Or = bf.OddsRatioDetector( flarelc,
                               bglen=bglen,
                               bgorder=bgorder,
                               flareparams={'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)},
                               noisepoly=True,
                               noiseimpulse=True,
                               noiseimpulseparams={'t0': (0, (bglen-1.)*flarelc.dt(), bglen)},
                               noiseexpdecay=True,
                               noiseexpdecayparams={'tauexp': (0.0, 0.25*60*60, 3)},
                               noiseexpdecaywithreverse=True,
                               ignoreedges=True,
                               noiseestmethod='tailveto',
                               tvsigma=1.0 )

    lnO, tst = Or.oddsratio()

    maxlnO = max(lnO) # get maximum value from time series

    if maxlnO != -np.inf: # don't add on if -infinity!
      Bfs.append(maxlnO) # add to Bfs

      print "Simulation %d has maximum Bayes factor of %lf" % (i+1, maxlnO)

      if opts.outdata:
        if gzipped:
          f = gzip.open(opts.outdata, "a")
        else:
          f = open(opts.outdata, "a")

        f.write("%lf" % maxlnO)

        if dosinusoids:
          f.write("\t%le\t%lf\n" % (freqs[i], amps[i]))
          print "Sinusoid frequency %lf (1/day), sinusoid amplitude %lf" % (freqs[i]*86400., amps[i])
        else:
          f.write("\n")
        f.close()

    # delete flare
    del flarelc
    del lnO

  # histogram values
  n, bins = np.histogram(Bfs, bins=len(Bfs), density=True)

  # get cumulative distributions of values
  cs = np.cumsum(n/n.sum())*100. # convert into percentage

  binmids = bins[:-1]+(np.diff(bins)[0]/2.)

  # use linear interpolation to find the value at FAR
  csu, ui = np.unique(cs, return_index=True)
  intf = interp1d(csu, binmids[ui], kind='linear')

  threshold = intf(100.-far)

  print "For a FAR of %f %% the log(odds ratio) threshold is %f." % (far, threshold)

  # write out data
  if opts.outdata is not None:
    base = os.path.splitext(opts.outdata)

    # get output image filename
    imageout = basename+'.png'

    # set matplotlib defaults
    mplparams = { \
      'backend': 'Agg',
      'text.usetex': True, # use LaTeX for all text
      'axes.linewidth': 0.5, # set axes linewidths to 0.5
      'axes.grid': True, # add a grid
      'grid.linewidth': 0.5,
      'font.family': 'serif',
      'font.size': 14 }

    matplotlib.rcParams.update(mplparams)

    fig = pl.figure(figsize=(6,5),dpi=200)
    # plot cumulative distribution and threshold value
    if opts.logplot:
      pl.semilogy(binmids, cs, 'b', [binmids[0], binmids[-1]], [100.-far, 100.-far], 'r', [threshold, threshold], [cs[0], 100.], 'r')
      pl.ylim((cs[0], 100.))
    else:
      pl.plot(binmids, cs, 'b', [binmids[0], binmids[-1]], [100.-far, 100.-far], 'r', [threshold, threshold], [0., 100.], 'r')
      pl.ylim((0., 100.))

    pl.xlim((binmids[0], binmids[-1]))

    pl.ylabel(r'Cumulative Probability (\%)', fontsize=14, fontweight=100)
    pl.xlabel(r'log(Odds Ratio)', fontsize=14, fontweight=100)
    fig.subplots_adjust(left=0.18, bottom=0.15) # adjust size

    try:
      fig.savefig(imageout)
    except:
      print >> sys.stderr, "Error outputting figure %s" % imageout
      sys.exit(-1)

    fig.clf()
    pl.close(fig)
