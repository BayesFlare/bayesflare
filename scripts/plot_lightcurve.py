#!/usr/bin/env python

"""
A script to plot a light curve and the associated log odds ratio. If a Kepler ID is supplied, and a
directory containing data from a Kepler release Quarter is given, then the Kepler data for that star
is used.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as pl
from math import log
import matplotlib.cm as cm
import bayesflare as bf
from matplotlib.mlab import find
import sys
import os
from copy import copy, deepcopy

from optparse import OptionParser

__version__= "1.0"

# main function
if __name__=='__main__':

  usage = "Usage: %prog [options]"

  description = \
  """
     Plot the light curve for a given Kepler star, or simulated data set, and the log odds ratio
     for flare detection.
  """

  parser = OptionParser( usage = usage, description = description, version = __version__ )

  parser.add_option("-L", "--kepler-dir", dest="filedir",
                    help="A directory of Kepler data files or an individual file.")

  parser.add_option("-k", "--kepler-id", dest="kid", help="A Kepler star ID", type="int")

  parser.add_option("-C", "--cadence", dest="cadence",
                    help="The Kepler light curve cadence [default: %default]",
                    default='long')

  parser.add_option("-s", "--std", dest="nstd",
                    help="Simulated Gaussian noise standard deviation [default: %default].",
                    type="float", default=1.0)

  parser.add_option("-a", "--add-sinusoid", dest="dosinusoids",
                    help="Add a sinusoid to the Gaussian noise [default: %default].",
                    action="store_true", default=False)

  parser.add_option("-m", "--sinusoid-amp", dest="amp",
                    help="Amplitude of sinusoid (in units of the noise standard deviation) \
[default: %default].", type="float", default=0.)

  parser.add_option("-f", "--sinusoid-freq", dest="freq",
                    help="Frequency of sinusoid (Hz) range [default: %default (=1/day)].",
                    type="float", default=(1./(2.*86400.)))

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

  parser.add_option("-S", "--num-sinusoids", dest="nsinusoids",
                    help="The number of sinusoids to fit as a background [default: %default].",
                    type="int", default=0)

  parser.add_option("-I", "--inject-flare", dest="injflare",
                    help="Inject a flare.", action="store_true", default=False)

  parser.add_option("-A", "--inj-amp", dest="injamp",
                    help="Amplitude of injected flare [default: %default].",
                    type="float", default=0.)

  parser.add_option("-g", "--inj-taug", dest="taug",
                    help="Gaussian rise width of injected flare (sec) [default: %default].",
                    type="float", default=1800.)

  parser.add_option("-e", "--inj-taue", dest="taue",
                    help="Exponential decay timescale of injected flare (sec) [default: %default].",
                    type="float", default=3600.)

  parser.add_option("-t", "--inj-t0", dest="t0",
                    help="Central time of injected flare (sec from start) [default: middle of time series].",
                    type="float", default=-1.)

  parser.add_option("-n", "--noise-method", dest="noisemethod",
                    help="The noise estimation method, either 'powerspectrum', or \
'tailveto' [default: %default].", default='powerspectrum')

  parser.add_option("-P", "--ps-frac", dest="psest",
                    help="If using 'powerspectrum' method this is the fraction of \
the spectrum with which to estimate the noise [default: %default].",
                    type="float", default=0.5)

  parser.add_option("-v", "--tv-sigma", dest="tvsigma",
                    help="If using 'tailveto' method this is the number \
of standard devaitons with which to estimate the noise [default: %default].",
                    type="float", default=2.5)

  parser.add_option("-u", "--overplot-inj",  dest="oinj",
                    help="Overplot the injection (and sinusoid) on the data.",
                    action="store_true", default=False)

  parser.add_option("-O", "--output-file",  dest="outfile",
                    help="Output the figure to this given file (e.g. lightcurve.png(pdf)).",
                    default=None)

  parser.add_option("-z", "--lnO-lower-limit", dest="lowlim",
                    help="Lower limit for the log(odd ratio) plot [default: None]",
                    default=None)

  parser.add_option("-Z", "--lnO-upper-limit", dest="uplim",
                    help="Upper limit for the log(odd ratio) plot [default: None]",
                    default=None)

  parser.add_option("-w", "--time-min", dest="timemin",
                    help="Set the start time for the output plots (days from start of light curve) \
[default: None]", type="float", default=-1.)

  parser.add_option("-W", "--time-max", dest="timemax",
                    help="Set the end time for the output plots (days from start of light curve) \
[default: None]", type="float", default=-.1)

  parser.add_option("-c", "--lightcurve-only", dest="lconly",
                    help="Only plot the lightcurve, without the odds ratio",
                    action="store_true", default=False)

  parser.add_option("-T", "--threshold", dest="threshold",
                    help="Find odds ratios above this value and overplot.",
                    type="float", default=None)

  # read in arguments
  (opts, args) = parser.parse_args()

  oinj = None

  # check if there's a list of Kepler light curve files
  if not opts.__dict__['filedir']: # no list so simulate the data
    kl = False # no Kepler file list

    dosinusoids = opts.dosinusoids # say whether to add sinusoids

    nstd = opts.nstd # the noise standard deviation

    if dosinusoids:
      amp = opts.amp
      freq = opts.freq

      # random initial phase between 0 and 2pi
      phase = 2.*np.pi*np.random.rand(1)

    # create times stamps for light curves (the same length as Kepler Q1 data)
    ts = np.arange(0., opts.tlength, opts.tstep, dtype='float64')

    flarelc = bf.Lightcurve()

    flarelc.clc = nstd*np.random.randn(len(ts))
    flarelc.cts = np.copy(ts)
    flarelc.cle = np.zeros(len(ts))
    flarelc.cadence = 'long'

    if dosinusoids: # add sinusoid
      sinewave = amp*np.sin(2.*np.pi*freq*ts + phase)

      if opts.oinj:
        oinj = np.copy(sinewave)

      flarelc.clc = flarelc.clc + sinewave
  else: # there is a directory or file
    kl = True

    if os.path.isdir(opts.filedir): # a directory of files
      try:
        filelist = os.listdir(opts.filedir)
      except:
        print >> sys.stderr, "Error... no Kepler data directory found"
        sys.exit(0)

      # find lightcurve for given Kepler ID
      if not opts.kid:
        print >> sys.stderr, "Error... no Kepler star ID given"
        sys.exit(0)

      kids = '%09d' % opts.kid
      kfile = ''
      for f in filelist:
        if kids in f:
          # check for cadence of data file
          if opts.cadence == 'long':
            if 'llc' in f:
              kfile = os.path.join(opts.filedir, f)
              break
          elif opts.cadence == 'short':
            if 'slc' in f:
              kfile = os.path.join(opts.filedir, f)
              break

      if kfile == '':
        print >> sys.stderr, "Error... no light curve file found for KID%d" % kids
        sys.exit(0)
    elif os.path.isfile(opts.filedir): # there is just a file
      kfile = opts.filedir
    else:
      print >> sys.stderr, "Error... lightcurve directory or file does not exist"
      sys.exit(0)

    try:
      flarelc = bf.Lightcurve(curve=kfile)
    except:
      print >> sys.stderr, "Error... could not open light curve file %s" % kfile
      sys.exit(0)

    ts = np.copy(flarelc.cts)

  bglen = opts.bglen
  bgorder = opts.bgorder
  nsinusoids = opts.nsinusoids

  if bglen % 2 == 0:
    print >> sys.stderr, "Error... background length (bglen) must be an odd number"
    sys.exit(0)

  # set amplitude priors to be large
  largeprior = 1.e6 # 1 million!
  amppriors = (np.ones(bgorder+2)*largeprior).tolist()

  tslen = len(ts)-bglen+1 # length of time series with edges removed

  # inject flare
  if opts.injflare:
    tmi = flarelc.cts-flarelc.cts[0]
    Mfi = bf.Flare(tmi, amp=1.)

    if opts.t0 == -1.:
      t0 = tmi[int(len(tmi)/2)]
    else:
      t0 = opts.t0

    # create flare
    pdict = {'t0': t0, 'amp': opts.injamp, 'taugauss': opts.taug, 'tauexp': opts.taue}
    injdata = np.copy(Mfi.model(pdict))

    if opts.oinj:
      if oinj != None:
        oinj = oinj + injdata
      else:
        oinj = np.copy(injdata)

    flarelc.clc = flarelc.clc + injdata

  # output different noise estimates
  noiseest = opts.noisemethod
  tmpcurve = deepcopy(flarelc)
  tmpcurve.detrend(method='savitzkygolay', nbins=bglen, order=bgorder)
  if noiseest == 'powerspectrum':
    sig = bf.estimate_noise_ps(tmpcurve, estfrac=opts.psest)[0]
  elif noiseest == 'tailveto':
    sig = bf.estimate_noise_tv(tmpcurve.clc, sigma=opts.tvsigma)[0]
  else:
    print "Error... noise estimation method %s not recognised." % noiseest
    sys.exit(0)

  print "Noise estimate with '%s' method = %f" % (noiseest, sig)

  if not opts.lconly:
    # get the odds ratio
    Or = bf.OddsRatioDetector( flarelc,
                               bglen=bglen,
                               bgorder=bgorder,
                               nsinusoids=nsinusoids,
                               noiseestmethod=noiseest,
                               psestfrac=opts.psest,
                               tvsigma=opts.tvsigma,
                               flareparams={'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)},
                               noisepoly=True,
                               noiseimpulse=True,
                               noiseimpulseparams={'t0': (0, (bglen-1.)*flarelc.dt(), bglen)},
                               noiseexpdecay=True,
                               noiseexpdecayparams={'tauexp': (0.0, 0.25*60*60, 3)},
                               noiseexpdecaywithreverse=True,
                               ignoreedges=True )

    lnO, tst = Or.oddsratio()

  # set matplotlib defaults
  mplparams = { \
    'text.usetex': True, # use LaTeX for all text
    'axes.linewidth': 0.5, # set axes linewidths to 0.5
    'axes.grid': True, # add a grid
    'grid.linewidth': 0.5,
    'font.family': 'serif',
    'font.size': 16,
    'legend.fontsize': 12 }

  matplotlib.rcParams.update(mplparams)

  # plot data
  if opts.lconly:
    fig, axarr = pl.subplots(1)
  else:
    fig, axarr = pl.subplots(2, sharex=True)

  if opts.outfile:
    fig.set_dpi(200.)

    if opts.lconly:
      fig.set_figheight(4.)
    else:
      fig.set_figheight(6.)

    fig.set_figwidth(10.)

  # set times to plot
  if opts.lconly:
    tst = (flarelc.cts-flarelc.cts[0])/86400.
    axarr.plot(tst, flarelc.clc, 'b')

    if opts.oinj:
      axarr.plot(tst, oinj, 'r')

    if opts.threshold != None:
      # get above threshold (i.e. flare) points.
      flarelist, Nflares, maxlist = Or.thresholder(lnO, opts.threshold, expand=1)

      ylims = axarr.get_ylim()

      for fl in flarelist:
        fs = tst[fl[0]]
        fe = tst[fl[1]]
        axarr.fill_between([fs, fe], [ylims[0], ylims[0]], [ylims[1], ylims[1]], alpha=0.25, facecolor='k')

      axarr.set_ylim((ylims[0], ylims[1]))

    axarr.set_ylabel('Flux', fontsize=16, fontweight=100)
  else:
    tst = (tst-tst[0])/86400.

    axarr[0].plot(tst, flarelc.clc[np.arange(int(bglen/2), len(ts)-int(bglen/2))], 'b')

    if opts.oinj: # overplot injection
      axarr[0].plot(tst, oinj[np.arange(int(bglen/2), len(ts)-int(bglen/2))], 'r')

    if opts.threshold != None:
      # get above threshold (i.e. flare) points.
      flarelist, Nflares, maxlist = Or.thresholder(lnO, opts.threshold, expand=1)

      ylims = axarr[0].get_ylim()

      for fl in flarelist:
        fs = tst[fl[0]]
        fe = tst[fl[1]]
        axarr[0].fill_between([fs, fe], [ylims[0], ylims[0]], [ylims[1], ylims[1]], alpha=0.35, facecolor='k', edgecolor='none')

      axarr[0].set_ylim((ylims[0], ylims[1]))

    axarr[0].set_ylabel('Flux', fontsize=16, fontweight=100)

  if not opts.lconly:
    axarr[1].plot(tst, lnO, 'k')
    axarr[1].set_ylabel('log(Odds Ratio)', fontsize=16, fontweight=100)
    axarr[1].set_xlabel('Time (days)', fontsize=16, fontweight=100)

    if opts.lowlim != None and opts.uplim != None:
      axarr[1].set_ylim((float(opts.lowlim), float(opts.uplim)))

    fig.subplots_adjust(hspace=0.075) # remove most space between plots
    # remove ticks for all bar the bottom plot
    pl.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    tstart = tst[0]
    tend = tst[-1]
    if opts.timemin >= 0.:
      tstart = opts.timemin
    if opts.timemax >= 0.:
      tend = opts.timemax
    axarr[0].set_xlim((tstart, tend))
    axarr[1].set_xlim((tstart, tend))
  else:
    axarr.set_ylabel('log(Odds Ratio)', fontsize=16, fontweight=100)
    axarr.set_xlabel('Time (days)', fontsize=16, fontweight=100)

    tstart = tst[0]
    tend = tst[-1]
    if opts.timemin >= 0.:
      tstart = opts.timemin
    if opts.timemax >= 0.:
      tend = opts.timemax
    axarr.set_xlim((tstart, tend))

  if opts.outfile:
    try:
      fig.savefig(opts.outfile)
    except:
      print >> sys.stderr, "Error... could not output figure to %s" % opts.outfile
      sys.exit(0)
  else:
    pl.show()

  fig.clf()
  pl.close(fig)
