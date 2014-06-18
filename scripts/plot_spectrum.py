#!/usr/bin/env python

"""
A script to plot a light curve's power spectrum. If a Kepler ID is supplied
then the Kepler data for that star is used.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as pl
from math import log
import bayesflare as bf
import sys
import os
from copy import copy

from optparse import OptionParser

__version__= "1.0"

# main function
if __name__=='__main__':

  usage = "Usage: %prog [options]"

  description = \
  """
     This script will plot the power spectrum for a given Kepler lightcurve, or
     simulated data.
  """

  parser = OptionParser( usage = usage, description = description, version = __version__ )

  parser.add_option("-L", "--kepler-dir", dest="filedir",
                    help="A directory of Kepler data files")

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
                    type="float", default=(1./86400.))

  parser.add_option("-l", "--length", dest="tlength",
                    help="The length (in seconds) for each simulated time series [default: %default \
(the length of Kepler Q1 data)].", type="float", default=2893536.)

  parser.add_option("-p", "--time-step", dest="tstep",
                    help="The time interval between points (in seconds) for the time series \
[default: %default (the interval for Kepler Q1 data)].", type="float", default=1765.55929)

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

  parser.add_option("-u", "--overplot-inj",  dest="oinj",
                    help="Overplot the injection (and sinusoid) on the data.",
                    action="store_true", default=False)

  parser.add_option("-O", "--output-file",  dest="outfile",
                    help="Output the figure to this given file (e.g. lightcurve.png(pdf)).",
                    default=None)

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

  parser.add_option("-b", "--background-length", dest="bglen",
                    help="The length (in number of data) of the running background window used \
in the Bayes factor calculation (must be an odd number) [default: %default (about 12 hours \
with the default time step)].", type="int", default=37)

  parser.add_option("-o", "--background-order", dest="bgorder",
                    help="The polynomial order of the fitted background variability [default: %default].",
                    type="int", default=4)

  parser.add_option("-y", "--overplot-noise", dest="onoise",
                    help="Overplot the noise estimate [default: %default].",
                    action="store_true", default=False)

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
    ts = np.arange(0., opts.tlength, opts.tstep, dtype='float32')

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
  else: # there is a file list
    kl = True

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

    try:
      flarelc = bf.Lightcurve(curve=kfile)
    except:
      print >> sys.stderr, "Error... could not open light curve file %s" % kfile
      sys.exit(0)

    ts = np.copy(flarelc.cts)

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

      flarelcinj = bf.Lightcurve()
      flarelcinj.clc = np.copy(oinj)
      flarelcinj.cts = np.copy(ts)
      flarelcinj.cle = np.zeros(len(ts))
      flarelcinj.cadence = 'long'

    flarelc.clc = flarelc.clc + injdata

  if opts.onoise:
    noiseest = opts.noisemethod
    tmpcurve = copy(flarelc)
    tmpcurve.detrend(opts.bglen, opts.bgorder)
    if noiseest == 'powerspectrum':
      sig = bf.estimate_noise_ps(tmpcurve, estfrac=opts.psest)[0]
    elif noiseest == 'tailveto':
      sig = bf.estimate_noise_tv(tmpcurve.clc, sigma=opts.tvsigma)[0]
    else:
      print "Error... noise estimation method %s not recognised." % noiseest
      sys.exit(0)

    detrendedsk, f = tmpcurve.psd()

    # convert back to one-sided power spectral density
    sig = 2.*(sig**2)/flarelc.fs()

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
  fig, axarr = pl.subplots()
  if opts.outfile:
    fig.set_dpi(200.)
    fig.set_figheight(5.)
    fig.set_figwidth(6.)

  # set times to plot
  sk, f = flarelc.psd()

  f = f*1.e6 # convert to microHz

  axarr.semilogy(f, sk, 'b', label='Original data')

  if opts.oinj and opts.injflare: # overplot injection
    sk, f = flarelcinj.psd()
    f = f*1.e6 # convert to microHz
    axarr.semilogy(f, sk, 'r', label='Signal')

  if opts.onoise: # overplot noise estimate
    print "Noise power spectral density = %f" % sig
    axarr.plot(f, detrendedsk, 'k', label='Filtered data')
    axarr.plot([f[0], f[-1]], [sig, sig], 'k--', label='Noise estimate')

  axarr.set_xlim((f[0], f[-1]))
  axarr.set_ylabel('Power spectral density', fontsize=16, fontweight=100)
  axarr.set_xlabel('Frequency ($\mu$Hz)', fontsize=16, fontweight=100)

  legend = axarr.legend()

  if opts.outfile:
    try:
      fig.savefig(opts.outfile)
    except:
      print >> sys.stderr, "Error... could not output figure to %s" % opts.outfile
      sys.exit(0)
  else: # show figure
    pl.show()

  fig.clf()
  pl.close(fig)
