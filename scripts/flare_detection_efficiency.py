#!/usr/bin/env python

"""
Script to calculate efficiency of the flare detection code over a range of
SNR signals given a Bayes factor threshold.

The script can be run with purely white noise, or noise including a random
low-frequency sinusoidal variation, or real Kepler data, with a signal added.
Flare signals, transit signal, or impulses can be added, although the
detection will always try and detect a flare.

The code can also output the false positive rate.
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
from copy import copy

from optparse import OptionParser

__version__= "1.0"

def get_efficicency_curve(injsnrs, detsnrs, snrrange=[0, 50], histbins=50, plothist=False,
                          plotfile=None, logplot=False):
  """
  Using a list of injected SNRs (injsnrs) and detected SNR (detsnrs)
  get the detection efficiency curve by histogramming both distributions
  (using the range given by snrrange and number of bins given by histbins)
  and returning the efficiency for each SNR bin.

  If plothist is true and a valid plotfile output file is given then the
  histogram will be plotted. If logplot is true then the plot will
  have log axes.
  """

  bins = np.linspace(snrranges[0], snrranges[-1], histbins) # histogram bin values

  # histogram the injected SNRs
  nsnrsinj, nbins = np.histogram(injsnrs, bins=bins)

  # histogram the detected SNRs
  nsnrsdet, nbins = np.histogram(detsnrs, bins=bins)

  binmids = bins[:-1]+(np.diff(bins)[0]/2.) # get middle of histogram bins

  efficiency = 100.*nsnrsdet/nsnrsinj

  if plothist and plotfile is not None:
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

    fig = pl.figure(figsize=(5,5),dpi=200)

    # get the efficiency plot
    if logplot:
      pl.loglog(binmids, efficiency, 'b')
      pl.ylim((np.min(efficiency), 100))
      pl.xlabel(r'log(SNR)', fontsize=14, fontweight=100)
    else:
      pl.plot(binmids, efficiency, 'b')
      pl.ylim((0., 100.))
      pl.xlabel(r'SNR', fontsize=14, fontweight=100)

    pl.xlim((binmids[0], binmids[-1]))
    pl.ylabel(r'Efficiency (\%)', fontsize=14, fontweight=100)
    fig.subplots_adjust(left=0.18, bottom=0.15) # adjust size

    try:
      fig.savefig(plotfile)
    except:
      print >> sys.stderr, "Error outputting figure %s" % plotfile
      sys.exit(-1)

    fig.clf()
    pl.close(fig)

  return efficiency, binmids


# main function
if __name__=='__main__':

  usage = "Usage: %prog [options]"

  description = \
  """
This script will calculate the efficiency for detecting a flares as a function of SNR for
a given Bayes factor detection threshold. The efficiency can be calculated using
injections into pure Gaussian noise, Gaussian noise with a sinusoidal variation in it,
or real Kepler data. The injections are flares by default, but can also be set to inject
a transit, or an impulse (although the efficiency for detecting a flare for both of these
cases will still be returned). When calculating when a signal has been detected or not
it requires an above threshold Bayes factor within +/-1 time step of the injected position.

Above threshold crossings that are disconnected from the injection time (i.e. not within
a contiguous block of above-threshold crossings around the injection time) will be counted
as false positives.
  """

  parser = OptionParser( usage = usage, description = description, version = __version__ )

  parser.add_option("-N", "--Nsims", dest="Nsims",
                    help="The number of simulated noise realisations, or real data sets, to \
use [default: %default].", type="int", default=1000)

  parser.add_option("-L", "--file-list", dest="filelist",
                    help="If a file list is given it is assumed to be an ascii file containing \
one Kepler light curve file per line. These are then used instead of simulated data to calculate \
the efficiency.")

  parser.add_option("-T", "--threshold", dest="threshold",
                    help="The log Bayes factor threshold [required].", type="float")

  parser.add_option("-s", "--std", dest="nstd",
                    help="The Gaussian noise standard deviation [default: %default].",
                    type="float", default=1.0)

  parser.add_option("-r", "--snr-min", dest="snrmin",
                    help="The minimum of the signal SNR range from which the injected signals \
will be drawn [default: %default].", type="float", default=0.)

  parser.add_option("-R", "--snr-max", dest="snrmax",
                    help="The maximum of the signal SNR range from which the injected signals \
will be drawn [default: %default].", type="float", default=50.)

  parser.add_option("-c", "--tau-gauss-min", dest="taugaussmin",
                    help="The minimum of the flare Gaussian rise timescales (seconds) from \
which the injected signals will be drawn [default: %default].", type="float", default=0.)

  parser.add_option("-C", "--tau-gauss-max", dest="taugaussmax",
                    help="The maximum of the flare Gaussian rise timescales (seconds) from \
which the injected signals will be drawn [default: %default].", type="float", default=(1.5*60.*60.))

  parser.add_option("-e", "--tau-exp-min", dest="tauexpmin",
                    help="The minimum of the flare exponential decay timescales (seconds) from \
which the injected signals will be drawn [default: %default].", type="float", default=(0.5*60.*60.))

  parser.add_option("-E", "--tau-exp-max", dest="tauexpmax",
                    help="The maximum of the flare exponential decay timescales (seconds) from \
which the injected signals will be drawn [default: %default].", type="float", default=(3.*60.*60.))

  parser.add_option("-K", "--inject-transit", dest="injtransit",
                    help="Inject planetary transit signals rather than the default flares [default: %default].",
                    action="store_true", default=False)

  parser.add_option("-d", "--tau-flat-min", dest="taufmin",
                    help="The minimum of the transit flat-bottom half-width timescale (seconds) from \
which the injected signals will be drawn [default: %default].", type="float", default=(0.5*60.*60.))

  parser.add_option("-D", "--tau-flat-max", dest="taufmax",
                    help="The maximum of the transit flat-bottom half-width timescale (seconds) from \
which the injected signals will be drawn [default: %default].", type="float", default=(4.*60.*60.))

  parser.add_option("-g", "--sigmag-min", dest="sigmagmin",
                    help="The minimum of the transit Gaussian edge timescale (seconds) from \
which the injected signals will be drawn [default: %default].", type="float", default=0.)

  parser.add_option("-G", "--sigmag-max", dest="sigmagmax",
                    help="The maximum of the transit Gaussian edge timescale (seconds) from \
which the injected signals will be drawn [default: %default].", type="float", default=(3.*60.*60.))

  parser.add_option("-I", "--inject-impulse", dest="injimpulse",
                    help="Inject impulse signals rather than the default flares [default: %default].",
                    action="store_true", default=False)

  parser.add_option("-P", "--impulse-negative", dest="impneg",
                    help="Set this to inject impulses with negative signs rather than the default \
positive value.", action="store_true", default=False)

  parser.add_option("-a", "--add-sinusoid", dest="dosinusoids",
                    help="Add a sinusoid to the Gaussian noise [default: %default].",
                    action="store_true", default=False)

  parser.add_option("-m", "--sinusoid-amp-min", dest="ampmin",
                    help="Minimum amplitude (in units of the noise standard deviation) range from \
which the sinusoid amplitudes will be drawn [default: %default].",
                    type="float", default=0.)

  parser.add_option("-M", "--sinusoid-amp-max", dest="ampmax",
                    help="Maximum amplitude (in units of the noise standard deviation) range from \
which the sinusoid amplitudes will be drawn [default: %default].", type="float", default=0.)

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
distribution.", default=None)

  parser.add_option("-n", "--hist-bins", dest="histbins",
                    help="The number of SNR histogram bins for the efficiency plot [default: %default].",
                    type="int", default=50)

  parser.add_option("-z", "--log-efficiency-plot", dest="logplot",
                    help="If outputting the data set this to plot the efficiency on a \
log-scale [default: %default].",
                    action="store_true", default=False)

  # read in arguments
  (opts, args) = parser.parse_args()

  if not opts.threshold:
    parser.error('Threshold value must be given.')
  else:
    threshold = opts.threshold

  # number of simulations
  Nsims = opts.Nsims

  dosinusoids = False

  # check if there's a list of Kepler light curve files
  if not opts.filelist: # no list so simulate the data
    kl = False # no Kepler file list

    dosinusoids = opts.dosinusoids # say whether to add sinusoids

    nstd = opts.nstd # the noise standard deviation

    if dosinusoids:
      ampranges = [opts.ampmin*nstd, opts.ampmax*nstd] # range of amplitudes of sinusoid (as a factor of the noise standard
      freqranges = [opts.freqmin, opts.freqmax] # frequencies ranging from 1/day to 1/month

      amps = ampranges[0] + np.random.rand(Nsims)*np.diff(ampranges) # uniform between ranges
      freqs = freqranges[0] + np.random.rand(Nsims)*np.diff(freqranges)

      # random initial phase between 0 and 2pi
      phase = 2.*np.pi*np.random.rand(Nsims)

    # create times stamps for light curves (the same length as Kepler Q1 data)
    ts = np.arange(0., opts.tlength, opts.tstep, dtype='float64')
  else: # there is a file list
    kl = True

    try:
      f = open(opts.filelist, "r")
    except:
      print >> sys.stderr, "Error... cannot open Kepler file list %s." % opts.filelist

    kfiles = f.readlines()
    f.close()

    Nlc = len(kfiles)

  # get snr range
  snrranges = [opts.snrmin, opts.snrmax]
  snrs = snrranges[0] + np.random.rand(Nsims)*np.diff(snrranges) # uniform over range

  if opts.injtransit: #  injecting a transit
    maxduration = 12.*60.*60 # maximum duration of transit

    # set tau flat range
    taufranges = [opts.taufmin, opts.taufmax]

    # set sigmag ranges
    sigmagranges = [opts.sigmagmin, opts.sigmagmax]

    taufs = np.zeros(Nsims)
    sigmags = np.zeros(Nsims)

    for i in range(Nsims):
      taufstmp = taufranges[0] + np.random.rand(1)*np.diff(taufranges)
      sigmagstmp = sigmagranges[0] + np.random.rand(1)*np.diff(sigmagranges)

      while taufstmp + 4.*sigmagstmp > maxduration:
        taufstmp = taufranges[0] + np.random.rand(1)*np.diff(taufranges)
        sigmagstmp = sigmagranges[0] + np.random.rand(1)*np.diff(sigmagranges)

      taufs[i] = taufstmp
      sigmags[i] = sigmagstmp

  elif opts.injimpulse: # injecting a delta-function impulse
    # get with positive or negative
    if opts.impneg:
      impsign = -1.
    else:
      impsign = 1.

  else: # the default of injecting flares
    # get tau exp (flare decay timescale) range
    tauexpranges = [opts.tauexpmin, opts.tauexpmax]
    tauexps = tauexpranges[0] + np.random.rand(Nsims)*np.diff(tauexpranges)

    # get tau gauss (flare rise timescale) range - must be shorter than the decay time
    taugaussranges = [opts.taugaussmin, opts.taugaussmax]
    taugausss = np.zeros(Nsims)
    for i in range(Nsims):
      taugausstmp = taugaussranges[0] + np.random.rand(1)*np.diff(taugaussranges)

      while taugausstmp > tauexps[i]:
        taugausstmp = taugaussranges[0] + np.random.rand(1)*np.diff(taugaussranges)

      taugausss[i] = taugausstmp

  bglen = opts.bglen
  bgorder = opts.bgorder

  if bglen % 2 == 0:
    print >> sys.stderr, "Error... background length (bglen) must be an odd number"
    sys.exit(-1)

  detectedsignals = [] # a list of tuples for detected signals containing the snr (and if a flare were injected) the taugauss and tauexp values
  falsepositives = 0 # count the false positives

  # create output files
  outd = None
  if opts.outdata is not None:
    outd = True

    base = os.path.splitext(opts.outdata) # split output file name

    if not opts.injtransit and not opts.injimpulse: # do only when injecting a flare
      detsnrs = []

      try:
        fout = open(opts.outdata, "w")
      except:
        print >> sys.stderr, "Error... could not open output file %s." % opts.outdata
        sys.exit(-1)

      fout.close() # will re-open and append data during the loop

      # write out non-detections
      nondetout = base[0]+'_nondetections'+base[1]

      try:
        fnondet = open(nondetout, "w")
      except:
        print >> sys.stderr, "Error... could not open output file %s." % nondetout
        sys.exit(-1)

      fnondet.close()

    # output the injected signal parameter
    paramout = base[0]+'_params'+base[1]

    try:
      fparams = open(paramout, "w")
    except:
      print >> sys.stderr, "Error... could not open output parameter file %s." % paramout
      sys.exit(-1)

    fparams.close() # will re-open and append data during the loop

    # write out false positives
    fpfile = base[0]+'_false_positives'+base[1]

    try:
      ffalse = open(fpfile, "w")
    except:
      print >> sys.stderr, "Error... could not open output file %s." % fpfile
      sys.exit(-1)

    ffalse.close()

  # perform loop
  for i in range(Nsims):
    print "Injection %d with SNR %f" % (i+1, snrs[i])

    # create data containing a flare and white noise
    if kl:
      # if Nsims is greater than the number of lightcurve input then repeat the use of
      # the curves
      fnum = i % Nlc

      if os.path.isfile(kfiles[fnum].strip()):
        flarelc = bf.Lightcurve(curve=kfiles[fnum].strip())
      else:
        print >> sys.stderr, "Error... file in list (%s) does not exist." % kfiles[fnum].strip()
        sys.exit(-1)

      ts = np.copy(flarelc.cts)
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

    # get noise standard deviation from a detrended lightcurve
    tmpcurve = copy(flarelc)
    tmpcurve.detrend(method='savitzkygolay', nbins=bglen, order=bgorder)
    #sk = bf.estimate_noise_ps(tmpcurve, estfrac=0.5)[0] # noise standard deviation
    sk = bf.estimate_noise_tv(tmpcurve.clc, sigma=1.0)[0]
    del tmpcurve

    # set central time of the injection randomly (but not within bglen/2 of data edges)
    idxt0 = np.random.randint(int(bglen/2), len(ts)-int(bglen/2)-1)
    t0 = flarelc.cts[idxt0]

    if outd:
      fparams = open(paramout, 'a') # open output parameter file for appending

    # create injections
    if opts.injtransit: # create a transit
      Mti = bf.Transit(flarelc.cts, amp=1)
      pdict = {'t0': t0, 'amp': 1., 'sigmag': sigmags[i], 'tauf': taufs[i]}
      injdata = np.copy(Mti.model(pdict))

      # output the transit parameters
      if outd:
        fparams.write("%f\t%f\t%f\n" % (snrs[i], sigmags[i], taufs[i]))

      del Mti
    elif opts.injimpulse: # create a delta-function impulse
      Mii = bf.Impulse(flarelc.cts, amp=1)
      pdict = {'t0': t0, 'amp': 1.*impsign}
      injdata = np.copy(Mii.model(pdict))

      # output the impulse parameters
      if outd is not None:
        fparams.write("%f\t%d\n" % (snrs[i], impsign))

      del Mii
    else:
      Mfi = bf.Flare(flarelc.cts, amp=1)
      pdict = {'t0': t0, 'amp': 1., 'taugauss': taugausss[i], 'tauexp': tauexps[i]}
      injdata = np.copy(Mfi.model(pdict))

      # output the flare parameters
      if outd is not None:
        fparams.write("%d\t%d\t%f\t%f\t%f" % (i, idxt0, snrs[i], taugausss[i], tauexps[i]))

        if dosinusoids:
          fparams.write("\t%e\t%f" % (freqs[i], amps[i]))

        fparams.write("\n")

      del Mfi

    if outd:
      fparams.close() # close parameter file

    # get the signal SNR (assumes the noise standard deviation is constant across the lightcurve)
    injsnr = (1./sk)*np.sqrt(np.sum(injdata**2))

    # add the injection to the data, rescaling to the required SNR
    flarelc.clc = flarelc.clc + injdata*(snrs[i]/injsnr)

    #fig = pl.figure(figsize=(10,5))
    #pl.plot(flarelc.cts, flarelc.clc)
    #fig.savefig('flaretimeseries.png')
    #fig.clf()
    #pl.close(fig)

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

    # find points above threshold
    lnOa = np.array(lnO)
    tstruncated = tst # times with edges removed

    tmpdetected = matplotlib.mlab.find(lnOa > threshold) # get values above the threshold
    #print v

    # delete some stuff
    # delete flare
    del lnO

    # get list of contiguous above threshold segments
    segidxs = []
    if len(tmpdetected) == 1:
      tmpseg = np.array([tmpdetected[0]], dtype='int32')
      tmpseg = np.concatenate(([tmpdetected[0]-2, tmpdetected[0]-1], tmpseg))
      tmpseg = np.concatenate((tmpseg, [tmpdetected[0]+1, tmpdetected[0]+2]))
      # make sure seg doesn't contain any < 0 values or > len(lnOa) values
      tmpseg = tmpseg[tmpseg > -1]
      tmpseg = tmpseg[tmpseg < len(tstruncated)]
      segidxs = [tmpseg]
    elif len(tmpdetected) > 1:
      dd = np.diff(tmpdetected)

      contlist = [tmpdetected[0]]
      for j, val in enumerate(dd):
        if j < len(dd)-1:
          if val == 1:
            contlist.append(tmpdetected[j+1])
          else:
            # add +/- two timesteps to each segment (i.e. allow flare central time to be within an hour of the segment)
            tmpseg = np.array(contlist, dtype='int32')
            tmpseg = np.concatenate(([contlist[0]-2, contlist[0]-1], contlist))
            tmpseg = np.concatenate((tmpseg, [contlist[-1]+1, contlist[-1]+2]))
            tmpseg = tmpseg[tmpseg > -1]
            tmpseg = tmpseg[tmpseg < len(tstruncated)]
            segidxs.append(tmpseg)
            contlist = [tmpdetected[j+1]]
        else: # fill in final value
          tmpseg = np.array(contlist, dtype='int32')
          tmpseg = np.concatenate(([contlist[0]-2, contlist[0]-1], tmpseg))
          tmpseg = np.concatenate((tmpseg, [contlist[-1]+1, contlist[-1]+2]))
          tmpseg = tmpseg[tmpseg > -1]
          tmpseg = tmpseg[tmpseg < len(tstruncated)]
          segidxs.append(tmpseg)

    if len(segidxs) == 0:
      print "No flare detected!"

      # output parameters of signals that weren't detected
      if outd:
        if not opts.injtransit and not opts.injimpulse:
          fnondet = open(nondetout, 'a')
          fnondet.write("%d\t%d\t%f\t%f\t%f\n" % (i, idxt0, snrs[i], taugausss[i], tauexps[i]))
          fnondet.close()

      continue
    elif len(segidxs) > 1:
      # check for overlapping or adjacent segments, and merge them
      j = 0
      newsegs = []
      while True:
        thisseg = segidxs[j]
        j = j+1
        for k in range(j, len(segidxs)):
          nextseg = segidxs[k]
          if thisseg[-1] >= nextseg[0]-1: # overlapping or adjacent segment
            thisseg = np.arange(thisseg[0], nextseg[-1]+1, dtype='int32')
            j = j+1
          else:
            break

        newsegs.append(thisseg)

        if j >= len(segidxs):
          break

      segidxs = list(newsegs) # copy new list
      del newsegs

    if outd:
      tmpdetlen = len(detsnrs)

    # go through contiguous segments and see if any match an injected flare to count as detected, otherwise each
    # segments if a false positive
    for seg in segidxs:
      if not opts.injtransit and not opts.injimpulse: # i.e. a flare was injected
        idxt0shift = idxt0-int(bglen/2) # the flare time shifted into the truncated time series indexes

        if idxt0shift in seg: # we've found a flare!
          print "Flare detected!"
          detectedsignals.append((i, idxt0, snrs[i], taugausss[i], tauexps[i], np.amax(lnOa[seg])))
          if outd: # output detected signal data (append to output)
            fout = open(opts.outdata, 'a')
            fout.write("%d\t%d\t%f\t%f\t%f\t%f\n" % (i, idxt0, snrs[i], taugausss[i], tauexps[i], np.amax(lnOa[seg])))
            fout.close()
            detsnrs.append(snrs[i]) # get the SNR of the detected signals
        else: # a false positive
          print "A false positive!"
          idxmax = np.argmax(lnOa[seg]) # get index of maximum odds ratio in segment
          if outd:
            ffalse = open(fpfile, 'a')
            ffalse.write("%d\t%d\t%d\t%f\n" % (i, seg[idxmax]+int(bglen/2), idxt0, lnOa[seg[idxmax]]))
            ffalse.close()
          falsepositives = falsepositives + 1
      else: # a transit or impulse was detected, so anything is a false positive
        print "A false positive!"
        idxmax = np.argmax(lnOa[seg])
        if outd:
          ffalse = open(fpfile, 'a')
          ffalse.write("%d\t%d\t%d\t%f\n" % (i, seg[idxmax]+int(bglen/2), idxt0, lnOa[seg[idxmax]]))
          ffalse.close()
        falsepositives = falsepositives + 1

    if outd:
      # check if potential detections where actually false positives
      if tmpdetlen == len(detsnrs):
        if not opts.injimpulse and not opts.injtransit:
          fnondet = open(nondetout, 'a')
          fnondet.write("%d\t%d\t%f\t%f\t%f\n" % (i, idxt0, snrs[i], taugausss[i], tauexps[i]))
          fnondet.close()

    #fig = pl.figure(figsize=(10,5))
    #fig, axarr = pl.subplots(2, sharex=True)
    #tst = (tstruncated-tstruncated[0])/86400.
    #axarr[0].plot(tst, flarelc.clc[np.arange(int(bglen/2), len(ts)-int(bglen/2))], 'b')
    #axarr[1].plot(tst, lnOa)
    #axarr[0].set_xlim((tst[0], tst[-1]))
    #axarr[1].set_ylim((-50., 100.))
    if dosinusoids:
      print "Sinusoid frequency %e, sinusoid amplitude %f, inj. amp. = %f" % (freqs[i], amps[i], snrs[i]/injsnr)
    #pl.show()
    #fig.savefig('bayestimeseries.png')
    #fig.clf()
    #pl.close(fig)

    # delete flare
    del flarelc

    # delete segments
    del segidxs

  # output the efficiency data
  if outd:
    print "Total injections = %d" % Nsims

    if not opts.injtransit and not opts.injimpulse: # i.e. a flare was injected
      # write out detection data and produce an efficiency plot
      print "Total detected = %d" % len(detectedsignals)

      # get efficiency curve
      efficiency, binmids = get_efficicency_curve(snrs, detsnrs, snrrange=snrranges,
                                                  histbins=opts.histbins, plothist=True,
                                                  plotfile=base[0]+'.png',
                                                  logplot=opts.logplot)

      # output efficiency data
      efffile = base[0]+'_efficiency'+base[1]

      try:
        f = open(efffile, "w")
      except:
        print >> sys.stderr, "Error... could not open output efficiency file %s." % efffile
        sys.exit(-1)

      for i in range(len(binmids)):
        f.write("%f\t%f\n" % (binmids[i], efficiency[i]))

      f.close()

    print "Total false positives = %d" % falsepositives
