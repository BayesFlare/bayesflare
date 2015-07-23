import numpy as np
import matplotlib
import matplotlib.pyplot as pl
from math import log
import matplotlib.cm as cm
import bayesflare as bf
from matplotlib.mlab import find
import sys
import os
from copy import copy

tlength = 2893536. #length of time data
tstep = 1765.55929 #time data step
nstd = 1. #noise standard deviation
bglen = 55 #background window length (must be odd) so there is always a bin in the centre
bgorder = 4 #The polynomial order of the fitted background variability
taug = 1800 #Gaussian rise width of injected flare (sec)
taue = 3600 #Exponential decay timescale of injected flare (sec)
noiseest = 'powerspectrum'
kneevalue = 0.00003858
psest = 0.5 #fraction of the spectrum with which to estimate the noise
nsinusoids = 0 # number sinusions to fit
lowlim = None
uplim = None
threshold = None
timemin = -1
timemax = -.1

lconly = False #or True
injamp = 50 #change this (injected flare amplitude)
detrendmeth = 'savitzkygolay' #'savitzkygolay''highpassfilter''runningmedian''supersmoother' filter used
alpha = None #can be None or from 0 to 10 smoothest is 10 (for super smoother)
oinj = None #or false

ts = np.arange(0., tlength, tstep, dtype='float64')
flarelc = bf.Lightcurve()
flarelc.clc = nstd*np.random.randn(len(ts)) #clc y data
flarelc.cts = np.copy(ts) #cts time stamp data
flarelc.cadence = 'long'

# set amplitude priors to be large
# largeprior = 1.e6 # 1 million!
# amppriors = (np.ones(bgorder+2)*largeprior).tolist() #array of 1s 6 long, * million, array 6 long each entry has 1 million in. turns array into list
#tslen = len(ts)-bglen+1 # length of time series with edges removed, (removed half a window on each side)

tmi = flarelc.cts-flarelc.cts[0] #makes time data start at 0
Mfi = bf.Flare(tmi, amp=1.) #creates a Flare object
t0 = tmi[int(len(tmi)/2)] #central time of flare set to middle of time data

pdict = {'t0': t0, 'amp': injamp, 'taugauss': taug, 'tauexp': taue}
injdata = np.copy(Mfi.model(pdict)) #creates a flare in the flare object Mfi, using data pdict, copies this into injdata

# if oinj:
#   if oinj != None:
#     oinj = oinj + injdata
#   else:
#     oinj = np.copy(injdat)

flarelc.clc = flarelc.clc + injdata #adds flare model to data

tmpcurve = copy(flarelc)

if detrendmeth == 'savitzkygolay':
	tmpcurve.detrend(method='savitzkygolay', nbins=bglen, order=bgorder)
elif detrendmeth == 'highpassfilter':
	tmpcurve.detrend(method='highpassfilter', knee=kneevalue)
elif detrendmeth == 'runningmedian':
	tmpcurve.detrend(method='runningmedian', nbins=bglen)
elif detrendmeth == 'supersmoother':
	tmpcurve.detrend(method='supersmoother', alpha=alpha)

sig = bf.estimate_noise_ps(tmpcurve, estfrac=psest)[0]

print "Noise estimate with '%s' method = %f" % (noiseest, sig)

# Or = bf.OddsRatioDetector( flarelc,
#                            bglen=bglen,
#                            bgorder=bgorder,
#                            nsinusoids=nsinusoids,
#                            noiseestmethod=noiseest,
#                            psestfrac=psest,
#                            #tvsigma=opts.tvsigma,
#                            flareparams={'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)},
#                            noisepoly=True,
#                            noiseimpulse=True,
#                            noiseimpulseparams={'t0': (0, (bglen-1.)*flarelc.dt(), bglen)},
#                            noiseexpdecay=True,
#                            noiseexpdecayparams={'tauexp': (0.0, 0.25*60*60, 3)},
#                            noiseexpdecaywithreverse=True,
#                            ignoreedges=True )

# lnO, tst = Or.oddsratio()

mplparams = { \
'text.usetex': True, # use LaTeX for all text
'axes.linewidth': 0.5, # set axes linewidths to 0.5
'axes.grid': True, # add a grid
'grid.linewidth': 0.5,
'font.family': 'serif',
'font.size': 16,
'legend.fontsize': 12 }

matplotlib.rcParams.update(mplparams)

fig, axarr = pl.subplots(1, sharex=True)

#tst = (tst-tst[0])/86400.

axarr.plot(flarelc.cts, flarelc.clc, 'b') #[np.arange(int(bglen/2), len(ts)-int(bglen/2))]

# if oinj: # overplot injection
#   axarr.plot(tst, oinj[np.arange(int(bglen/2), len(ts)-int(bglen/2))], 'r')

"""if threshold != None:
  # get above threshold (i.e. flare) points.
  flarelist, Nflares, maxlist = Or.thresholder(lnO, threshold, expand=1)

  ylims = axarr.get_ylim()

  for fl in flarelist:
    fs = tst[fl[0]]
    fe = tst[fl[1]]
    axarr.fill_between([fs, fe], [ylims[0], ylims[0]], [ylims[1], ylims[1]], alpha=0.35, facecolor='k', edgecolor='none')

  axarr.set_ylim((ylims[0], ylims[1]))"""

axarr.set_ylabel('Flux', fontsize=16, fontweight=100)

"""if not lconly:
	axarr[1].plot(tst, lnO, 'k')
	axarr[1].set_ylabel('log(Odds Ratio)', fontsize=16, fontweight=100)
	axarr[1].set_xlabel('Time (days)', fontsize=16, fontweight=100)"""

"""if lowlim != None and uplim != None:
	axarr[1].set_ylim((float(lowlim), float(uplim)))"""

fig.subplots_adjust(hspace=0.075) # remove most space between plots
# remove ticks for all bar the bottom plot
pl.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

tstart = flarelc.cts[0]
tend = flarelc.cts[-1]
if timemin >= 0.:
  tstart = timemin
if timemax >= 0.:
  tend = timemax
axarr.set_xlim((tstart, tend))
#axarr[1].set_xlim((tstart, tend))

pl.show()

fig.clf()
pl.close(fig)