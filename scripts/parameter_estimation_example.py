#!/usr/bin/env python

"""
   Plot an example of parameter estimation for a simulated signal and in fake data.
"""

import matplotlib

import matplotlib.pyplot as pl
import numpy as np
import bayesflare as bf
import sys

# set up matplotlib output format
mplparams = {
  'text.usetex': True, # use LaTeX for all text
  'axes.linewidth': 0.5, # set axes linewidths to 0.5
  'axes.grid': False, # add a grid
  'grid.linewidth': 0.5,
  'font.family': 'serif',
  'font.size': 16,
  'legend.fontsize': 12 }

matplotlib.rcParams.update(mplparams)

# set fake light curve time stamps
ts = np.arange(0., 2893536., 1765.55929, dtype='float64')
nstd = 1. # noise standard deviation for fake light curve

# create fake light curve 
flarelc = bf.Lightcurve()
flarelc.cts = np.copy(ts)
flarelc.clc = nstd*np.random.randn(len(ts))
flarelc.cle = np.zeros(len(ts))
flarelc.cadence = 'long'

# create flare
Mfi = bf.Flare(flarelc.cts, amp=1.)
t0 = 400000.  # central time
amp = 20.0    # amplitude
taug = 1060.0 # Gaussian rise
taue = 2168.0 # exponential decay
injvals = {'amp': amp, 'taugauss': taug, 'tauexp': taue, 't0': t0}
injdata = np.copy(Mfi.model(injvals))
idxt0 = int(t0/flarelc.dt()) # get index of centre of flare

valname = {'amp': 'Amplitude', 'taugauss': '$\\tau_g$ (hours)', 'tauexp': '$\\tau_e$ (hours)', 't0': '$t_0$ (hours)'}
convfac = {'amp': 1., 'taugauss': 1./3600., 'tauexp': 1./3600., 't0': 1./3600.} # convert output times to hours for plots
itemorder = {'amp': 0, 'taugauss': 1, 'tauexp': 2, 't0': 3}

flarelc.clc = flarelc.clc + injdata

# set up parameter estimation class
pe = bf.ParameterEstimationGrid('flare', flarelc)

pe.lightcurve_chunk(idxt0, 55) # cut out chunk of light curve around the flare

# set amp range (use max - min range of flare position to lightcurve chunk)
amprange = (0., 1.6*(np.amax(pe.lightcurve.clc) - np.amin(pe.lightcurve.clc)), 40)
# here the t0 parameter is not being estimated and is fixed at the injection value
pe.set_grid(ranges={'taugauss': (0., 3600., 40), 'tauexp': (0., 12000., 40), 'amp': amprange, 't0': (t0,)})

pe.calculate_posterior(bgorder=3) # just have a cubic polynomial background
pe.marginalise_all()

# calculate the injected signal-to-noise ratio
injsnr = np.sqrt(np.sum(injdata**2))/nstd

# print out injected signal-to-noise ratio and signal-to-noise ratio of the best fit parameters
print "Injected SNR = %f" % injsnr
print "Recovered SNR = %f" % pe.maximum_posterior_snr()
print pe.maxpostparams

# plot parameters
fig, ax = pl.subplots(1, 3)
fig.set_dpi(200.)
fig.set_figheight(4.)
fig.set_figwidth(10.)

i = 0
for item in pe.paramValues:
  if item != 't0':
    ax[i].plot(pe.paramValues[item]*convfac[item], pe.margposteriors[item], 'b')
    ax[i].plot([injvals[item]*convfac[item], injvals[item]*convfac[item]], [0., 1.1*np.amax(pe.margposteriors[item])], 'k--')
    ax[i].set_ylim((0., 1.1*np.amax(pe.margposteriors[item])))
    ax[i].set_xlabel(valname[item])

    if i == 0:
      ax[i].set_ylabel('Probability density')

    i = i+1

fig.subplots_adjust(wspace=0.075, bottom=0.15) # remove most width between plots
# remove ticks for all bar the left-most plot
pl.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)

fig.canvas.draw() # before we can modify axes text, we need to draw!

# remove final x tick label for first two plots, so that text doesn't overlap
for axi in fig.axes[:-1]:
  xt = axi.get_xticklabels()
  axi.set_xticklabels([item.get_text() for item in xt[:-1]]) #, rotation=45)

#outfig = 'pe_example_injection.pdf'
#fig.savefig(outfig)
pl.show()

fig.clf()
pl.close(fig)
