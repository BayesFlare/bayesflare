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

####################################################################################
# Don't change these	
tlength = 2893536. #length of time data
tstep = 1765.55929 #time data step
nstd = 1. #noise standard deviation
bglen = 55 #background window length (must be odd) so there is always a bin in the centre
bgorder = 4 #The polynomial order of the fitted background variability
taug = 2700 #Gaussian rise width of injected flare (sec) * (45 min)
taue = 6300 #Exponential decay timescale of injected flare (sec) 1 hr 45
noiseest = 'powerspectrum'
kneevalue = 0.00003858
psest = 0.5 #fraction of the spectrum with which to estimate the noise
freq = (1./(4.*86400.))
amp = 4
period=350000
injamp = 100 # injected flare amplitude
alpha = None #can be None or from 0 to 10 smoothest is 10 (for super smoother)
#####################################################################################

test_which = "runningmedian"  #supersmoother finds best alpha value(only runs with one curve)
					#runningmedian runs runningmedian on 100 diff curves an gets ave red chi dq
					#all tests 100 curves and prints out that filter chich gave lowest red chi sq

def chi_sq(no_noise_data, smoothed, sigma):
	res = no_noise_data - smoothed
	return np.sqrt(np.sum(res**2)/(len(no_noise_data)-1.))/sigma

def make_curve():
	ts = np.arange(0., tlength, tstep, dtype='float64')
	flarelc = bf.Lightcurve()
	flarelc.clc = nstd*np.random.randn(len(ts)) #clc y data
	flarelc.cts = np.copy(ts) #cts time stamp data
	flarelc.cadence = 'long'
	tmi = flarelc.cts-flarelc.cts[0] #makes time data start at 0
	Mfi = bf.Flare(tmi, amp=1.) #creates a Flare object
	t0 = tmi[int(len(tmi)/2)] #central time of flare set to middle of time data
	pdict = {'t0': t0, 'amp': injamp, 'taugauss': taug, 'tauexp': taue}
	injdata = np.copy(Mfi.model(pdict)) #creates a flare in the flare object Mfi, using data pdict, copies this into injdata
	flarelc.clc = flarelc.clc + injdata #adds flare model to data	
	# phase = 2.*np.pi*np.random.rand(1)
	# sinewave = amp*np.sin(2.*np.pi*freq*ts + phase)
	# flarelc.clc = flarelc.clc + sinewave
	curve = copy(flarelc)	
	return curve, injdata

def find_best_alpha():
	alphas = []
	min_alpha = []
	tmpcurve, injdata = make_curve()
	tmpcurve.detrend(method='supersmoother', alpha=None)
	Chi4 = chi_sq(injdata, tmpcurve.clc, nstd)
	alphas.append(["None", Chi4])
	for i in range (0,11):
		supersmoother(tmpcurve, i)
		Chi4 = chi_sq(injdata, tmpcurve.clc, nstd)
		alphas.append([i, Chi4])
	min_alpha.append(min(alphas, key=lambda x: x[1]))
	return min_alpha

def find_best_filter():
	min_chi = []
	for i in range (0,100):
		curve, injdata = make_curve()

		tmpcurve = savitzkygolay(copy(curve))
		Chi1 = chi_sq(injdata, tmpcurve.clc, nstd)

		tmpcurve = highpassfilter(copy(curve))
		Chi2 = chi_sq(injdata, tmpcurve.clc, nstd)
		
		tmpcurve = runningmedian(copy(curve))
		Chi3 = chi_sq(injdata, tmpcurve.clc, nstd)
		
		tmpcurve = supersmoother(copy(curve))
		Chi4 = chi_sq(injdata, tmpcurve.clc, nstd)

		# tmpcurve = copy(curve)
		# tmpcurve.detrend(method='periodsmoother', alpha=alpha,phase=phase,period=period)
		# Chi5 = chi_sq(injdata, tmpcurve.clc, nstd)

		chi_val = {'savitzkygolay': Chi1, 
					'highpassfilter': Chi2,
					'runningmedian': Chi3,
					'supersmoother': Chi4} #'periodsmoother': Chi5

		min_chi.append(min(chi_val.items(), key=lambda x: x[1]))
	return min_chi

def supersmoother (curve, alpha_=alpha):
	curve.detrend(method='supersmoother', alpha=alpha_)
	return curve

def savitzkygolay (curve, nbins=bglen, order=bgorder):
	curve.detrend(method='savitzkygolay', nbins=nbins, order=order)
	return curve

def runningmedian (curve, nbins=bglen):
	curve.detrend(method='runningmedian', nbins=nbins)
	return curve

def highpassfilter (curve, knee=kneevalue):
	curve.detrend(method='highpassfilter', knee=knee)
	return curve

def avg_chi(no_tests, filter_used):
	tot = 0.
	for i in range (0,no_tests):
		tmpcurve, injdata = make_curve()
		tmpcurve = filter_used(tmpcurve)
		Chi = chi_sq(injdata, tmpcurve.clc, nstd)
		tot += Chi
	return tot/no_tests	


if test_which == "supersmoother":
	best_alpha = find_best_alpha()
	print best_alpha

elif test_which == "runningmedian":
	print avg_chi(100, runningmedian)

elif test_which == "all":
	best_filter = find_best_filter()
	print best_filter




	
