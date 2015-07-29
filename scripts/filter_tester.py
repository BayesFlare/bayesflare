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
amp = 4
period=350000
injamp = 100 # injected flare amplitude
alpha = None #can be None or from 0 to 10 smoothest is 10 (for super smoother)None #can be None or from 0 to 10 smoothest is 10 (for super smoother)
nstd = 1.
#####################################################################################

def chi_sq(no_noise_data, smoothed, sigma):
	res = no_noise_data - smoothed
	return np.sqrt(np.sum(res**2)/(len(no_noise_data)-1.))/sigma

def make_curve(add_sine=False,amp=None,period=None):
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

	if add_sine == True:
		freq=1./period
		phase = 2.*np.pi*np.random.rand(1)
		sinewave = amp*np.sin(2.*np.pi*freq*ts + phase)
		flarelc.clc = flarelc.clc + sinewave	
	curve = copy(flarelc)
	return curve, injdata

def find_best_alpha():
	alphas = []
	min_alpha = []
	tmpcurve, injdata = make_curve()
	supersmoother(tmpcurve, None)
	alphas.append(["None", chi_sq(injdata, tmpcurve.clc, nstd)])
	for i in range (0,11):
		supersmoother(tmpcurve, i) 
		alphas.append([i, chi_sq(injdata, tmpcurve.clc, nstd)])
	min_alpha.append(min(alphas, key=lambda x: x[1]))
	return min_alpha

def find_best_filter(curve_maker):
	count = {
		'savitzkygolay': 0, 
		'highpassfilter': 0,
		'runningmedian': 0,
		'supersmoother': 0
	}
	for i in range (0,100):
		curve, injdata = curve_maker()
		min_chi = best_filter_for_curve(curve, injdata)
		count[min_chi[0]] += 1

	return count

def find_best_periodic_filter():
	min_period = 0.2*24*60*60
	max_period = 10*24*60*60
	step = (max_period - min_period) / 100.
	i = min_period
	best = []
	while i <= max_period:			
		curve, injdata = make_curve(True,10,i)
		min_chi = best_filter_for_curve(curve, injdata)

		best.append([i, min_chi[0], min_chi[1]])
		i += step
	return best

def best_filter_for_curve(curve, injdata):
	tmpcurve = savitzkygolay(copy(curve))
	Chi1 = chi_sq(injdata, tmpcurve.clc, nstd)

	tmpcurve = highpassfilter(copy(curve))
	Chi2 = chi_sq(injdata, tmpcurve.clc, nstd)
	
	tmpcurve = runningmedian(copy(curve))
	Chi3 = chi_sq(injdata, tmpcurve.clc, nstd)
	
	tmpcurve = supersmoother(copy(curve))
	Chi4 = chi_sq(injdata, tmpcurve.clc, nstd)

	chi_val = {
		'savitzkygolay': Chi1, 
		'highpassfilter': Chi2,
		'runningmedian': Chi3,
		'supersmoother': Chi4
	}
	return min(chi_val.items(), key=lambda x: x[1])
	
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

def avg_chi(no_tests, filter_used, curve_maker):
	tot = 0.
	for i in range (0,no_tests):
		tmpcurve, injdata = curve_maker()
		tmpcurve = filter_used(tmpcurve)
		Chi = chi_sq(injdata, tmpcurve.clc, nstd)
		tot += Chi
	return tot/no_tests

def format_periodic_list(best_filters):
	# sort them so that the same filters are adjacent 
	# and within these filters elements are sorted by period
	best_filters.sort(lambda x, y: cmp(x[0], y[0]) if x[1] == y[1] else cmp(x[1], y[1]))

	# extract a range of periods for which the filter is best
	def extract_ranges(ranges, element):
		"""Returns a dictionary in the form {'filter': [min_period, max_period]}"""
		filter_name, period = element[1], element[0]
		if filter_name not in ranges:
			ranges[filter_name] = [period, period]
		ranges[filter_name][1] = period
		return ranges
	
	ranges = reduce(extract_ranges, best_filters, {})

	# convert a dictionary to a list for subsequent sorting
	ranges_list = [[x, ranges[x][0], ranges[x][1]] for x in ranges]
	# sort the ranges so that smaller periods go first
	ranges_list.sort(key=lambda x: x[1])

	# print out the results, one on each line
	format_range = lambda x: "[" + str(int(x[1])) + ", " + str(int(x[2])) + "]"
	return "\n".join([x[0] + ": " + format_range(x) for x in ranges_list])


###############################################################

filters = {
	'runningmedian': runningmedian,
	'highpassfilter': highpassfilter,
	'supersmoother': supersmoother,
	'savitzkygolay': savitzkygolay
}


if __name__=='__main__':
	test_which = "all" if len(sys.argv) == 1 else sys.argv[1]

	if test_which == "alpha":
		best_alpha = find_best_alpha()
		print best_alpha

	elif test_which == "all":
		best_filter = find_best_filter(lambda: make_curve(True, 50, 24*60*60))
		print best_filter

	elif test_which == "periodic":
		# find best filters for each period
		print format_periodic_list(find_best_periodic_filter())

	else:
		print avg_chi(100, filters[test_which], lambda: make_curve(True, 50, 24*60*60))



	
		


	# test, injdata = make_curve(True,50,10*24*60*60)
	# pl.plot(test.cts,test.clc)
	# pl.show()



	
