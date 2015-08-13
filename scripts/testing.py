import numpy as np
import bayesflare as bf
import matplotlib.pyplot as pl
from copy import copy, deepcopy

def get_odds_ratio(curve):
    Or = bf.OddsRatioDetector( curve,
                               bglen=None,
                               bgorder=4,
                               nsinusoids=0,
                               noiseestmethod='powerspectrum',
                               psestfrac=0.5,
                               tvsigma=None,
                               flareparams={'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)},
                               noisepoly=True,
                               noiseimpulse=True,
                               noiseimpulseparams={'t0': (np.inf,)},
                               noiseexpdecay=True,
                               noiseexpdecayparams={'tauexp': (0.0, 0.25*60*60, 3)},
                               noiseexpdecaywithreverse=True,
                               ignoreedges=True )
    lnO, ts = Or.oddsratio()
    lnO = np.array(lnO)
    ts = np.array(ts)
    return lnO, ts, Or

def get_flares(curve_file):
	my_curve = bf.Lightcurve(curve_file)
	my_curve.detrend(method='runningmedian', nbins=55)
	lnO, ts, Or = get_odds_ratio(my_curve)
	full_length_ts = copy(ts)
	lnO, ts = zero_excluder(my_curve.clc, lnO, ts)
	lnO, ts = Or.impulse_excluder(lnO, ts)

	flarelist, numflares, maxlist = Or.thresholder(lnO, 5, 1)
	# print curve_file
	# print str(numflares) + " " + str(flarelist)
	ts0 = ts[0]
	ts = (ts-ts0)/86400.
	full_length_ts0 =full_length_ts[0]
	full_length_ts = (full_length_ts - full_length_ts0)/86400.
	
	fig, axarr = pl.subplots(2)


	axarr[0].plot(full_length_ts, my_curve.clc)
	axarr[1].plot(ts, lnO)	
	ylims = axarr[0].get_ylim()

	for fl in flarelist:
		fs = ts[fl[0]]
		fe = ts[fl[1]]
		axarr[0].fill_between([fs, fe], [ylims[0], ylims[0]], [ylims[1], ylims[1]], alpha=0.35, facecolor='k', edgecolor='none')

	axarr[0].set_ylim((ylims[0], ylims[1]))
	pl.show()

def get_flares_old(curve_file):
	my_curve = bf.Lightcurve(curve_file)
	my_curve.detrend(method='runningmedian', nbins=55)
	lnO, ts, Or = get_odds_ratio(my_curve)
	
	flarelist, numflares, maxlist = Or.thresholder(lnO, 5, 1)
	# print curve_file
	# print str(numflares) + " " + str(flarelist)
	ts0 = ts[0]
	ts = (ts-ts0)/86400.
	fig, axarr = pl.subplots(2)

	axarr[0].plot(ts, my_curve.clc)
	axarr[1].plot(ts, lnO)
	ylims = axarr[0].get_ylim()
	for fl in flarelist:
		fs = ts[fl[0]]
		fe = ts[fl[1]]
		axarr[0].fill_between([fs, fe], [ylims[0], ylims[0]], [ylims[1], ylims[1]], alpha=0.35, facecolor='k', edgecolor='none')
		axarr[0].set_ylim((ylims[0], ylims[1]))
	
	pl.show()


def zero_excluder(curve, lnO, ts):
	indexs = curve.nonzero()
	indexs = indexs[0]
	zero_gap = []
	for i in indexs:
		if i > 5 and i < len(indexs) - 6:
			if indexs[i] - indexs[i-1] > 2:
				zero_gap.append([indexs[i-6], indexs[i+5]])
	
	print zero_gap 
	# trying to do something with zero_gap and lnO and ts here

	lnO = lnO[indexs]
	ts = ts[indexs]
	return lnO, ts

curve_file = '/home/holly/data/007598326/kplr007598326-2010355172524_llc.fits'
get_flares(curve_file)


# a = [1,2,3,4,5,6,7,8,9]
# a[1:2] = False
# print a