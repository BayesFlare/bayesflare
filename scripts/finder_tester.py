import bayesflare as bf
import matplotlib.pyplot as pl
import numpy as np
import os
import re
from copy import copy, deepcopy
from bisect import bisect_left

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
    return lnO, ts, Or

# def get_flares(curve_file):
# 	my_curve = bf.Lightcurve(curve_file)
# 	my_curve.detrend(method='runningmedian', nbins=55)
# 	lnO, ts, Or = get_odds_ratio(my_curve)
# 	full_length_ts = copy(ts)
# 	lnO, ts = Or.zero_excluder(my_curve.clc, lnO, ts)
# 	lnO, ts = Or.impulse_excluder(lnO, ts)
# 	flarelist, numflares, maxlist = Or.thresholder(lnO, 5, 1)
# 	# print curve_file
# 	# print str(numflares) + " " + str(flarelist)
# 	ts0 = ts[0]
# 	ts = (ts-ts0)/86400.
# 	full_length_ts0 =full_length_ts[0]
# 	full_length_ts = (full_length_ts - full_length_ts0)/86400.
	
# 	fig, axarr = pl.subplots(2)

# 	results = re.search(r"kplr(.*)_llc\.fits", curve_file)
# 	print results.group(1)
# 	axarr[0].set_title(results.group(1))

# 	axarr[0].plot(full_length_ts, my_curve.clc)
# 	axarr[1].plot(ts, lnO)	
# 	ylims = axarr[0].get_ylim()

# 	for fl in flarelist: # one of the stars didn't have an fl[1]? maybe not fl[0] which star? put in a skip incase this happens again 
# 		fs = ts[fl[0]]
# 		fe = ts[fl[1]]
# 		axarr[0].fill_between([fs, fe], [ylims[0], ylims[0]], [ylims[1], ylims[1]], alpha=0.35, facecolor='k', edgecolor='none')

# 	axarr[0].set_ylim((ylims[0], ylims[1]))
# 	pl.show()
# 	pl.savefig(results.group(1))
# 	pl.close()		
	
def get_flares(curve_file):
	my_curve = bf.Lightcurve(curve_file)
	my_curve.detrend(method='runningmedian', nbins=55)

	lnO, ts, Or = get_odds_ratio(my_curve)
	full_length_ts = copy(ts)
	lnO, ts = zero_excluder(my_curve.clc, lnO, ts)
	lnO, ts = Or.impulse_excluder(lnO, ts)

	flarelist, numflares, maxlist = Or.thresholder(lnO, 5, 1)
	ts0 = ts[0]
	ts = (ts-ts0)/86400.
	full_length_ts0 =full_length_ts[0]
	full_length_ts = (full_length_ts - full_length_ts0)/86400.


	results = re.search(r"kplr(.*)_llc\.fits", curve_file)
	no_subplot = len(flarelist)+2
	print results.group(1)
	ax1 = pl.subplot2grid((2,no_subplot), (0,0), colspan=2)
	ax2 = pl.subplot2grid((2,no_subplot), (1,0), colspan=2)
	ax1.set_title(results.group(1))
	ax1.plot(full_length_ts, my_curve.clc)
	ax2.plot(ts, lnO)	
	ylims = ax1.get_ylim()

	column_no = 2

	for fl in flarelist:
		fs = ts[fl[0]]
		fe = ts[fl[1]]
		ind_s = bisect_left(full_length_ts, fs)
		ind_e = bisect_left(full_length_ts, fe)
		ax1.fill_between([fs, fe], [ylims[0], ylims[0]], [ylims[1], ylims[1]], alpha=0.35, facecolor='k', edgecolor='none')

		flare = pl.subplot2grid((2,no_subplot), (0,column_no), rowspan=2)
		flare.plot(full_length_ts[ind_s:ind_e],my_curve.clc[ind_s:ind_e])	

		column_no += 1
	ax1.set_ylim((ylims[0], ylims[1]))
	pl.show()
	pl.savefig(results.group(1))
	pl.close()

def zero_excluder(curve, lnO, ts):
    indexs = curve.nonzero()[0]

    non_zero = np.ones(len(lnO), dtype=np.bool)

    w = 6 # window border
    for i in xrange(w, len(indexs) - w*2):
        if indexs[i] - indexs[i-1] > 2:
            non_zero[indexs[i-w] : indexs[i+w*2]] = False

    # return arrays with parts excluded
    return np.copy(lnO)[non_zero], np.copy(ts)[non_zero]

def find_files(rootDir):
	files = []
	for dirName, subdirList, fileList in os.walk(rootDir):
		# print ('Found directory: %s' % dirName)
		for fname in fileList:
			lc = '{0}/{1}'.format(dirName, fname)
			files.append(lc)
	return files

all_files = find_files('/home/holly/data')

for lc in all_files:
	get_flares(lc)








