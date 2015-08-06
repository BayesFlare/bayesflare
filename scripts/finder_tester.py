import bayesflare as bf
import matplotlib.pyplot as pl
import numpy as np
import os

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

def get_flares(curve_file):
	my_curve = bf.Lightcurve(curve_file)
	my_curve.detrend(method='runningmedian', nbins=55)
	lnO, ts, Or = get_odds_ratio(my_curve)
	Or.impulse_excluder(lnO, ts)
	flarelist, numflares, maxlist = Or.thresholder(lnO, 5, 1)
	print curve_file
	print str(numflares) + " " + str(flarelist)
	pl.plot(my_curve.cts, my_curve.clc)
	pl.plot(ts, lnO)				
	pl.show()

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








