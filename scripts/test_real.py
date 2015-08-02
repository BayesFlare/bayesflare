import bayesflare as bf
import matplotlib.pyplot as pl
import numpy as np
import matplotlib.mlab as ml
from copy import deepcopy

def supersmoother (curve, alpha):
	curve.detrend(method='supersmoother', alpha=alpha)
	return curve

def savitzkygolay (curve, nbins, order):
	curve.detrend(method='savitzkygolay', nbins=nbins, order=order)
	return curve

def runningmedian (curve, nbins):
	curve.detrend(method='runningmedian', nbins=nbins)
	return curve

def highpassfilter (curve, knee):
	curve.detrend(method='highpassfilter', knee=knee)
	return curve

def get_odds_ratio(curve, bglen):
    if bglen != None:
      noiseimpulseparams={'t0': (0, (bglen-1.)*curve.dt(), bglen)}
    else:
      noiseimpulseparams={'t0': (np.inf,)}  
    Or = bf.OddsRatioDetector( curve,
                               bglen=bglen,
                               bgorder=4,
                               nsinusoids=nsinusoids,
                               noiseestmethod='powerspectrum',
                               psestfrac=0.5,
                               tvsigma=None,
                               flareparams={'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)},
                               noisepoly=True,
                               noiseimpulse=True,
                               noiseimpulseparams=noiseimpulseparams,
                               noiseexpdecay=True,
                               noiseexpdecayparams={'tauexp': (0.0, 0.25*60*60, 3)},
                               noiseexpdecaywithreverse=True,
                               ignoreedges=True )

    lnO, ts = Or.oddsratio()
    return lnO, ts

curve_file = '/home/holly/data/001873543/kplr001873543-2009166043257_llc.fits'
my_curve = bf.Lightcurve(curve_file, detrend=False)


bglen = None
nsinusoids = 0


filters = [
  lambda curve: curve,
  lambda curve: savitzkygolay(curve, 55, 4),
  lambda curve: runningmedian(curve, 55),
  lambda curve: supersmoother(curve, 0),
  lambda curve: highpassfilter(curve, 0.00003858)
]
fig, ax = pl.subplots(5)
i = 0
for f in filters:
  tmpcurve = f(deepcopy(my_curve))
  tmpcurve2 = deepcopy(tmpcurve)
  lnO, ts = get_odds_ratio(tmpcurve,None)
  ax[i].plot(ts, lnO)
  lnO, ts = get_odds_ratio(tmpcurve2,55)
  ax[i].plot(ts, lnO)
  i += 1
pl.show()


