import bayesflare as bf
import matplotlib.pyplot as pl
import numpy as np

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
	flarelist, numflares, maxlist = Or.thresholder(lnO, 10, 1)

	if numflares > 0:
		print curve_file
		print str(numflares) + " " + str(flarelist)
		pl.plot(ts, lnO)
		pl.show()

curve_files = {'/home/holly/Kepler/Q1_public/kplr001429589-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr003329643-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr003852865-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr004851356-2009166043257_llc.fits',
				# '/home/holly/Kepler/Q1_public/kplr004949027-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr005781991-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr005965629-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr006128245-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr007115200-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr007191523-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr007598326-2009166043257_llc.fits',
				# '/home/holly/Kepler/Q1_public/kplr008043882-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr008167504-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr008939211-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr008949512-2009166043257_llc.fits',
				'/home/holly/Kepler/Q1_public/kplr009025739-2009166043257_llc.fits'
			}
for curve in curve_files:
	get_flares(curve)



