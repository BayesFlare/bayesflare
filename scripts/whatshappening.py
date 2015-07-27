import bayesflare as bf
import matplotlib.pyplot as pl
import numpy as np
import matplotlib.mlab as ml
import supersmoother as ss
curve_file = '/home/holly/data/001429589/kplr001429589-2009166043257_llc.fits'

my_curve = bf.Lightcurve(curve_file, detrend=False)

t_fit = np.linspace(my_curve.cts[0], my_curve.cts[-1], len(my_curve.cts))

period = 1000000

model = ss.SuperSmoother()
y_t = model.fit(my_curve.cts, my_curve.clc).predict(t_fit)
pl.plot(t_fit, y_t)
pl.show()


