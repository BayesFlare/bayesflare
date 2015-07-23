import bayesflare as bf
import matplotlib.pyplot as pl
import numpy as np
import matplotlib.mlab as ml
import supersmoother as ss
curve_file = '/home/holly/data/001429589/kplr001429589-2009166043257_llc.fits'

my_curve = bf.Lightcurve(curve_file, detrend=False)

"""def make_test_set(N=200, rseed_x=None, rseed_y=None):

    rng_x = np.random.RandomState(rseed_x)
    rng_y = np.random.RandomState(rseed_y)
    x = rng_x.rand(N)
    dy = x
    y = np.sin(2 * np.pi * (1 - x) ** 2) + dy * rng_y.randn(N)
    return x, y, dy
t, y, dy = make_test_set(rseed_x=0, rseed_y=1)
# fit the supersmoother model
model = ss.SuperSmoother()
model.fit(t, y)

# find the smoothed fit to the data
tfit = np.linspace(0, 1, 1000)
yfit = model.predict(tfit)"""

smooth_test = ss.SuperSmoother()
smooth_test.fit(my_curve.cts, my_curve.clc)
#tfit = np.linspace(my_curve.cts[0], my_curve.cts[len(my_curve.cts)-1], len(my_curve.cts))
yfit = smooth_test.predict(my_curve.cts)
 

fig, ax = pl.subplots(1)
pl.title('Lightcurve for KIC'+ str(my_curve.id))
my_curve.trace = ax.plot(my_curve.cts/(24*3600), yfit)
#fig.autofmt_xdate()
#pl.xlabel('Time [days]')
#pl.ylabel('Luminosity')

#my_curve = bf.Lightcurve(curve_file, detrend=True, detrendmethod = 'runningmedian', nbins = 10000)
#my_curve.trace = ax[1].plot(my_curve.cts/(24*3600.0), my_curve.clc)"""


pl.show()


