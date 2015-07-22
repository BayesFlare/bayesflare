import bayesflare as bf
import matplotlib.pyplot as pl
import numpy as np
import matplotlib.mlab as ml
curve_file = '/home/holly/data/001429589/kplr001429589-2009166043257_llc.fits'
my_curve = bf.Lightcurve(curve_file, detrend=False)

fig, ax = pl.subplots(2)
pl.title('Lightcurve for KIC'+ str(my_curve.id))
my_curve.trace = ax[0].plot(my_curve.cts/(24*3600.0), my_curve.clc)
fig.autofmt_xdate()
pl.xlabel('Time [days]')
pl.ylabel('Luminosity')

my_curve = bf.Lightcurve(curve_file, detrend=True, 
detrendmethod ='highpassfilter')
my_curve.trace = ax[1].plot(my_curve.cts/(24*3600.0), my_curve.clc)


pl.show()


