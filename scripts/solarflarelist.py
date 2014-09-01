"""This script is intended to act as a way of compiling a flare list
from GOES data, and then insert it into a database. This allows new
entries to be easily appended as new data becomes available.

"""

# Set up the script using command line options

from optparse import OptionParser
import datetime

__version__ = "1.0.0"

parser = OptionParser()# usage = usage, description = description, version = __version__ )

parser.add_option("-d", "--database", dest="database",
                  help="The SQLite database which will store the results.")

parser.add_option("-T", "--threshold", dest="threshold",
                    help="Odds ratios above this value will count as detections [default: %default].",
                    type="float", default=7.027591)

parser.add_option("-s", "--start", dest="start",
                  help="The start \
                  datetime string for the analysis. Format should be\
                  YYYY-MM-DD HH:MM:SS; time may be safely\
                  omitted. [default: %default].",
                  type="float",
                  default=None)

parser.add_option("-e", "--end", dest="end",
                  help="The end \
                  datetime string for the analysis. Format should be\
                  YYYY-MM-DD HH:MM:SS; time may be safely\
                  omitted. [default: %default].",
                  type="float",
                  default=str(datetime.datetime.now()))

(opts, args) = parser.parse_args()

# Initial imports
import matplotlib
import matplotlib.pyplot as pl
import numpy as np
import bayesflare as bf
import pylightcurve as lc
import pyscidata
import pyscidata.solar
import pyscidata.solar.goes
from dateutil import parser
from sunpy.net import hek
import pandas as pd
from math import ceil

import dateutil.relativedelta as rd
import datetime

# These are the time ranges to work over, for now I'm just going to
# make this very unintelligent, and then replace it with a
# command-line options method later.

# Establish a connection to the database
list = bf.Flare_List('/home/danielw/anotherflare.db')

if opts.start == None:
  start = list.latest()
else:
  start = opts.start
  start = parser.parse(start)
if opt.end == None:
  end = str(datetime.datetime.now())
  end = parser.parse(end)
else:
  end = opts.end
list.setup_flare_table()

# A wee utility function I need to put somewhere better
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

# Now to get on with the heavy lifting; a massive loop to progress the analysis two days at a time.



step = datetime.timedelta(days=2)

length = (end - start).days

for i in range(length):
    sec_start = start + step * i
    sec_end = start + step * (i+1)
    print sec_start, sec_end
    data = pyscidata.solar.goes.GOESLightcurve(sec_start, sec_end, \
                                           title='GOES', default='xrsb', cadence='1min')

    bglen = 55
    noiseestmethod='powerspectrum'
    psestfrac=0.5
    flareparams={'taugauss': (0,  500, 5), 
                 'tauexp'  : (0, 3000, 10)}
    odds = bf.OddsRatioDetector( 
        data.interpolate(),
        bglen=bglen,
        bgorder=0,
        noiseestmethod='powerspectrum',
        tvsigma=1,
        psestfrac=0.5,
        flareparams=flareparams,
        noisepoly=True,
        noiseimpulse=True,
        noiseimpulseparams={'t0': (0, (bglen-1.)*data.dt(), bglen)},
        noiseexpdecay=False,
        noiseexpdecayparams={'tauexp': (600, 3*60*60, 6)},
        noiseexpdecaywithreverse=True,
        ignoreedges=False,
        edgelength=10
    )
    odds_out = odds.oddsratio()
    odds_array = odds_out[0]
    flarelist, Nflares, maxlist = odds.thresholder(odds_array, 7.027591, expand=3)

    for i in range(Nflares):
        flarelc = data[flarelist[i][0]-10:flarelist[i][1]+10]
	if (np.count_nonzero(~np.isnan(flarelc.data.xrsa)) == 0) \
	or (np.count_nonzero(~np.isnan(flarelc.data.xrsb)) == 0):
		continue

	
        pe = bf.ParameterEstimationGrid('flare', flarelc)
        app_cent = np.where(pe.lightcurve.clc==np.nanmax(pe.lightcurve.clc))[0][0]
        max_len = ceil((3*flarelc.cts[-1]-flarelc.cts[0])/3600)*3600
        if max_len > 10800:
            max_len = 10800
        # Calculate the amplitude range
        amprange = (0.5*pe.lightcurve.data.min()['xrsb'], 
                    1.2*pe.lightcurve.data.max()['xrsb'], 
                    20)
        #
        if app_cent+5 < len(pe.lightcurve.clc):
            endtime = app_cent+5
        else:
            endtime = len(pe.lightcurve.clc)-1
        if app_cent-5 < 0:
            starttime = 0
        else:
            starttime = app_cent-5

        print endtime - starttime, endtime, starttime


        try:
		pe.set_grid(ranges={'taugauss': (0, 500 , 10), 
                            'tauexp'  : (0, 3000, 60),
                            'amp': amprange, 
                            't0': (pe.lightcurve.cts[starttime],pe.lightcurve.cts[endtime], (endtime - starttime)+1)})
	except:
		continue

        pe.calculate_posterior(bgorder=0)
       	pe.marginalise_all()
        pe.maximum_posterior_snr()
       	convfac = {'amp': 1., 'taugauss': 1./3600, 
               	   'tauexp': 1./3600, 't0': 1} # convert output times to hours for plots
        valname = {'amp': 'Amplitude', 'taugauss': '$\\tau_g$ (hours)', 
       	           'tauexp': '$\\tau_e$ (hours)', 't0': '$t_0$ (hours)'}
        maxi = pe.maxpostparams
        best = find_nearest(pe.lightcurve.cts,maxi['t0'])[0]
        list.save_flare(
       	    data.data.index[flarelist[i][0]], \
            data.data.index[maxlist[i][1]], \
            data.data.xrsb[maxlist[i][1]], \
            data.data.index[flarelist[i][1]], \
            maxi['taugauss']/3600,\
            maxi['tauexp']/3600,\
            pe.lightcurve.data.index[best],\
       	    maxi['amp']+data._dcoffsets['xrsb'])
