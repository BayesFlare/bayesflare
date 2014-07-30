"""This script is intended to act as a way of compiling a flare list
from GOES data, and then insert it into a database. This allows new
entries to be easily appended as new data becomes available.

"""

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

start = '2014-03-01 00:00'
end = '2014-07-01 00:00'

# Establish a connection to the database
list = bf.Flare_List('/home/danielw/flare.db')
list.setup_flare_table()

# A wee utility function I need to put somewhere better
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

# Now to get on with the heavy lifting; a massive loop to progress the analysis two days at a time.

start = parser.parse(start)
end = parser.parse(end)
step = datetime.timedelta(days=2)

length = (end - start).days

for i in range(length):
    sec_start = start + step * i
    sec_end = start + step * (i+1)
    data = pyscidata.solar.goes.GOESLightcurve(sec_start, sec_end, \
                                           title='GOES', default='xrsb', cadence='1min')

    bglen = 55
    noiseestmethod='powerspectrum'
    psestfrac=0.5
    flareparams={'taugauss': (0,  20*60, 5), 
                 'tauexp'  : (0, 5*60*60, 60)}
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
        pe = bf.ParameterEstimationGrid('flare', flarelc)
        app_cent = np.where(pe.lightcurve.clc==pe.lightcurve.clc.max())[0]
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

        pe.set_grid(ranges={'taugauss': (2*60, max_len*0.25, 10), 
                            'tauexp'  : (1*60, max_len*0.5, 20),
                            'amp': amprange, 
                            't0': (pe.lightcurve.cts[starttime],pe.lightcurve.cts[endtime], 11)})
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