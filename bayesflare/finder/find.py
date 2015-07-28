"""
"""
from ..misc import mkdir
from ..noise import estimate_noise_ps, estimate_noise_tv
from ..models import *
from ..stats import *
from math import log
import pyfits
import numpy as np
from copy import copy, deepcopy

def contiguous_regions(condition):
        """
        Find contiguous regions for the condition e.g. array > threshold and return
        a list as two columns of the start and stop indices for each region
        (see http://stackoverflow.com/a/4495197/1862861)

        Parameters
        ----------
        condition : string
           A test condition (e.g. 'array > threshold') returning a :class:`numpy.array`.

        Returns
        -------
        idx : array-like
           A two column array containing the start and end indices of
           contiguous regions obeying the condition.

        """

        # Find the indicies of changes in "condition"
        d = np.diff(condition)
        idx, = d.nonzero()

        # We need to start things after the change in "condition". Therefore,
        # we'll shift the index by 1 to the right.
        idx += 1

        if condition[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]

        if condition[-1]:
            # If the end of condition is True, append the length of the array
            idx = np.r_[idx, condition.size] # Edit

        # Reshape the result into two columns
        idx.shape = (-1,2)
        return idx


#class Finder():
#    """
#    Operations intended to identify flaring events from the array of Bayesian odds-ratios,
#    and to save the results and the odds-ratios out as a FITS file.
#
#    Parameters
#    ----------
#    bayesfactors : BayesFlare Bayes instance
#       The object holding the odds-ratio information.
#    thresholder : BayesFlare Thresholder instance
#       The object containing the thresholds.
#    results_root : str, optional
#       A file path to the location where results should be output. Defaults to the value of
#       the environmental variable KPLR_RES_ROOT.
#
#    """
#
#
#    results = np.array([])
#
#    def __init__(self, bayesfactors, thresholder, results_root=None):
#        self.results_root = results_root
#        if results_root is None:
#            self.results_root = KPLR_RES_ROOT
#        self.bayesfactors = bayesfactors
#        self.thresholder = thresholder
#        self.find()
#
#    def mptaus(self, i):
#        """
#        Find the most probable set of decay constants at a given location in an array of lnB values
#        marginalised over amplitude.
#
#        Parameters
#        ----------
#        i : int
#           The location index in the array of lnB values to be examined
#        lnBmargAmp : np.ndarray
#           An array of ln(bayes factor) values which have been marginalised over amplitude
#           but not over the decay constants.
#        tausGauss : np.ndarray
#           An array of the gaussian decay constants
#        tausExp : np.ndarray
#           An array of the exponential decay constants
#
#
#        Returns
#        -------
#        mptaugauss : float
#           The most probable value for tau gauss.
#        mptauexp : float
#           The most probable value of tau exp.
#
#        """
#        G = self.bayesfactors
#        mod = 'flare'
#        col1, col2 = np.unravel_index(np.argmax(G.premarg[mod][:,:,i]), np.shape(G.premarg['flare'][:,:,i]))
#        tausGauss = G.model[0].ranges[0]
#        tausExp = G.model[0].ranges[1]
#        mptaugauss = tausGauss[col1]/3600.
#        mptauexp = tausExp[col2]/3600.
#        return mptaugauss, mptauexp
#
#    def find(self):
#        """
#        The principle flare finding method.
#        Must be run to generate the results table.
#
#        """
#
#        G = self.bayesfactors
#        T = self.thresholder
#
#        starti = endi = 0
#        flares = np.zeros((1,9))
#        in_flare = 0
#        for i in np.arange(len(G.lnBmargAmp)):
#            if G.lnBmargAmp[i] > T.threshold:
#                if in_flare == 0:
#                    in_flare = 1         # We're now in a flare
#                    starti   = i         # Note the starting position of the flare detection
#                # Otherwise we just go on about our business.
#            else:
#                if in_flare == 1:
#                    in_flare = 0     # We're now out of the flare
#                    endi     = i     # Note the end position of the flare
#
#                    mp = starti+np.argmax(G.lnBmargAmp[starti:endi])
#                    ma = starti+np.argmax(G.lightcurve.clc[starti:endi])
#
#                    mp_tau_gauss, mp_tau_exp = self.mptaus(mp)
#
#                    background = 0
#                    if (endi < len(G.lightcurve.clc)-2):
#                        background   = (G.lightcurve.clc[starti-1] + G.lightcurve.clc[endi+1])/2.0
#                    elif starti > 0:
#                        background   = G.lightcurve.clc[starti-1]
#
#                    amplitude = G.lightcurve.clc[ma] - background
#                    damplitude = np.abs((amplitude+G.lightcurve.dc)/(background+G.lightcurve.dc))
#
#                    EW               = np.sum( (G.lightcurve.clc[starti:endi] - background +G.lightcurve.dc) / background+G.lightcurve.dc )
#                    logEW            = np.log(EW)
#
#                    g_duration       = np.sqrt( -2 * log(0.1)) * mp_tau_gauss
#                    e_duration       = - mp_tau_exp * log(0.1)
#                    duration         = e_duration + g_duration
#
#                    confidence = T.bayes_to_confidence(G.lnBmargAmp[mp])
#
#                    flare    = [starti, endi, mp_tau_gauss, mp_tau_exp, amplitude, damplitude, duration, logEW, confidence]
#                    starti = endi = 0
#                    if len(self.results)>0:
#                        self.results = np.vstack((self.results, flare))
#                    else:
#                        self.results = [flare]
#                in_flare = 0
#
#
#    def star_summary(self):
#        """
#        Generates a summary of the flaring activity of the star, and writes the KIC number of the star to an appropriate file.
#
#        .. note: At the moment this method isn't really doing anything, and may be deprecated in the near future.
#
#        """
#
#        quarter = self.bayesfactors.lightcurve.quarter
#        cadence = self.bayesfactors.lightcurve.cadence
#        objs  = str(self.bayesfactors.lightcurve.id)
#        nl = "\n"
#        flares = self.results
#
#        mkdir(self.results_root)
#        #sum_file = open(self.results_root+'/quarter'+str(quarter)+'/'+str(cadence)+'/summary/'+object+'.kep', 'w')
#
#        #pos_file = open(self.results_root + '/flare_stars.kep', 'a')
#        neg_file = open(self.results_root +'/normal_stars.kep', 'a')
#        #mis_file = open(self.results_root +'/missing.kep', 'a')
#        #hsi_file = open(self.results_root +'/high_sigma.kep', 'a')
#        #tra_file = open(self.results_root +'/transit_stars.kep', 'a')
#
#        if len(flares) == 0:
#            neg_file.write(str(objs)+'\n')
#
#        neg_file.close()
#
#    def star_vitals(self):
#        """
#        Generates summary statistics for the flaring of the star, and outputs it to a file.
#
#        """
#
#        quarter = self.bayesfactors.lightcurve.quarter
#        cadence = self.bayesfactors.lightcurve.cadence
#        object  = str(self.bayesfactors.lightcurve.id)
#        nl = "\n"
#        flares = self.results
#
#        if len(flares)==0:
#            return
#
#        mkdir(self.results_root+'/quarter'+str(quarter)+'/'+str(cadence)+'/flares')
#        det_file = open(self.results_root+'/quarter'+str(quarter)+'/'+cadence+'/flares/'+object+'.kep', 'w')
#
#        det_file.write('# Start the flaring statistics' + nl)
#        det_file.write('# Flaring statistics for KIC' + object + nl)
#        det_file.write('# Total number of flares: ' + str(len(flares)) +nl)
#        det_file.write('# ------------------------------------------------------'+nl)
#        det_file.write('# Start \t End \t MP t_g \t MP t_e \t Amp \t DAmp \t Dur \t lEW \t Conf'+nl)
#        det_file.write(nl)
#        for i in np.arange(len(self.results)):
#           det_file.write('%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n' % (flares[i][0],
#                                                                                      flares[i][1],
#                                                                                      flares[i][2],
#                                                                                      flares[i][3],
#                                                                                      flares[i][4],
#                                                                                      flares[i][5],
#                                                                                      flares[i][6],
#                                                                                      flares[i][7],
#                                                                                      flares[i][8])
#        )
#        print np.shape(flares)
#        det_file.close()
#
#    def save(self):
#        """
#        This method writes the log Bayes factors out to a fits file,
#        along with the header information from the light curve file.
#        The original light curve is stored also.
#
#        Parameters
#        ----------
#        self : Finder object
#        A Finder object which contains the light curve and the time
#        series of log Bayes factors which are to be written out.
#
#        """
#
#        path = self.results_root
#        object  = str(self.bayesfactors.lightcurve.id)
#        G = self.bayesfactors
#
#        # Prepare the new meta data
#        hdr = pyfits.Header()
#        hdr['D_LENGTH'] = (G.lightcurve.detrend_length, '[samples] Detrending window len')
#        hdr['D_LENGTS'] = (G.lightcurve.detrend_length*G.lightcurve.dt(), '[sec] Detrending window len')
#        hdr['D_ORDER'] = (G.lightcurve.detrend_order, 'Detrending order')
#        hdr['THRESH_B'] = (self.thresholder.threshold, '[ln Bayes] Thresholding cutoff')
#        hdr['THRESH_P'] = (self.thresholder.confidence, '[pct] Thresholding cutoff')
#
#        # Store the various time series in a TableHDU
#        bayes = pyfits.Column(name='odds', format='E', array=G.lnBmargAmp)
#        times = pyfits.Column(name='time', format='E', array=G.lightcurve.cts)
#        lightcurve = pyfits.Column(name='lightcurve', format='E', array=G.lightcurve.clc)
#        data_table = pyfits.new_table([bayes, lightcurve, times], header=hdr)
#
#        # Store the Model parameters from the search into a TableHDU
#        model_nos = []
#        model_names = []
#        par_nos = []
#        par_mins = []
#        par_maxs = []
#        par_ints = []
#
#        for j in np.arange(len(G.model)):
#            model = G.model[j]
#            for i in np.arange(len(model.ranges)):
#                    model_nos   += [j]
#                    model_names += [model.identity_type()]
#                    par_nos     += [i]
#                    par_mins    += [model.ranges[i][0]]
#                    par_maxs    += [model.ranges[i][-1]]
#                    par_ints    += [len(model.ranges[i])]
#
#        cols = [
#            pyfits.Column(name='model_no', format='E', array=model_nos),
#            pyfits.Column(name='model_name', format='10A', array=model_names),
#            pyfits.Column(name='par_no', format='E', array=par_nos),
#            pyfits.Column(name='par_min', format='E', array=par_mins, unit='Seconds'),
#            pyfits.Column(name='par_max', format='E', array=par_maxs, unit='Seconds'),
#            pyfits.Column(name='par_qty', format='E', array=par_ints, unit='Seconds')
#            ]
#
#        model_table = pyfits.new_table(cols)
#
#        primary = pyfits.PrimaryHDU(header=G.lightcurve.header)
#
#        if len(self.results)==0:
#            hdulist = pyfits.HDUList([primary, data_table, model_table])
#        elif len(self.results)==1:
#            # Store the flare-finding results to a TableHDU
#            print "One flare detected"
#            cols = [
#                pyfits.Column(name='start_time', format='E', array=[self.results[0][0]], unit='Seconds'),
#                pyfits.Column(name='end_time', format='E', array=[self.results[0][1]], unit='Seconds'),
#                pyfits.Column(name='mp_t_exp', format='E', array=[self.results[0][2]], unit='Seconds'),
#                pyfits.Column(name='mp_t_gau', format='E', array=[self.results[0][3]], unit='Seconds'),
#                pyfits.Column(name='amplitude', format='E', array=[self.results[0][4]]),
#                pyfits.Column(name='diff_amplitude', format='E', array=[self.results[0][5]]),
#                pyfits.Column(name='duration', format='E', array=[self.results[0][6]]),
#                pyfits.Column(name='log_EW', format='E', array=[self.results[0][7]]),
#                pyfits.Column(name='confidence', format='E', array=[self.results[0][8]], unit='Percent')
#                ]
#            results_table = pyfits.new_table(cols)
#            hdulist = pyfits.HDUList([primary, data_table, model_table, results_table])
#        else:
#            # Store the flare-finding results to a TableHDU
#            cols = [
#                pyfits.Column(name='start_time', format='E', array=self.results[:,0], unit='Seconds'),
#                pyfits.Column(name='end_time', format='E', array=self.results[:,1], unit='Seconds'),
#                pyfits.Column(name='mp_t_exp', format='E', array=self.results[:,2], unit='Seconds'),
#                pyfits.Column(name='mp_t_gau', format='E', array=self.results[:,3], unit='Seconds'),
#                pyfits.Column(name='amplitude', format='E', array=self.results[:,4]),
#                pyfits.Column(name='diff_amplitude', format='E', array=self.results[:,5]),
#                pyfits.Column(name='duration', format='E', array=self.results[:,6]),
#                pyfits.Column(name='log_EW', format='E', array=self.results[:,7]),
#                pyfits.Column(name='confidence', format='E', array=self.results[:,8], unit='Percent')
#                ]
#            results_table = pyfits.new_table(cols)
#            hdulist = pyfits.HDUList([primary, data_table, model_table, results_table])
#
#        hdulist.writeto(self.results_root+'/'+ object + ".fits")


class SigmaThresholdMethod():
    """
    Search for points on a light curve that cross a threshold that is based on a number of
    standard deviations calculated from the data. This is based on the method used in [1]_.

    Parameters
    ----------
    lightcurve : :class`.Lightcurve`
       The light curve to be processed.
    detrendpoly : bool, optional, default: False
       Set to True to remove a second-order polynomial to fit the whole curve.
    detrendmedian : bool, optional, default: True
       Set to `True` to detrend the `lightcurve` data using a median filtering technique.
    noiseestmethod : {None, 'powerspectrum', 'tailveto'}, optional
       The method used to estimate the noise in the light curve. If `None` is chosen the noise will
       be estimated as the standard deviation of the entire light curve, including any signals.
       Defaults to `None`.
    psestfrac : float, optional, default: 0.5
       The fraction of the power spectrum to be used in estimating the noise, if
       `noiseestmethod=='powerspectrum'`.
    tvsigma : float, optional, default: 1.0
       The number of standard deviations giving the cumulative probability
       to be included in the noise calculation e.g. if sigma=1 then the central
       68% of the cumulative probability distribution is used.

    See Also
    --------

    estimate_noise_ps : The power spectrum noise estimator.
    estimate_noise_tv : The tail veto noise estimator.
    Lightcurve.running_median : The running median detrender.

    References
    ----------

    .. [1] Walkowicz *et al*, *AJ*, **141** (2011), `arXiv:1008.0853 <http://arxiv.org/abs/1008.0853>`_

    """

    sigma = 0
    nflares = 0
    flarelist = []

    def __init__(self, lightcurve, detrendpoly=False, detrendmedian=True, noiseestmethod=None, psestfrac=0.5, tvsigma=1.0):
        # detrend the lightcurve using a 2nd order polynomial fit to the whole curve
        # if required. Also detrend with a running median over 10 hr intervals.

        self.lightcurve = deepcopy(lightcurve)

        if detrendpoly:
            # remove 2nd order polynomial fit
            ts = self.lightcurve.cts - self.lightcurve.cts[0]
            z = np.polyfit(ts, self.lightcurve.clc, 2)
            f = np.poly1d(z)

            self.lightcurve.clc = self.lightcurve.clc - f(ts)

        if detrendmedian:
            nbins = int(10.*60.*60./self.lightcurve.dt())
            self.lightcurve.detrend(method='runningmedian', nbins=nbins)

        # get the standard deviation using the outlier removal method or not
        if noiseestmethod == None:
            self.sigma = np.std(self.lightcurve.clc)
        elif noiseestmethod == 'powerspectrum':
            self.sigma = estimate_noise_ps(self.lightcurve, estfrac=peestfrac)[0]
        elif noiseestmethod == 'tailveto':
            self.sigma = estimate_noise_tv(self.lightcurve.clc, sigma=tvsigma)[0]
        else:
            print "Error... noise estimation method not recognised"

    def thresholder(self, sigmathresh=4.5, mincontiguous=3, usemedian=False, removeedges=True):
        """
        Perform the thresholding on the data.

        Parameters
        ----------
        sigmathresh : float, default: 4.5
           The number of standard deviations above which a value must be to count as a detection.
        mincontiguous : int, default: 3
           The number of contiguous above threshold values required to give a detection.
        usemedian : bool, default: False
           If True subtract the median value from the light curve, otherwise subtract the mean value.
        removeedges : bool, default: True
           If True remove the edges of the light curve with 5 hours (half the hardcoded running median
           window length) from either end of the data.

        Returns
        -------
        flarelist : list of tuples
           A list of tuples containing the star and end array indices of above-threshold regions
        len(flarelist)
           The number of 'flares' detected.
        """

        flarelist = []
        numcount = 0

        clc = copy(self.lightcurve.clc)

        if removeedges: # remove 5 hours (half the running median window) from either end of the data
            nremove = int(5.*60.*60./self.lightcurve.dt())
            clc = clc[nremove:-nremove]
        else:
            nremove = 0

        if usemedian:
            clc = clc - np.median(clc)
        else:
            clc = clc - np.mean(clc)

        condition = clc > sigmathresh*self.sigma

        # find contiguous regions
        for start, stop in contiguous_regions(condition):
            if stop-start > mincontiguous-1:
                # found a flare!
                flarelist.append((start+nremove, stop+nremove))

        self.nflares = len(flarelist)
        self.flarelist = flarelist

        return flarelist, len(flarelist) # return list of indices and number of flares


class OddsRatioDetector():
    """
    Class to produce the odds ratio detection statistic for a flare versus a selection of noise
    models. The class will also provides thresholding of the log odds ratio for the purpose of
    flare detection.

    Parameters
    ----------
    lightcurve : :class:`Lightcurve`
        The light curve data in which to search for signal.
    bglen : int, default: 55
        The length of the analysis window that slides across the data.
    bgorder : int, default: 4
        The order of the polynomial background variations used in the signal and noise models.
    nsinusoids : int, default: 0
        The number of sinusoids to find and use in the background variations.
    noiseestmethod : string, default: 'powerspectrum'
        The method used to estimate the noise standard deviation of the light curve data.
    psestfrac : float, default: 0.5
        If using the 'powerspectrum' method of noise estimation (:func:`.estimate_noise_ps`) this
        gives the fraction of the spectrum (starting from the high frequency end) used in the noise
        estimate. This value can be between 0 and 1.
    tvsigma : float, default: 1.0
        If using the 'tailveto' method of noise estimation (:func:`.estimate_noise_tv`) this given
        the standard deviation equivalent to the probability volume required for the noise estimate
        e.g. a value of 1.0 means the estimate is formed from the central 68% of the data's
        cumulative probability distribution. This value must be greater than 0.
    flareparams : dict, default: {'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)}
        A dictionary containing the flare parameters 'tauexp' and 'taugauss' giving tuples of each of
        their lower and upper values (in seconds) along with the number of grid points spanning that
        range (if the tuple contains only a single value then this will be the fixed value of that
        parameter). These will be numerically marginalised over to produce the log odds ratio.
    noisepoly : bool, default: True
        If True then the noise model will include a polynomial background variation (with the same
        length and order as used in the signal model and set by `bglen` and `bgorder`.
    noiseimpulse : bool, default: True
        If True then the noise model will include an impulse model (:class:`.Impulse`) on top of a
        polynomial background variation.
    noiseimpulseparams : dict, default: {'t0', (0.,)}
        A dictionary containing the impulse parameters 't0' giving a tuple of its lower, and upper
        values (in seconds) and the number of grid points spanning that range (if a single value is
        given in the tuple then the parameter will be fixed at that value). This range will be
        numerically marginalised over. For the default values `0.` corresponds to the impulse being
        at the centre of the analysis window.
    noiseexpdecay : bool, default: True
        If True then the noise model will include a purely exponential decay model
        (:class:`Expdecay`) on top of a polynomial background variation.
    noiseexpdecayparams : dict, default: {'tauexp': (0.0, 0.25*60*60, 3)}
        A dictionary containing the exponential decay parameter 'tauexp' giving a tuples of its lower
        and upper values (in seconds) and the number of grid points spanning that range (if the
        tuple contains only a single value then this will be the fixed value of that parameter).
        This will be numerically marginalised over to produce the log odds ratio.
    noiseexpdecaywithreverse : bool, default: True
        If True then the noise model will include an exponential rise model (just the reverse of
        the exponential decay) on top of a polynomial background variation. This will have the same
        parameters as defined in `noiseexpdecayparams`.
    noisestep : bool, default: False
        If True then the noise model will include a step function model (:class:`.Step`) on top of a
        polynomial background variation.
    noisestepparams : dict, default: {'t0', (0.,)}
        A dictionary containing the step function parameters 't0' giving a tuple of its lower, and upper
        values (in seconds) and the number of grid points spanning that range (if a single value is
        given in the tuple then the parameter will be fixed at that value). This range will be
        numerically marginalised over. For the default values `0.` corresponds to the step being
        at the centre of the analysis window.
    ignoreedges : bool, default: True
        If this is true then any output log odds ratio will have the initial and final `bglen` /2
        values removed. This removes values for which the odds ratio has been calculated using
        fewer data points.

    Notes
    -----
    In the future this could be made more generic to allow any model as the signal model,
    rather than specifically being the flare model. Further noise models could also be added.
    """

    def __init__(self,
                 lightcurve,
                 bglen=55,
                 bgorder=4,
                 nsinusoids=0,
                 noiseestmethod='powerspectrum',
                 psestfrac=0.5,
                 tvsigma=1.0,
                 flareparams={'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)},
                 noisepoly=True,
                 noiseimpulse=True,
                 noiseimpulseparams={'t0': (np.inf,)},
                 noiseexpdecay=True,
                 noiseexpdecayparams={'tauexp': (0.0, 0.25*60*60, 3)},
                 noiseexpdecaywithreverse=True,
                 noisestep=False,
                 noisestepparams={'t0': (np.inf,)},
                 ignoreedges=True):

        self.lightcurve = deepcopy(lightcurve)
        self.bglen = bglen
        self.bgorder = bgorder
        self.nsinusoids = nsinusoids

        # set flare ranges
        self.set_flare_params(flareparams=flareparams)

        # set noise estimation method
        self.set_noise_est_method(noiseestmethod=noiseestmethod, psestfrac=psestfrac, tvsigma=tvsigma)

        # set noise models
        self.set_noise_poly(noisepoly=noisepoly) # polynomial background
        self.set_noise_impulse(noiseimpulse=noiseimpulse, noiseimpulseparams=noiseimpulseparams) # impulse background
        self.set_noise_expdecay(noiseexpdecay=noiseexpdecay, noiseexpdecayparams=noiseexpdecayparams, withreverse=noiseexpdecaywithreverse)
        self.set_noise_step(noisestep=noisestep, noisestepparams=noisestepparams)
        self.set_ignore_edges(ignoreedges=ignoreedges)

    def set_ignore_edges(self, ignoreedges=True):
        """
        Set whether to ignore the edges of the odds ratio i.e. points within half the
        background window of the start and end of the light curve.

        Parameters
        ----------
        ignoreedges : bool, default: True
            If True then the ends of the log odds ratio will be ignored.
        """
        self.ignoreedges = ignoreedges

    def set_flare_params(self, flareparams={'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)}):
        """
        Set the Gaussian rise ('taugauss') and exponential decay ('tauexp') timescale parameters for the
        flare parameter grid. This can also contain parameter ranges for 't0' if required,
        but otherwise this will default to inf (which gives the centre of the time series).

        Parameters
        ----------
        flareparams : dict, default: {'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)}
            A dictionary of tuples for the parameters 'taugauss' and 'tauexp'. Each must either be a
            single value of three values for the low end, high end (both in seconds) and number of
            parameter points.
        """

        if not flareparams.has_key('taugauss'):
            raise ValueError("Error... dictionary has no parameter 'taugauss'")
        if not flareparams.has_key('tauexp'):
            raise ValueError("Error... dictionary has no parameter 'tauexp'")

        if not flareparams.has_key('t0'):
            flareparams['t0'] = (np.inf,)

        flareparams['amp'] = (1.,)

        self.flareparams = flareparams

    def set_noise_est_method(self, noiseestmethod='powerspectrum', psestfrac=0.5, tvsigma=1.0):
        """
        Set the noise estimation method and its parameters.

        Parameters
        ----------
        noiseestmethod : string, default: 'powerspectrum'
            The noise estimation method. Either 'powerspectrum' to use :func:`.estimate_noise_ps`, or
            'tailveto' to use :func:`.estimate_noise_tv`.
        psestfrac : float, default: 0.5
            The fraction of the upper end of the power spectrum to use for the 'powerspectrum'
            method (must be between 0 and 1).
        tvsigma : float, default: 1.0
            The number of 'standard deviations' corresponding to the central probability volume
            used in the 'tailveto' method.
        """
        self.psestfrac = None
        self.tvsigma = None
        self.noiseestmethod = noiseestmethod
        if noiseestmethod == 'powerspectrum':
            self.psestfrac = psestfrac
        elif noiseestmethod == 'tailveto':
            self.tvsigma = tvsigma
        else:
            print "Noise estimation method %s not recognised" % noiseestmethod

    def set_noise_poly(self, noisepoly=True):
        """
        Set the noise model to include a polynomial background model.

        Parameters
        ----------
        noisepoly : bool, default: True
            Set to True if this model is to be used.
        """
        self.noisepoly = noisepoly

    def set_noise_impulse(self, noiseimpulse=True, noiseimpulseparams={'t0': (np.inf,)}, positive=False):
        """
        Set the noise model to include a delta function impulse (:class:`.Impulse`) on a polynomial
        background variation. Also set the range of times of the impulse, which will be numerically
        marginalise over.

        Parameters
        ----------
        noiseimpulse : bool, default: True
            Set to True if this model is used.
        noiseimpulseparams : dict, default: {'t0': (inf,)}
            A dictionary of tuples of the parameter ranges. 't0' is the only allowed parameter. The
            tuple should either be a single value or three values giving the low end, high end
            and number of parameter points. A default 't0' of inf will set it to the centre of the
            time series.
        positive : bool, default: False
            If True then only have the impulse amplitude marginalised over positive values.
            Otherwise it can have either sign and in marginalised between -infinity and infinity.
        """
        self.noiseimpulse = noiseimpulse

        if not noiseimpulseparams.has_key('t0'):
            raise ValueError("Error... no 't0' value set")

        noiseimpulseparams['amp'] = (1.,)

        self.noiseimpulseparams = noiseimpulseparams

        self.noiseimpulsepositive = positive

    def set_noise_expdecay(self, noiseexpdecay=True, noiseexpdecayparams={'tauexp': (0.0, 0.25*60*60, 3)}, withreverse=True):
        """
        Set the noise model to include an exponential decay (and potentially additionally, as an
        extra noise model, an exponential rise) on top of a polynomial background variation.
        Also, set the range of the time scale parameter 'tauexp' for the exponential decay (used for
        both the decay and rise models), which will be analytically marginalised over. The
        parameters can also take the 't0' ranges values, but otherwise this will be set to 0
        (the centre of the time series).

        Parameters
        ----------
        noiseexpdecay : bool, default: True
            Set to True if this model is used.
        noiseexpdecayparams : dict, default: {'tauexp': (0.0, 0.25*60*60, 3)}
            A dictionary of tuples for the parameter 'tauexp'. It must either be a single value of
            three values for the low end, high end (both in seconds) and number of parameter points.
        withreverse : bool, default: True
            Set to true if there should also be an exponential rise model including in the noise
            model.
        """
        self.noiseexpdecay = noiseexpdecay
        self.noiseexpdecaywithreverse = withreverse

        if not noiseexpdecayparams.has_key('tauexp'):
            raise ValueError("Error... 'tauexp' parameter range not given.")

        if not noiseexpdecayparams.has_key('t0'):
            noiseexpdecayparams['t0'] = (np.inf,)

        noiseexpdecayparams['amp'] = (1.,)

        self.noiseexpdecayparams = noiseexpdecayparams

    def set_noise_step(self, noisestep=False, noisestepparams={'t0': (np.inf,)}):
        """
        Set the noise model to include a step function (:class:`.Step`) on a polynomial
        background variation. Also set the range of times of the step function, which will be numerically
        marginalise over.

        Parameters
        ----------
        noisestep : bool, default: False
            Set to True if this model is used.
        noisestepparams : dict, default: {'t0': (inf,)}
            A dictionary of tuples of the parameter ranges. 't0' is the only allowed parameter. The
            tuple should either be a single value or three values giving the low end, high end
            and number of parameter points. A 't0' default of inf will set t0 to the centre of the
            time series.
        """
        self.noisestep = noisestep

        if not noisestepparams.has_key('t0'):
            raise ValueError("Error... 't0' parameter range not set.")

        noisestepparams['amp'] = (1.,)

        self.noisestepparams = noisestepparams

    def oddsratio(self):
        """
        Get a time series of log odds ratio for data containing a flare *and* polynomial background
        versus a selection of noise models. For the flare and noise models all parameter values
        (expect the central time of the model) are analytically, or numerically (using the
        trapezium rule) marginalised over.

        Each of the noise models, :math:`\\mathcal{O}^{\\textrm noise}_i`, in the denominator of
        the odds ratio are combined independently, such that

        .. math::

            \\mathcal{O} = \\frac{\\mathcal{O}^{\\textrm signal}}{\\sum_i \\mathcal{O}^{\\textrm noise}_i}

        where :math:`\\mathcal{O}^{\\textrm signal}` is the signal model.

        If no noise models are specified then the returned log odds ratio will be for the signal
        model versus Gaussian noise.
        """

        # get flare odds ratio
        Mf = Flare(self.lightcurve.cts, amp=1, paramranges=self.flareparams)
        Bf = Bayes(self.lightcurve, Mf)
        Bf.bayes_factors_marg_poly_bgd(bglen=self.bglen,
                                       bgorder=self.bgorder,
                                       nsinusoids=self.nsinusoids,
                                       noiseestmethod=self.noiseestmethod,
                                       psestfrac=self.psestfrac,
                                       tvsigma=self.tvsigma)
        Of = Bf.marginalise_full()

        noiseodds = []
        # get noise odds ratios
        if self.noisepoly:
            Bg = Bf.bayes_factors_marg_poly_bgd_only(bglen=self.bglen,
                                                     bgorder=self.bgorder,
                                                     nsinusoids=self.nsinusoids,
                                                     noiseestmethod=self.noiseestmethod,
                                                     psestfrac=self.psestfrac,
                                                     tvsigma=self.tvsigma)

            noiseodds.append(Bg)

        del Mf

        if self.noiseimpulse:
            # setup impulse model
            M = Impulse(self.lightcurve.cts, amp=1, paramranges=self.noiseimpulseparams)
            Bi = Bayes(self.lightcurve, M)
            Bi.bayes_factors_marg_poly_bgd(bglen=self.bglen,
                                           bgorder=self.bgorder,
                                           nsinusoids=self.nsinusoids,
                                           halfrange=self.noiseimpulsepositive,
                                           noiseestmethod=self.noiseestmethod,
                                           psestfrac=self.psestfrac,
                                           tvsigma=self.tvsigma)
            Oi = Bi.marginalise_full()

            noiseodds.append(Oi.lnBmargAmp)
            del M

        if self.noiseexpdecay:
            # setup short flare-like model
            M = Expdecay(self.lightcurve.cts, amp=1, paramranges=self.noiseexpdecayparams)
            Be = Bayes(self.lightcurve, M)
            Be.bayes_factors_marg_poly_bgd(bglen=self.bglen,
                                           bgorder=self.bgorder,
                                           nsinusoids=self.nsinusoids,
                                           noiseestmethod=self.noiseestmethod,
                                           psestfrac=self.psestfrac,
                                           tvsigma=self.tvsigma)
            Oe = Be.marginalise_full()
            noiseodds.append(Oe.lnBmargAmp)
            del M

            if self.noiseexpdecaywithreverse:
                M = Expdecay(self.lightcurve.cts, amp=1, reverse=True, paramranges=self.noiseexpdecayparams)
                Ber = Bayes(self.lightcurve, M)
                Ber.bayes_factors_marg_poly_bgd(bglen=self.bglen,
                                                bgorder=self.bgorder,
                                                nsinusoids=self.nsinusoids,
                                                noiseestmethod=self.noiseestmethod,
                                                psestfrac=self.psestfrac,
                                                tvsigma=self.tvsigma)
                Oer = Ber.marginalise_full()
                noiseodds.append(Oer.lnBmargAmp)
                del M

        if self.noisestep:
            # setup step function model
            M = Step(self.lightcurve.cts, amp=1, paramranges=self.noisestepparams)
            Bs = Bayes(self.lightcurve, M)
            Bs.bayes_factors_marg_poly_bgd(bglen=self.bglen,
                                           bgorder=self.bgorder,
                                           nsinusoids=self.nsinusoids,
                                           halfrange=False,
                                           noiseestmethod=self.noiseestmethod,
                                           psestfrac=self.psestfrac,
                                           tvsigma=self.tvsigma)

            Os = Bs.marginalise_full()
            noiseodds.append(Os.lnBmargAmp)
            del M

        # get the total odds ratio
        if self.ignoreedges and self.bglen != None:
            valrange = np.arange(int(self.bglen/2), len(Of.lnBmargAmp)-int(self.bglen/2))
            ts = np.copy(self.lightcurve.cts[valrange])
        else:
            valrange = range(0, len(Of.lnBmargAmp))
            ts = np.copy(self.lightcurve.cts)

        lnO = []
        for i in valrange:
            denom = -np.inf
            for n in noiseodds:
                denom = logplus(denom, n[i])

            if len(noiseodds) > 0:
                lnO.append(Of.lnBmargAmp[i] - denom)
            else:
                lnO.append(Of.lnBmargAmp[i])

        return lnO, ts

    def impulse_excluder(self, lnO, ts, exclusionwidth=5):
        """
        Return a copy of the odds ratio time series with sections excluded based on containing features
        consistent with impulse artifacts. The type of feature is that which comes about due to impulses
        in the data ringing up the signal template as it moves onto and off-of the impulse. These give
        rise to a characteristic M-shaped feature with the middle dip (when the impulse model well
        matches the data) giving a string negative odds ratio.

        Parameters
        ----------
        lnO : list or :class:`numpy.array`
            A time series array of log odds ratios.
        exclusionwidth : int, default: 5
            The number of points either side of the feature to be excluded. In practice this should be
            based on the charactistic maximum flare width.
        """

        # find log odds ratios < -5 (i.e. favouring the impulse/noise model
        negidxs = np.arange(len(lnO))[np.copy(lnO) < -5.]

        idxarray = np.ones(len(lnO), dtype=np.bool) # array to say whether values should be excluded or not

        # check whether to exclude or not based on M shaped profile
        for idx in negidxs:
            if idx > 1 and idx < len(lnO)-2:
                c1 = False
                c2 = False
                # check previous value is positive and value before that is positive, but less than next one
                if lnO[idx-1] > 0 and lnO[idx-2] > 0 and lnO[idx-2] < lnO[idx-1]:
                    c1 = True
                # check next value is positive and value after that is positive, but less than previous one
                if lnO[idx+1] > 0 and lnO[idx+2] > 0 and lnO[idx+2] < lnO[idx+1]:
                    c2 = True

                # set exclusion is both these are true
                if c1 and c2:
                    stidx = idx - exclusionwidth
                    if stidx < 0:
                        stidx = 0
                    enidx = idx + exclusionwidth
                    if enidx > len(lnO)-1:
                        enidx = len(lnO)-1

                    idxarray[stidx:enidx] = False

        # return arrays with parts excluded
        return np.copy(lnO)[idxarray], np.copy(ts)[idxarray]


    def thresholder(self, lnO, thresh, expand=0, returnmax=True):
        """
        Output an list of array start and end indices for regions where the log odds ratio is
        greater than a given threshold `thresh`. Regions can be expanded by a given amount to allow
        close-by regions to be merged.

        This is used for flare detection.

        Parameters
        ----------
        lnO : list or :class:`numpy.array`
            A time series array of log odds ratios.
        thresh : float
            The log odds ratio threshold for "detections".
        expand : int, default:0
            Expand each contiguous above-threshold region by this number of indices at either side.
            After expansion any overlapping or adjacent regions will be merged into one region.
        returnmax : bool, default: True
            If True then return a list of tuples containing the maximum log odds ratio value in each
            of the "detection" segments and the index of that value.

        Returns
        -------
        flarelist : list of tuples
            A list of tuples of start and end indices of contiguous regions for the "detections".
        numflares : int
            The number of contiguous regions i.e. the number of detected flares.
        maxlist : list of tuples
            If `returnmax` is true then this contains a list of tuples with the maximum log
            odds ratio value in each of the "detection" segments and the index of that value.
        """

        # find contiguous regions
        flarelist = []
        for start, stop in contiguous_regions(np.copy(lnO) > thresh): #  make sure lnO is a numpy array by copying
           flarelist.append((start, stop))

        # expand segments if required, and then merge any adjacent or overlapping segments
        if expand > 0:
            if len(flarelist) == 1: # if only one flare
                segtmp = list(flarelist[0])
                segtmp[0] = segtmp[0]-expand    # expand the segment
                segtmp[-1] = segtmp[-1]+expand  # expand the segment

                # check if segment now goes out of range and if so correct it
                if segtmp[0] < 0:
                    segtmp[0] = 0
                if segtmp[-1] >= len(lnO):
                    segtmp[-1] = len(lnO)

                flarelist = [segtmp]
            elif len(flarelist) > 1:
                flisttmp = []

                # expand each segment
                for segn in flarelist:
                    segtmp = list(segn)
                    segtmp[0] = segtmp[0]-expand
                    segtmp[-1] = segtmp[-1]+expand

                    # check if segment now goes out of range and if so correct it
                    if segtmp[0] < 0:
                        segtmp[0] = 0
                    if segtmp[-1] >= len(lnO):
                        segtmp[-1] = len(lnO)

                    flisttmp.append(tuple(segtmp))

                flarelist = flisttmp

                # with expanded segments now check for overlapping or adjacent segments and merge
                j = 0
                newsegs = []
                while True:
                    thisseg = flarelist[j]
                    j = j+1
                    for k in range(j, len(flarelist)):
                        nextseg = flarelist[k]
                        if thisseg[-1] >= nextseg[0]: # overlapping or adjacent segment
                            thisseg = (thisseg[0], nextseg[-1])
                            j = j+1
                        else:
                            break

                    newsegs.append(thisseg)

                    # break from loop
                    if j >= len(flarelist):
                        break

                flarelist = list(newsegs)

        lnOc = np.copy(lnO)

        # return the list of maximum values and indices for the detections
        if returnmax:
            maxlist = []
            for segn in flarelist:
                v = np.arange(segn[0], segn[-1])
                i = np.argmax(lnOc[v])
                maxlist.append((lnOc[v[i]], v[i]))

            return flarelist, len(flarelist), maxlist
        else:
            return flarelist, len(flarelist) # return list of indices and number of flares
