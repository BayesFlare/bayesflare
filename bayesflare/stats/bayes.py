"""

"""
from math import log
import numpy as np
from copy import copy, deepcopy
from ..noise import estimate_noise_ps, estimate_noise_tv, highpass_filter_lightcurve, savitzky_golay
from ..models import *
from .general import *
#from .thresholding import Thresholder
from math import *
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as pl
from multiprocessing import Pool


# function to find peaks in a time series, and return a sorted list of peak positions and amplitudes
# from largest to smalls
def find_peaks(x):
    """
    Find the peaks/maxima in a time series `x`. Output the peak amplitude and the indices of them
    in descending order of amplitude.
    """

    # a peak will be a point where the gradient changes from positive to negative, get gradient of data
    grad = np.diff(x)

    peakidx = [] # array to hold indices of peaks

    # now find the zero crossings from positive to negative
    for i in range(len(grad)-1):
        if grad[i] > 0. and grad[i+1] < 0.:
            peakidx.append(i)

    # convert into a numpy array
    nppeakidx = np.array(peakidx);

    # get the amplitudes of the peaks
    peakamps = x[nppeakidx]

    # sort into descending order
    sortarray = np.array(sorted(zip(peakamps, nppeakidx), reverse=True))

    # return the peak amplitudes in descending order and their indices
    return sortarray[:,0], sortarray[:,1].astype(int)


def spectrum_peak_frequencies(lc, npeaks=5):
    """
    Find the frequencies of the 'npeaks' largest peaks in a periodogram of the lightcurve.

    Parameters
    ----------
    lc : :class:`.Lightcurve` object
       An instance of a :class:`.Lightcurve`
    npeaks : int, optional, default: 5
    """

    # get periodogram
    pgram, freqs = lc.periodogram(oversample=25)

    # get the peaks
    pamps, pidxs = find_peaks(pgram)

    # return the frequecies of the n largest peaks
    if len(pamps) < npeaks:
        print >> sys.stderr, "There were fewer peaks than requested (%d), so returning %d" % (npeaks, len(pamps))
        npeaks = len(pamps)

    lc.sinusoid_freqs = freqs[pidxs[:npeaks]]

    return freqs[pidxs[:npeaks]]


class Bayes():
    """
    The Bayes class contains the functions responsible for calculating the Bayesian odds ratios for
    a model given the light curve data.

    Parameters
    ----------
    lightcurve : :class:`.Lightcurve` object
       An instance of a :class:`.Lightcurve` which the Bayesian odds ratios will be calculated for.
    model : Model object
       An instance of a Model object which will be used to generate odds ratios.
    """

    # Object data
    premarg = {}         # Place to store the pre-marginalised bayes factor arrays

    #def __init__(self, lightcurve, model, confidence=0.999, results_root=None):
    def __init__(self, lightcurve, model):
        """
        The initiator method
        """
        self.lightcurve = lightcurve
        self.model      = deepcopy(model)
        self.ranges     = deepcopy(model.ranges)
        self.confidence = 0.999
        self.noise_ev = self.noise_evidence()

    def bayes_factors(self, **kwargs):
        """
        Work out the logarithm of the Bayes factor for a signal consisting of the model (e.g. a
        flare) in the light curve data compared with Gaussian noise (the model and the light curve
        must be defined on initialise of the class) for a range of model parameters. Of the model
        parameters the amplitude will be analytically marginalised over. The Bayes factor for each
        model time stamp (i.e. the central time of a flare) will be calculated over the parameter
        space containing the additional model parameters, as defined by the model. All these will
        require subsequent marginalisation if necessary.

        If the light curve has had detrending applied then the model will also get detrended in the
        same way.
        """

        model = self.model

        N = len(self.lightcurve.cts)
        s = np.copy(model.shape)
        l = np.product(s)
        s = np.append(s,N)

        self.lnBmargAmp = np.zeros(s)

        x = self.lightcurve.cts
        z = self.lightcurve.clc
        sk = estimate_noise_ps(self.lightcurve)[1]
        #sk = estimate_noise_tv(self.lightcurve.clc, 2.5)[0]**2

        for i in np.arange(l):
            q = np.unravel_index(i, model.shape)
            # m = model(i)
            m = model(i, filt=self.lightcurve.detrended, filtermethod=self.lightcurve.detrend_method, nbins=self.lightcurve.detrend_nbins,
                      order=self.lightcurve.detrend_order, filterknee=self.lightcurve.detrend_knee)
            if m == None:
                # if the model is not defined (e.g. for the flare model when tau_g > tau_e)
                # set probability to zero (log probability to -inf)
                self.lnBmargAmp[q][:] = np.ones(N)*-np.inf
                continue
            # Generate the model flare
            m = m.clc

            # Run the xcorr and perform the analytical marginalisation over amplitude
            B = log_marg_amp(z, m, sk)
            # Apply Bayes Formula
            self.lnBmargAmp[q][:] = B + np.sum(model.priors)

            self.premarg[model.identity_type()] = self.lnBmargAmp

    def bayes_factors_marg_poly_bgd(self,
                                    bglen=55,               # length of background polynomial window (must be odd)
                                    bgorder=4,              # background polynomial order
                                    nsinusoids=0,           # the number of sinusoids in the background model
                                    noiseestmethod='powerspectrum',
                                    psestfrac=0.5,
                                    tvsigma=1.0,
                                    halfrange=True,
                                    ncpus=None):
        """
        Work out the logarithm of the Bayes factor for a signal consisting of the model (e.g. a
        flare) *and* a background variation defined by a polynomial of order, `bgorder`, *and*
        potentially a background variation consisting of `nsinusoids` sinusoidal frequency components,
        compared with Gaussian noise given light curve data (the
        model and data must be defined on initialisation of the class). The Bayes factor is
        analytically marginalised over the model amplitude, background polynomial coefficients and
        sinusoid amplitudes and phases. The Bayes factor for each model time stamp (i.e. the central
        time of a flare) will be calculated over the parameter space containing the model parameters,
        as defined by the model. All these will require subsequent marginalisation if necessary. If
        just a polynomial background is being used then the background variation will be calculated
        as a running window (of length `bglen`) around the main model time stamp. If sinusoids are used
        in the background model then a sliding window is not used and the sinusoids plus polynomial are
        assumed to model the whole light curve (Note: this could be changed in the future to allow a
        sliding window again if such models still leave significant residuals when trying to fit
        background variations).

        No priors are used on the amplitude parameters being marginalised over. These could
        subsequently be applied, so long as they are constant values that are large compared
        to the expected parameter space over which the likelihood is non-negligable. However,
        if comparing models for which the same amplitude parameters are used then these priors
        would cancel anyway. The exception is if the model amplitude marginalisation, rather than
        the background polynomial amplitude, covers the 0 to :math:`\infty` range rather than
        the full :math:`-\infty` to :math:`\infty` range. In this case a prior of 0.5 is
        applied.

        Parameters
        ----------
        bglen : int, default: 55
            The length, in bins, of the background variation polynomial window. If used this must be odd,
            but if set to None then the window will just be the whole light curve.
        bgorder : int, default: 4
            The order of the polynomial background variation. If `bgorder` is -1 then no polynomial
            background variation is used, and this functions defaults to use :func:`bayes_factors`.
        nsinusoids : int, default: 0
            The number of background frequency components to be marginalised over. If this is used
            then the data won't be broken up into `bglen` segments, but `bglen` will just be used
            as the length of the model.
        noiseestmethod : string, default: 'powerspectrum'
            The method for estimating the noise standard deviation. This can either be
            'powerspectrum' (which estimates the noise from the power spectrum of the data) or
            'tailveto' (which estimates the noise using the central part of the data's
            distribution).
        psestfrac : float, default: 0.5
            If 'powerspectrum' is the required noise estimation method then the value set here
            (between >0 and 1) gives the fraction of the upper end of the power spectrum to be used.
        tvsigma : float, default: 1.0
            If 'tailveto' is the required noise estimation method then the value here (>0) gives the
            number of standard deviations for the probability volume of the central distribution to
            be used.
        halfrange : boolean, default: True
            If this is 'True' then the defined signal model amplitude will be integrated over the
            ranges from 0 to infinity. If it is 'False' then the integral will be from -infinity to
            infinity.
        ncpus : int, default: None
            The number of parallel CPUs to run the likelihood calculation on using
            :mod:`multiprocessing`. The default of None means that all available CPUs on a machine
            will be used.

        See Also
        --------
        bayes_factors : This function performs no analytical marginalisation over a polynomial background model
        bayes_factors_marg_poly_bgd_only : Similar to this function, but without the signal model.
        bayes_factors_marg_poly_bgd_chunk : Similar to this function, but only computing the Bayes
                                            factor for a small chunk of the light curve data.
        """

        # check bglen is odd
        if bglen != None:
            if bglen % 2 == 0 and nsinusoids == 0:
                print "Error... Background length (bglen) must be an odd number"
                return

        # get noise estimate on a filtered lightcurve to better represent just the noise
        tmpcurve = deepcopy(self.lightcurve)
        if tmpcurve.detrended == False: # only detrend if not already detrended
            tmpcurve.detrend(method='savitzkygolay', nbins=bglen, order=bgorder)
        if noiseestmethod == 'powerspectrum':
            sk = estimate_noise_ps(tmpcurve, estfrac=psestfrac)[0]
        elif noiseestmethod == 'tailveto':
            sk = estimate_noise_tv(tmpcurve.clc, sigma=tvsigma)[0]
        else:
            print "Noise estimation method must be 'powerspectrum' or 'tailveto'"
            return None
        del tmpcurve

        N = len(self.lightcurve.cts)
        if bglen == None:
            nsteps = int(N/2)
        else:
            nsteps = int(bglen/2)

        npoly = bgorder+1 # number of polynomial coefficients

        model = self.model

        # get data
        d = np.copy(self.lightcurve.clc)

        if bglen != None and nsinusoids == 0:
            dt = model.ts[1]-model.ts[0]                # time step
            idxt0 = int((model.t0-model.ts[0])/dt)+1    # index of t0 for the model

            idx1 = idxt0 - nsteps
            idx2 = idxt0 + nsteps + 1

            if idx1 < 0:
                # shift times
                mts = model.ts[:bglen]
            elif idx2 > N-1:
                mts = model.ts[-bglen:]
            else:
                mts = model.ts[idx1:idx2] # time stamps for model creation
        else:
            mts = model.ts # just use original model times

        if nsinusoids != 0 and self.lightcurve.sinusoid_freqs is None:
            # get the frequencies of the nsinusoids largest peaks in the spectrum
            freqs = spectrum_peak_frequencies(self.lightcurve, npeaks=nsinusoids)
        elif nsinusoids != 0 and len(self.lightcurve.sinusoid_freqs) == nsinusoids:
            freqs = self.lightcurve.sinusoid_freqs

        # set amplitude priors
        if halfrange:
            ampprior = np.log(0.5)
        else:
            ampprior = 0.

        nwaves = 2*nsinusoids

        if nsinusoids != 0 or bglen == None:
            bgmodels = np.ndarray((npoly + nwaves, N))
            tsp = np.linspace(0., 1., N)
        else:
            bgmodels = np.ndarray((npoly, bglen))
            tsp = np.linspace(0., 1., bglen)

        ts = self.lightcurve.cts-self.lightcurve.cts[0]

        for i in range(npoly + nwaves):
            if i < nsinusoids:
                bgmodels[i] = np.cos(2.*np.pi*freqs[i]*ts)
            elif i >= nsinusoids and i < nwaves:
                bgmodels[i] = np.sin(2.*np.pi*freqs[i-nsinusoids]*ts)
            else:
                bgmodels[i] = tsp**(i-nwaves)

        bgcross = np.zeros((npoly + nwaves, npoly + nwaves, N))

        # get noise time series (data error and error estimate added in quadrature)
        noisevar = (sk**2+self.lightcurve.cle**2)

        # background cross terms for each time step
        for i in range(npoly + nwaves):
            for j in range(i, npoly + nwaves):
                if nsinusoids == 0 and bglen != None:
                    #  use the correct part of the noise variance for the particular stamp
                    for k in range(N):
                        if k < nsteps:
                            bgm = bgmodels[i,nsteps-k:]*bgmodels[j,nsteps-k:]
                            bgm = bgm/noisevar[:len(bgm)]
                        elif k >= N-nsteps:
                            bgm = bgmodels[i,:(N-nsteps-k-1)]*bgmodels[j,:(N-nsteps-k-1)]
                            bgm = bgm/noisevar[-len(bgm):]
                        else:
                            bgm = bgmodels[i]*bgmodels[j]/noisevar[k-nsteps:k+nsteps+1]

                        bgcross[i,j,k] = np.sum(bgm)
                else:
                    bgcross[i,j] = np.sum(bgmodels[i]*bgmodels[j]/noisevar)*np.ones(N)

        # store models, so not regenerating them (these are truncated to the length of bglen)
        if bglen != None:
            ms = np.ndarray(tuple(model.shape) + (bglen,))
        else:
            ms = np.ndarray(tuple(model.shape) + (N,))
        priors = np.ndarray(tuple(model.shape))
        mparams = {}

        # squared model terms for each time step
        mdcross = np.ndarray(tuple(model.shape)+(N,))

        # get the "model" and background cross terms
        mdbgcross = np.ndarray(tuple(model.shape) + (npoly+nwaves,N))

        for i in range(np.product(model.shape)):
            m = model(i, ts=mts, filt=False) # use the original model without the shape having been changed
            q = np.unravel_index(i, model.shape)

            # get prior
            for k in range(len(model.shape)):
                # set parameter dict for prior function
                mparams[model.paramnames[k]] = self.ranges[model.paramnames[k]][q[k]]

            priors[q] = model.prior(mparams)

            if m == None or priors[q] == -np.inf:
                if bglen != None:
                    ms[q] = -np.inf*np.ones(bglen)
                else:
                    ms[q] = -np.inf*np.ones(N)
                mdcross[q] = -np.inf*np.ones(N)
            else:
                ms[q] = m.clc

                for k in range(N):
                    if k < nsteps:
                        mm = m.clc[nsteps-k:]**2
                        mm = mm/noisevar[:len(mm)]
                    elif k >= N-nsteps:
                        mm = m.clc[:(N-nsteps-k-1)]**2
                        mm = mm/noisevar[-len(mm):]
                    else:
                        mm = m.clc**2/noisevar[k-nsteps:k+nsteps+1]

                    mdcross[q+(k,)] = np.sum(mm)

            # model*background terms
            for j in range(npoly+nwaves):
                if m != None and priors[q] != -np.inf:
                    if nsinusoids == 0 and bglen != None:
                        for k in range(N):
                            if k < nsteps:
                                mgm = bgmodels[j,nsteps-k:]*ms[q][nsteps-k:]
                                mgm = mgm/noisevar[:len(mgm)]
                            elif k >= N-nsteps:
                                mgm = bgmodels[j,:(N-nsteps-k-1)]*ms[q][:(N-nsteps-k-1)]
                                mgm = mgm/noisevar[-len(mgm):]
                            else:
                                mgm = bgmodels[j]*ms[q]/noisevar[k-nsteps:k+nsteps+1]

                            mdbgcross[q+(j,k)] = np.sum(mgm)
                    else:
                        mdbgcross[q+(j,)] = np.correlate(bgmodels[j]/noisevar, ms[q], 'same')
                else:
                    mdbgcross[q+(j,)] = np.zeros(N)

        # get the data crossed with the background polynomial terms
        dbgr = np.ndarray((npoly+nwaves, N))

        # whiten the data using the estimated noise and the lightcurve noise added in quadrature
        d = d/noisevar

        for i in range(npoly+nwaves):
            if nsinusoids == 0 and bglen != None:
                dbgr[i] = np.correlate(d, bgmodels[i], 'same')
            else:
                dbgr[i] = np.sum(d*bgmodels[i])*np.ones(N)

        # initialise the log-likelihood ratio
        s = tuple(model.shape) + (N,)
        self.lnBmargAmp = -np.inf*np.ones(s)

        # Parallel-ize it! Run different model parameter calculation in a parallel way if multiple CPUs
        # are available.
        l = np.product(model.shape)

        sk = np.sqrt(noisevar)

        pool = Pool(processes=ncpus)
        Ms = pool.map_async(log_marg_amp_full_model_wrapper,
                        ((i, model.shape, sk, npoly+nwaves, halfrange, d,
                          ms, bgcross, mdbgcross, mdcross, dbgr)
                          for i in range(l))).get()

        # clean-up
        pool.close()
        pool.join()

        for i in range(l):
            q = np.unravel_index(i, model.shape)

            # get Bayes factors and apply priors
            self.lnBmargAmp[q] = Ms[i] + priors[q] + ampprior

        self.premarg = np.copy(self.lnBmargAmp)

    def bayes_factors_marg_poly_bgd_only(self,
                                         bglen=55,
                                         bgorder=4,
                                         nsinusoids=0,
                                         noiseestmethod='powerspectrum',
                                         psestfrac=0.5,
                                         tvsigma=1.0):
        """
        Get the log Bayes factor for the data matching a sliding polynomial background window (of
        length `bglen` and polynomial order `bgorder`) compared to Gaussian noise. This marginalises
        over the polynomial amplitude coefficients analytically. The difference between this
        function and :func:`bayes_factors_marg_poly_bgd` is that this function does not include the
        signal model.

        See Also
        --------
        bayes_factors_marg_poly_bgd : Similar to this function, but including a signal model, such
                                      as a flare.
        """

        # check bglen is odd
        if bglen != None:
            if bglen % 2 == 0 and nsinusoids == 0:
                print "Error... Background length (bglen) must be an odd number"
                return

        N = len(self.lightcurve.cts)
        if bglen == None:
            nsteps = int(N/2)
        else:
            nsteps = int(bglen/2)

        if bglen > N:
            print "Error... bglen is greater than the data length!"
            return

        """ get noise estimate on a filtered lightcurve to better represent just the noise """
        tmpcurve = deepcopy(self.lightcurve)
        if tmpcurve.detrended == False: # only detrend if not already detrende
            tmpcurve.detrend(method='savitzkygolay', nbins=bglen, order=bgorder)
        if noiseestmethod == 'powerspectrum':
            sk = estimate_noise_ps(tmpcurve, estfrac=psestfrac)[0]
        elif noiseestmethod == 'tailveto':
            sk = estimate_noise_tv(tmpcurve.clc, sigma=tvsigma)[0]
        else:
            print "Noise estimation method must be 'powerspectrum' or 'tailveto'"
            return None
        del tmpcurve

        npoly = bgorder+1 # number of polynomial coefficients
        nwaves = 2*nsinusoids

        # get the background polynomial model cross terms for each t0
        if nsinusoids != 0 or bglen == None:
            bgmodels = np.ndarray((npoly + nwaves, N))
            tsp = np.linspace(0., 1., N)
        else:
            bgmodels = np.ndarray((npoly, bglen))
            tsp = np.linspace(0., 1., bglen)

        if nsinusoids != 0 and self.lightcurve.sinusoid_freqs is None:
            # get the frequencies of the nsinusoids largest peaks in the spectrum
            freqs = spectrum_peak_frequencies(self.lightcurve, npeaks=nsinusoids)
        elif nsinusoids != 0 and len(self.lightcurve.sinusoid_freqs) == nsinusoids:
            freqs = self.lightcurve.sinusoid_freqs

        ts = self.lightcurve.cts-self.lightcurve.cts[0]

        for i in range(npoly + nwaves):
            if i < nsinusoids:
                bgmodels[i] = np.cos(2.*np.pi*freqs[i]*ts)
            elif i >= nsinusoids and i < nwaves:
                bgmodels[i] = np.sin(2.*np.pi*freqs[i-nsinusoids]*ts)
            else:
                bgmodels[i] = tsp**(i-nwaves)

        # background cross terms for each time step
        bgcross = np.zeros((npoly + nwaves, npoly + nwaves, N))

        # get noise time series (data error and error estimate added in quadrature)
        noisevar = (sk**2+self.lightcurve.cle**2)

        # background cross terms for each time step
        for i in range(npoly + nwaves):
            for j in range(i, npoly + nwaves):
                if nsinusoids == 0 and bglen != None:
                    #  use the correct part of the noise variance for the particular stamp
                    for k in range(N):
                        if k < nsteps:
                            bgm = bgmodels[i,nsteps-k:]*bgmodels[j,nsteps-k:]
                            bgm = bgm/noisevar[:len(bgm)]
                        elif k >= N-nsteps:
                            bgm = bgmodels[i,:(N-nsteps-k-1)]*bgmodels[j,:(N-nsteps-k-1)]
                            bgm = bgm/noisevar[-len(bgm):]
                        else:
                            bgm = bgmodels[i]*bgmodels[j]/noisevar[k-nsteps:k+nsteps+1]

                        bgcross[i,j,k] = np.sum(bgm)
                else:
                    bgcross[i,j] = np.sum(bgmodels[i]*bgmodels[j]/noisevar)*np.ones(N)

        # get data
        d = np.copy(self.lightcurve.clc)

        # get the data crossed with the background model terms
        dbgr = np.ndarray((npoly+nwaves, N))

        # whiten the data using the estimated noise and the lightcurve noise added in quadrature
        d = d/noisevar

        for i in range(npoly+nwaves):
            if nsinusoids == 0 and bglen != None:
                dbgr[i] = np.correlate(d, bgmodels[i], 'same')
            else:
                dbgr[i] = np.sum(d*bgmodels[i])*np.ones(N)

        self.lnBmargBackground = -np.inf*np.ones(N)

        sk = np.sqrt(noisevar)

        B = log_marg_amp_full_background(sk, N, npoly+nwaves, bgcross, dbgr)

        self.lnBmargBackground = B

        return B

    def marginalise(self, pname):
        """
        Function to reduce the dimensionality of the `lnBmargAmp` :class:`numpy.ndarray` from `N` to
        `N-1` through numerical marginalisation (integration) over a given parameter.

        Parameters
        ----------
        axis: string
            The parameter name of the array that is to be marginalised.

        Returns
        -------
        B : :class:`Bayes`
            A :class:`Bayes` object in which the `lnBmargAmp` :class:`numpy.ndarray` has had one
            parameter marginalised over.
        """

        arr = self.lnBmargAmp
        places = self.ranges[pname]
        axis = self.model.paramnames.index(pname)
        if len(places) > 1:
            x = np.apply_along_axis(logtrapz, axis, arr, places)
        elif len(places) == 1:
            # no marginalisation required just remove the specific singleton dimension via reshaping
            z = arr.shape
            q = np.arange(0,len(z)).astype(int) != axis
            newshape = tuple((np.array(list(z)))[q])
            x = np.reshape(arr, newshape)

        model = copy(self.model)
        model.paramnames.remove(pname)

        B = Bayes(self.lightcurve, model)

        ranges = copy(self.ranges)
        del ranges[pname]
        B.ranges = ranges

        B.lnBmargAmp = x
        return B

    def marginalise_full(self):
        """
        Marginalise over each of the parameters in the `ranges` list in turn.

        Returns
        -------
        A : :class:`Bayes`
            A :class:`Bayes` object for which the `lnBmargAmp` array has been marginalised over all
            parameters in the `ranges` list
        """

        A = self
        for p in self.ranges:
            A = A.marginalise(p)

        return A

    def noise_evidence(self):
        """
        Calculate the evidence that the data consists of Gaussian noise. This calculates the noise
        standard deviation using the 'tailveto' method of :func:`.estimate_noise_tv`.

        Returns
        -------
        The log of the noise evidence value.

        .. note::
            In this the :func:`.estimate_noise_tv` method is hardcoded to use a `tvsigma` value of
            1.0.
        """
        var = estimate_noise_tv(self.lightcurve.clc, 1.0)[0]**2
        noise_ev = -0.5*len(self.lightcurve.clc)*np.log(2.*pi*var) - np.sum(self.lightcurve.clc**2)/(2.*var)

        return noise_ev


def log_marg_amp_full_model_wrapper(params):
    """
    Wrapper to :func:`.log_marg_amp_full_model` and :func:`.log_marg_amp_full_2Dmodel` function that
    takes in a tuple of all the required parameters. This is required to use the
    :mod:`multiprocessing` `Pool.map_async` function.

    Parameters
    ----------
    params : tuple
        A tuple of parameters required by :func:`.log_marg_amp_full_2Dmodel` or
        :func:`.log_marg_amp_full_model`

    Returns
    -------
    margamp : :class:`numpy.ndarray`
        An array containing the logarithm of the likelihood ratio.
    """
    shape = params[1]

    if len(shape) == 2: # specific case for a model with two parameters
        return log_marg_amp_full_2Dmodel(params[0], params[1], params[2], params[3], params[4],
                                         params[5], params[6], params[7], params[8], params[9],
                                         params[10])
    else:
        return log_marg_amp_full_model(params[0], params[1], params[2], params[3], params[4],
                                       params[5], params[6], params[7], params[8], params[9],
                                       params[10])

def log_likelihood_marg_background_wrapper(params):
    """
    Wrapper to :func:`.log_likelihood_marg_background` that takes a tuple of all the required
    parameters. This is required to use the :mod:`multiprocessing` `Pool.map_async` function.

    Parameters
    ----------
    params : tuple
        A tuple of parameters required by :func:`.log_likelihood_marg_background`.

    Returns
    -------
    margamp : :class:`numpy.ndarray`
        An array containing the logarithm of the likelihood ratio.
    """
    return log_likelihood_marg_background(params[0], params[1], params[2], params[3])


class ParameterEstimationGrid():
    """
    Class to perform parameter estimation (i.e. evaluate the posterior distributions
    of a models parameters) for a particular model (e.g. a flare) given some light curve
    data and a grid of parameter points.

    Parameters
    ----------
    modelType : string
        The model for which the parameters are to be estimated (e.g. 'flare')
    lightcurve : :class:`.Lightcurve`
       The light curve data with which to estimate the model parameters.

    """

    modelType = None         #: A string giving the model type
    model = None             #: A model class e.g. :class:`.Flare`
    paramNames = None        #: A list of the parameter names for a model
    paramValues = {}         #: A dictionary of the parameter values used to create the grid
    lightcurve = None        #: The light curve data to be fitted
    noiseSigma = None        #: The light curve noise standard deviation
    prior = None             #: The prior function to be used
    posterior = None         #: The full log posterior (a :class:`numpy.ndarray`)
    margposteriors = {}      #: A dictionary of the marginalised posterior for each parameter
    maxposterior = None      #: The maximum posterior value
    maxpostparams = {}       #: A dictionary of the parameters of the maximum posterior value

    def __init__(self, modelType=None, lightcurve=None):
        """
        Initialise with the model type (currently this can be either 'flare' or 'transit'
        (for the :class:`.Flare` model or :class:`.Transit` model respectively), and a
        :class:`.Lightcurve`.
        """
        if lightcurve == None:
            print "A lightcurve is required as input"
        else:
            self.lightcurve = deepcopy(lightcurve) # the lightcurve data to use

        if modelType == None:
            print "Specify the model type with set_model_type()"
        elif modelType.lower() == 'flare' or modelType.lower() == 'transit':
            self.set_model_type(modelType)
        else:
            print "Unknown model type"

        # set the noise standard deviation

        self.set_sigma(self.lightcurve)

    def set_model_type(self, modelType):
        """
        Set the model type if not done during initialisation. This will also set the associated
        parameter names, and the prior function.

        Parameters
        ----------
        modelType : string
            A string giving the model type (currently either 'flare' or 'transit').
        """
        self.modelType = modelType.lower()
        if self.modelType == 'flare':
            from ..models import Flare
            self.model = Flare(self.lightcurve.cts)
            self.paramNames = self.model.paramnames
        elif self.modelType == 'transit':
            from ..models import Transit
            self.model = Transit(self.lightcurve.cts)
        elif self.modelType == 'gaussian':
            from ..models import Gaussian
            self.model = Gaussian(self.lightcurve.cts)
        elif self.modelType == 'expdecay':
            from ..models import Expdecay
            self.model = Expdecay(self.lightcurve.cts)
        elif self.modelType == 'impulse':
            from ..models import Impulse
            self.model = Impulse(self.lightcurve.cts)
        elif self.modelType == 'step':
            from ..models import Step
            self.model = Step(self.lightcurve.cts)

        self.paramNames = self.model.paramnames
        self.prior = self.model.prior

    def set_grid(self, ranges={}):
        """
        Set the parameter grid on which parameter estimation will be performed. E.g. for the flare
        model:

        >>> ranges = {'t0': (0., 3600., 5), 'taugauss': (0., 5400., 10), 'tauexp': (0., 7200., 10)),
        >>> ... 'amp': (0., 100., 10)}

        Parameters
        ----------
        ranges : dict
            A dictionary of ranges for each parameter. The ranges in general should be a tuple
            containing the lower and upper value of the range range, and number of grid points, but
            it can be a single value.
        """

        if len(ranges) == 0:
            print "Must specify a dictionary of ranges"
            return

        # set ranges in model if nothing is already set
        if len(self.model.ranges) == 0:
            self.model.set_params(ranges)

        # create vectors for each model parameter
        for i, p in enumerate(self.paramNames):
            # check item is a named parameter
            try:
                irange = ranges[p]
            except:
                print "Error. The parameter %s is not in the dictionary" % p
                return

            if not isinstance(irange, tuple):
                irange = (irange,) # convert to tuple if just a value

            if len(irange) == 3:
                if irange[0] < irange[1] and irange[2] > 1:
                    vals = np.linspace(irange[0], irange[1], int(irange[2]))
                else:
                    print "%s range has an upper bound smaller than the lower bound! Try again." % item
                    return
            elif len(irange) == 1:
                vals = np.array([irange[0]], dtype='float32')

            self.paramValues[p] = vals

    def lightcurve_chunk(self, centidx, length):
        """
        Extract a short piece of the light curve to perform parameter estimation on (i.e. extract
        just a couple of days of data centred around a flare that has been found).

        Parameters
        ----------
        centidx : int
            The index of the original light curve to be used as the centre of the extracted chunk.
        length : int
            The length of data to extract (as a number of time bins).
        """
        ll = len(self.lightcurve.cts)

        dl = int(length/2)
        startidx = centidx - dl
        endidx = centidx + dl + 1

        # make sure chunk is within the data
        if startidx < 0:
            startidx = 0

        if endidx > ll-1:
            endidx = ll

        # just set the lightcurve to the required chunk
        self.lightcurve.cts = self.lightcurve.cts[startidx:endidx]
        self.lightcurve.clc = self.lightcurve.clc[startidx:endidx]
        self.lightcurve.cle = self.lightcurve.cle[startidx:endidx]

    def lightcurve_dynamic_range(self):
        """
        Get the dynamic range of the light curve i.e. the difference between the maximum and minimum
        values. This can be useful for setting the range of the amplitude parameter required in e.g.
        the flare model.

        Returns
        -------
        dr : float
            The dynamic range value.
        """

        return (np.amin(self.lightcurve.clc), np.amax(self.lightcurve.clc))

    def set_sigma(self,
                  lightcurve,
                  detrend=True,
                  dtlen=55,
                  dtorder=4,
                  noiseestmethod='powerspectrum',
                  estfrac=0.5,
                  tvsigma=1.0):
        """
        Calculate and set the noise standard deviation to be used in the parameter estimation. This
        uses the whole of the input light curve for the calculation. The calculation can either use
        the 'powerspectrum' method from :func:`.estimate_noise_ps`, the 'tailveto' method from
        :func:`.estimate_noise_tv`, or just the standard calculation used by
        :func:`numpy.std`.

        Parameters
        ----------
        lightcurve : :class:`.Lightcurve`
            The lightcurve for which to calculate the noise standard devaition.
        detrend : boolean, default: True
            If 'True' the light curve will be detrended before noise estimation using the method in
            :meth:`.Lightcurve.detrend`.
        dtlen : int, default: 55
            The running window length for detrending the data.
        dtorder : int, default: 4
            The polynomial order for detrending the data.
        noiseestmethod : string, default: 'powerspectrum'
            The method used for estimating the noise ('powerspectrum' or 'tailveto').
        estfrac : float, default: 0.5
            The fraction of the power spectrum used in the 'powerspectrum' method.
        tvsigma : float, default: 1.0
            The 'standard deviation' probability volume used in the 'tailveto' method.
        """
        tmpcurve = copy(lightcurve)
        if detrend:
            tmpcurve.detrend(method='savitzkygolay', nbins=dtlen, order=dtorder)

        if noiseestmethod == 'powerspectrum':
            sigma = estimate_noise_ps(tmpcurve, estfrac=estfrac)[0]
        elif noiseestmethod == 'tailveto':
            sigma = estimate_noise_tv(tmpcurve.clc, sigma=tvsigma)[0]
        elif noiseestmethod == 'std':
            sigma = np.std(tmpcurve.clc)
        else:
          print "Noise estimation method must be 'powerspectrum' or 'tailveto'"

        del tmpcurve

        self.noiseSigma = sigma

    def calculate_posterior(self, paramValues=None, lightcurve=None, sigma=None, margbackground=True, bgorder=4, ncpus=None):
        """
        Calculate the unnormalised log posterior probability distribution function over the grid of
        parameters assuming a Gaussian likelihood function. If requiring that a background
        polynomial variation is present and to be marginalised over then this function will use
        :func:`.log_likelihood_marg_background` for the likelihood calculation, otherwise it
        will use :func:`.log_likelihood_ratio`.

        Unless the below input parameters are specified the values defined already int the class
        are used.

        Parameters
        ----------
        paramValues : dict
            A user specified dictionary containing the parameter values that can be used instead of
            the one defined in the class.
        lightcurve : :class:`.Lightcurve`
            A user specified lightcurve that can be used instead of the one defined in the class.
        sigma : float
            A user specified data noise standard deviation that can be used instead of the one
            defined by the class.
        margbackground : boolean, default: True
            If true then marginalise over a fit to a polynomial background, otherwise assume no
            polynomial background.
        bgorder : int, default: 4
            The order of the polynomial background fit (the length of the polynomial model will be
            the same as the light curve length).
        ncpus : int, default: None
            The number of parallel CPUs to use with :mod:`multiprocessing`. The default of None
            means all available CPUs will be used.
        """

        if paramValues is not None:
            pv = paramValues

            # check pv contains values for each name
            for p in self.paramNames:
                try:
                    item = pv[p]
                except:
                    print "Parameter %s is not in the supplied paramValues" % p
                    return None

            if len(pv) != len(self.paramNames):
                print "Input parameter values dictionary is not the right length!"
                return None
        else:
            pv = self.paramValues

        if lightcurve is not None:
            lc = lightcurve
        else:
            lc = self.lightcurve

        if sigma is not None:
            sk = sigma
        else:
            sk = self.noiseSigma

        dl = len(lc.cts) # data length

        sp = []
        for p in self.paramNames:
            sp.append(len(pv[p]))

        l = np.product(sp) # total size of parameter space

        # calculate posterior if marginalising over a polynomial background
        if margbackground:
            npoly = bgorder+1 # number of polynomial coefficients

            # set up polynomial models
            # create array of cross-model terms for each set of parameters
            polyms = np.ndarray((npoly, dl))  # background models
            dmcross = np.ndarray((l,npoly+1)) # data cross with the models
            subdm = np.zeros(npoly+1)
            ts = np.linspace(0, 1, dl)
            for i in range(npoly):
                polyms[i] = ts**i
                subdm[i] = np.sum(polyms[i]*lc.clc) # just for polynomial background
            dmcross[:] = subdm

            # cross terms for all model components
            mmcross = np.zeros((l, npoly+1, npoly+1))
            submm = np.zeros((npoly+1, npoly+1))
            for i in range(npoly): # just for background polynomials here
                for j in range(i, npoly):
                    submm[i,j] = np.sum(polyms[i]*polyms[j])
            mmcross[:] = submm

            priorval = np.zeros(l)

        # get size of posterior
        posterior = -np.inf*np.ones(tuple(sp)) # initialise to -inf log likelihood

        # loop over grid
        for idx in range(l):
            q = np.unravel_index(idx, tuple(sp))

            # get parameter values
            ps = {}
            for i, p in enumerate(self.paramNames):
                ps[p] = pv[p][q[i]]

            # get model by inputting dictionary of parameters
            m = self.model.model(ps, ts=lc.cts)

            # check if lightcurve has been detrended
            if lc.detrended and lc.detrend_method=='savitzkygolay':
                # do the same detrending to the model
                mfit = savitzky_golay(m, lc.detrend_nbins, lc.detrend_order)
                m = m-mfit

            if margbackground:
                # get the "model" and background cross terms
                for j in range(npoly):
                    mmcross[idx, j,npoly] = np.sum(m*polyms[j])

                # get the "model" crossed with itself
                mmcross[idx,npoly,npoly] = np.sum(m**2)

                # data crossed with model
                dmcross[idx,-1] = np.sum(lc.clc*m)

                priorval[idx] = self.prior(ps)
            else:
                posterior[q] = log_likelihood_ratio(m, lc.clc, self.noiseSigma) + self.prior(ps)

        if margbackground:
            # use parallel processors to get likelihood
            pool = Pool(processes=ncpus)
            Ms = pool.map_async(log_likelihood_marg_background_wrapper,
                                ((mmcross[i], dmcross[i], npoly+1, self.noiseSigma)
                                for i in range(l))).get()
            # clean-up
            pool.close()
            pool.join()

            # set posteriors
            for idx in range(l):
                q = np.unravel_index(idx, tuple(sp))
                posterior[q] = Ms[idx] + priorval[idx]

        self.posterior = np.copy(posterior) # set posterior

    def marginalised_posterior(self, parameter=None):
        """
        Calculate the posterior for the given parameter marginalised over the other parameters.
        This normalises and exponentiates the log posterior held by the class.

        Parameters
        ----------
        parameter : string
           The parameter posterior to be left after marginalisation over other parameters.

        Returns
        -------
        margp : :class:`numpy.array`
            A 1D array containing the normalised posterior for the given parameter.
        """

        if parameter not in self.paramNames:
            print "Given parameter (%s) is not in model" % parameter
            return None

        if self.posterior == None:
            print "Posterior not yet defined!"
            return None

        # get dimension of parameter
        idx = 0
        shortpars = []
        for i, p in enumerate(self.paramNames):
            if parameter.lower() == p:
                idx = i
            else:
                shortpars.append(p) # list of parameters to be marginalised

        nump = len(self.paramNames) # number of parameters
        pv = self.paramValues     # parameter grids

        # marginalise over other parameters
        margp = np.copy(self.posterior) # get temporary copy of posterior

        # move the required axis to the end of the array
        if idx < nump-1:
            margp = np.rollaxis(margp, idx, nump)

        sp = margp.shape
        for i, p in enumerate(shortpars):
            # check if axis is singleton
            if sp[0] == 1:
                # don't need to marginalise
                margp = margp[0]
            else:
                margp = np.apply_along_axis(logtrapz, 0, margp, pv[p])

            sp = margp.shape

        # normalise and exponentiate the posterior
        if len(margp) > 1:
          area = logtrapz(margp, pv[parameter.lower()])
        else:
          area = 1.
        margp = np.exp(margp-area)

        self.margposteriors[parameter.lower()] = np.copy(margp)

        return margp

    def marginalise_all(self):
        """
        Calculate the marginalised posterior for each of the parameters in the model in turn.
        """

        for p in self.paramNames:
            self.marginalised_posterior(p)

    def marginalised_posterior_2D(self, parameters=None):
        """
        Calculate the 2D posterior for the two given parameters marginalised over the other parameters.

        Parameters
        ----------
        parameter : list
            A list containing the two model parameters to be left after marginalisation.

        Returns
        -------
        marg2d : :class:`numpy.ndarray`
            A 2D array containing the normalised posterior.
        """

        if parameters == None:
            print "Must supply a list of two parameters"
            return None
        elif not isinstance(parameters, list):
            print "Must supply a list of two parameters"
            return None
        elif len(parameters) != 2:
            print "Must supply a list of two parameters"
            return None

        for p in parameters:
            if p.lower() not in self.paramNames:
                print "Given parameter (%s) is not in model" % p
                return None

        if self.posterior == None:
            print "Posterior not yet defined!"
            return None

        nump = len(self.paramNames) # number of parameters
        pv = self.paramValues     # parameter grids

        if nump < 3:
            print "No need to marginalise, posterior is already 2d or 1d"
            return None

        # get indices of two parameters
        idx1 = idx2 = 0
        shortpars = []
        for i, p in enumerate(self.paramNames):
            if parameters[0].lower() == p:
                idx1 = i
            elif parameters[1].lower() == p:
                idx2 = i
            else:
                shortpars.append(p)

        # marginalise over other parameters
        margp = np.copy(self.posterior) # get temporary copy of posterior

        # move the required axes to the end of the array
        if idx2 < nump-1:
            margp = np.rollaxis(margp, idx2, nump)
            if idx2 < idx1:
                idx1 = idx1-1

        if idx1 < nump-2:
            margp = np.rollaxis(margp, idx1, nump-1)

        sp = margp.shape
        for i, p in enumerate(shortpars):
            # check if axis is singleton
            if sp[0] == 1:
                # don't need to marginalise
                margp = margp[0]
            else:
                margp = np.apply_along_axis(logtrapz, 0, margp, pv[p])

            sp = margp.shape

        # get the volume of the posterior
        margparea = np.copy(margp)
        margparea = np.apply_along_axis(logtrapz, 0, margparea, pv[parameters[0].lower()])
        margparea = logtrapz(margparea, pv[parameters[1].lower()])

        return np.exp(margp-margparea) # return two dimensional posterior (normalised and exponentiated)

    def maximum_posterior(self):
        """
        Find the maximum log(posterior) value and the model parameters at that value

        Returns
        -------
        maxpost : float
            The maximum log posterior value
        maxparams : dict
            A dictionary of parameters at the maximum posterior value.
        """

        if self.posterior == None:
            print "Posterior not defined"
            return None

        # get index of maximum of posterior
        i = self.posterior.argmax() # returns index for flattened array
        q = np.unravel_index(i, self.posterior.shape) # convert into indices for each dimension

        self.maxposterior = self.posterior[q]
        for j, p in enumerate(self.paramNames):
            self.maxpostparams[p] = self.paramValues[p][q[j]]

        return self.maxposterior, self.maxpostparams

    def maximum_posterior_snr(self):
        """
        Calculate an estimate of the signal-to-noise ratio for the model matching that at the
        maximum posterior value.

        The signal-to-noise ratio is calculated as:

        .. math::
            \\rho = \\frac{1}{\\sigma}\\sqrt{\\sum_i m_i^2},

        where :math:`\sigma` is the noise standard deviation and :math:`m` is the model evaluated
        at the maximum posterior parameters.

        Returns
        -------
        snr : float
            An estimate of the signal-to-noise ratio.
        """

        if self.maxposterior == None:
            # get maximum posterior
            self.maximum_posterior()

        # get model for maximum posterior values
        m = self.model.model(self.maxpostparams, ts=self.lightcurve.cts)

        snr = np.sqrt(np.sum(m**2))/self.noiseSigma

        return snr

    def maximum_posterior_ew(self):
        """
        Calculate an estimate of the equivalent width (integral of the signal over the underlying
        noise level) of the signal for the model matching that at the maximum posterior value.


        We define the equivalent width as:

        .. math::
           EW = \\frac{1}{b}\\int m \ {\\textrm d}t,

        where :math:`b` is the underlying noise floor level (calculated by subtracting the best fit
        model from the light curve, re-adding any previously subtracted DC offset, and getting the
        median value) and :math:`m` is the model evaluated at the maximum posterior values. The
        integral is calculated using the trapezium rule.

        Returns
        -------
        ew : float
            An estimate of the equivalent width.
        """

        if self.maxposterior == None:
            # get maximum posterior
            self.maximum_posterior()

        # get model for maximum posterior values
        m = self.model.model(self.maxpostparams, ts=self.lightcurve.cts)

        # integrate model and divide underlying noise level (as estimated by
        # subtracting the best fit model from the data and taking the median value)
        ew = np.trapz(m, self.lightcurve.cts)/np.median(self.lightcurve.clc-m+self.lightcurve.dc)

        return ew

    def confidence_interval(self, parameter=None, ci=0.95, upperlimit=False):
        """
        Calculate the confidence interval bounds for the shortest range spanning the given amount of
        probability. Alternatively, if upper limit is True, return the upper bound containing the
        given probability with a lower bound of zero.

        Parameters
        ----------
        parameter : string
            The parameter for which to calculate the interval.
        ci : float, default: 0.95
            The confidence interval probability volume.
        upperlimit : bool, default: False
            Set to True if requiring a lower bound of zero.

        Returns
        -------
        bounds : tuple
            A tuple containing the lower and upper bounds as floats.
        """

        if parameter == None:
            print "Must provide a parameter"
            return None

        if parameter.lower() not in self.paramNames:
            print "Given parameter (%s) is not in model" % parameter
            return None

        # check marginalised posteriors have been set
        if len(self.margposteriors) == 0:
            print "No marginalised posteriors have been set"
            return None

        try:
            post = self.margposteriors[parameter.lower()]
        except:
            print "Marginalised posterior does not exist for %s" % parameter
            return None

        try:
            pvals = self.paramVals[parameter.lower()]
        except:
            print "Parameter grid does not exist for %s" % parameter
            return None

        # get cumulative probability distribution
        cp = np.cumsum(post)
        cp = cp/np.amax(cp) # normalise to have maximum of 1

        # get unique values of cumulative probability
        cpu, ui = np.unique(cp, return_index=True)
        intf = interp1d(cpu, pvals[ui], kind='linear') # interpolation function

        idx = 0
        span = np.inf
        while True:
            lbound = pvals[ui[idx]]

            ubound = intf(ci, cpu[idx:])

            if upperlimit:
                bounds = (0, ubound)
                break

            # check whether bound is shorter than previously
            if ubound - lbound < span:
                span = ubound - lbound
                bounds = (lbound, ubound)

            # subtract the current cumulative probability from them all
            cpu = cpu - cpu[idx]
            idx = idx+1

            # exit once all the value confidence intervals have been tested
            if cpu[-1] < ci:
                break

        return bounds

    def clear(self):
        """ Clear memory """
        del self.paramMesh
        del self.paramValues
        del self.lightcurve
        del self.paramNames
        del self.model
        del self.modelType
        del self.noiseSigma
