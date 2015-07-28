#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyfits
import glob
import numpy as np
import bayesflare as bf
import scipy.signal as signal
from scipy.interpolate import interp1d
import matplotlib.mlab as ml
import matplotlib.pyplot as pl
import copy

__all__ = ["Loader", "Lightcurve"]

class Loader():
    """ An interface for finding Kepler data stored locally given a specific directory structure.

    Parameters
    ----------
    data_root -- string
       A string containing the path to a folder containing the data from the Kepler mission.
       This data should be stored in the following format:
       ::
          | data_root/
          |          |/Q1_public
          |          |/Q2_public
          |          .
          |          .

       This parameter is optional, and can also be loaded from the ``KPLR_ROOT`` environment
       variable.
    """

    def __init__(self, data_root=None):
        self.data_root = data_root
        if data_root is None:
            self.data_root = KPLR_ROOT

    def __str__(self):
        return "<BayesFlare Data Loader (data_root=\"{0}\")>".format(self.data_root)

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__()

    def find(self, kic, short=False, quarters="1-9"):
        """ Finds all local files which correspond to a given star,
        from all available data collection quarters, either from
        short or long cadence data.

        Parameters
        ----------
        kic : int, str
           The Kepler Input Catalogue ID for the star
        short : bool, optional
           A boolean flag which determines whether or not short cadence light curves should be returned.
           Defaults to False.
        quarters : str, optional
           A quarter, or a range of quarters in the format "first-last", for example

              >>> Loader.find(757450, quarters="2-3")

           will retrieve all quarter 2 and 3 data for the star KIC757450.
           By default, retrieves all quarters between 1 and 9.

        Returns
        -------
        list
           A list containing the paths to every qualifying lightcurve.

        Examples
        --------

        >>> Loader.find(757450, short=True, quarters="1")

        Will return the short cadence light curves for KIC757450 in quarter 1.
        """
        kic = str('%09d'%int(kic))
        if short==False:
            return sorted(glob.glob(self.data_root+'/Q['+quarters+']_public/kplr'+kic+'*_llc.fits'))
        else:
            return sorted(glob.glob(self.data_root+'/Q['+quarters+']_public/kplr'+kic+'*.fits'))


class Lightcurve():
    """
    A class designed to handle the day-to-day requirements for
    Kepler light curves, including removing DC offsets.

    Parameters
    ----------
    curve : string
       The file name for a light curves.
    detrend : bool, optional, default: False
       Set whether to detrend the lightcurve.
    detrendmethod: string, optional, default: 'savitzkygolay'
       Set the detrending method. Can be 'savitzkygolay, 'runningmedian',
       or 'highpassfilter'.
    nbins : int, optional, default: 101
       Number of time bins to use for the Savitsky-Golay or running median
       detrending.
    order : int, optional, default: 3
       The polynomial order of the Savitsky-Golay filter.
    knee : float, optional, default: 1./(0.3*86400) Hz
       Knee frequency for detrending with 3rd order Butterworth high-pass filter.
    """

    id = 0   # The KIC number of the star

    clc = np.array([])
    cts = np.array([])
    cle = np.array([])
    datamask = np.array([]) # a mask to give back only finite numbers

    ulc = np.array([])    # Place to store the unsmoothed curve.

    dc  = 0 # Stored DC offset.

    filterfit = np.array([])

    quarter = ""

    detrended = False
    detrend_method = None
    detrend_nbins = 0
    detrend_order = 0
    detrend_knee = 0
    detrend_fit = np.array([])

    running_median_dt = 0
    running_median_fit = np.array([])

    sinusoid_freqs = None # periodogram peak frequencies

    def __init__(self, curve=None, detrend=False, detrendmethod='savitzkygolay', nbins=101, order=3, knee=(1./(0.3*86400))):

        clc = None
        clc = np.array([])
        if curve != None:
            self.add_data(curve=curve, detrend=detrend, detrendmethod=detrendmethod, nbins=nbins, order=order, knee=knee)

    def __str__(self):
        return "<bayesflare Lightcurve for KIC "+str(self.id)+">"

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__()

    def identity_string(self):
        """
        Returns a string which identifies the lightcurve.

        Returns
        -------
        str
           An identifier of the light curve based on its length and cadence.
        """
        return "lc_len_"+str(len(self.clc))+"_cad_"+str(self.cadence)

    def dt(self):
        """
        Calculate the sample separation for the light curve.

        Returns
        -------
        float
           The separation time for data in the light curve.
        """
        return self.cts[1] - self.cts[0]

    def fs(self):
        """
        Calculate the sample frequency of the light curve.

        Returns
        -------
        float
           The sample frequency of the light curve.
        """
        return 1.0 / self.dt()

    def psd(self):
        """
        Calculate the one-sided non-windowed power spectrum of the light curve. This uses the
        :func:`matplotlib.mlab.psd` function for computing the power spectrum, with a single
        non-overlapping FFT.

        Returns
        -------
        sk : array-like
           The Power spectral density of the light curve.
        f  : array-like
           An array of the frequencies.
        """
        l = len(self.clc)

        # get the power spectrum of the lightcurve data
        sk, f = ml.psd(x=self.clc, window=signal.boxcar(l), noverlap=0, NFFT=l, Fs=self.fs(), sides='onesided')

        # return power spectral density and array of frequencies
        return sk, f

    def periodogram(self, oversample=1, flow=None, fhigh=None):
        """
        Calculate the Lomb-Scargle periodogram of the light curve using :func:`scipy.signal.lombscargle`.

        Parameters
        ----------
        oversample : int, optional, default: 1
           Create the periodogram frequency bin number as the length of the light curve
           multiplied by this factor.
        flow : float, optional, default: 'none'
           The low frequency cut-off for the frequency range. If this is 'none' then a value
           is taken as one tenth of the inverse of the total data span.
        fhigh: float, optional, default: 'none'
           The high frequency cut-off for the frequency range. If this is 'none' then the
           Nyquist frequency will be used.
        """

        from scipy.signal import lombscargle

        # set the frequency array
        if flow is None: # set lower frequency for periodogram calculation
            flow = 1./(10.*(self.cts[-1]-self.cts[0]))

        if fhigh is None: # set the upper frequency
            fhigh = 0.5/self.dt() # Nyquist frequency

        if flow > fhigh:
            raise ValueError("[Error] flow is greater than fhigh!")

        freqs = np.linspace(flow, fhigh, len(self.clc)*oversample)
        angfreqs = 2.*np.pi*freqs # Lomb-Scargle uses angular frequencies

        # get periodogram
        pgram = lombscargle(self.cts[np.nonzero(self.clc)].astype(np.float64), self.clc[np.nonzero(self.clc)].astype(np.float64), angfreqs)

        return pgram, freqs

    def add_data(self, curve=None, detrend=False, detrendmethod='none', nbins=101, order=3, knee=None):
        """
        Add light curve data to the object..

        Parameters
        ----------
        curve : string
           The file path file pointing to a light curve fits file.
        detrend : bool, optional, default: False
           A boolean flag which determines whether light curve should be detrended.
        detrendmethod : string, optional, default: 'none'
           The method for detrending the data. The options are 'savitzky_golay' to use
           :func:`.savitzky_golay`, 'runningmedian' to use :func:`.running_median`, or
           'highpassfilter' tp use :func:`.highpass_filter_lightcurve`
        nbins : int, optional, default: 101
           The width of the detrending window in bins of the light curve.
        order : int, optional, default: 3
           The polynomial order of the detrending filter.

        Exceptions
        ----------
        NameError
           This needs to be replaced with an exception specific to the package!
           Error is raised if there is an I/O error while accessing a light curve file.
        """
        if curve == None:
            raise NameError("[Error] No light curve file given")

        try:
            dcurve = pyfits.open(curve)
        except IOError:
            raise NameError("[Error] An IO error occured when trying to access "+curve )
            mis_file = open('ioerror-files.log', 'a')
            mis_file.write(curve+'\n')
            mis_file.close()
            return

        if len(self.clc) == 0:
            self.id = dcurve[0].header['KEPLERID']
        elif dcurve[0].header['KEPLERID'] != self.id:
            raise NameError("Tried to add data from KIC"+str(dcurve[0].header['KEPLERID'])+" to KIC"+str(self.id))

        if dcurve[0].header['OBSMODE'] == 'long cadence':
            self.cadence = 'long'
        else:
            self.cadence = 'short'

        self.quarter = str(self.quarter)+str(dcurve[0].header['QUARTER'])

        # Assemble the new data into the class
        self.clc = np.append(self.clc, copy.deepcopy(dcurve[1].data['PDCSAP_FLUX']))
        self.cts = np.append(self.cts, copy.deepcopy(dcurve[1].data['TIME']*24*3600))
        self.cle = np.append(self.cle, copy.deepcopy(dcurve[1].data['PDCSAP_FLUX_ERR']))

        dcurve.close()
        self.dcoffset() # remove a DC offset (calculated as the median of the light curve)
        self.maskdata()

        if detrend:
            if detrendmethod != 'none':
                self.detrend(detrendmethod, nbins, order)

        del dcurve

    def dcoffset(self):
        """
        Method to remove a DC offset from a light curve by subtracting the median value of the
        light curve from all values.
        """
        self.dc  = np.median(self.clc[np.isfinite(self.clc)])
        self.clc = self.clc - self.dc

    def maskdata(self):
        """
        Set all non-finite values to zero, and their associated errors to infinity.

        Examples
        --------

           >>> camelot = bf.Lightcurve(curves)
           >>> camelot.interpolate()

        """

        self.datamask = ~np.isfinite(self.clc)
        self.clc[self.datamask] = 0.
        #self.cle[self.datamask] = np.inf
        self.cle[self.datamask] = 1e5 # a large, but finite value

        # sort out any timestamps that are screwy!
        tidxs = np.isfinite(self.cts)
        cidxs = np.arange(len(self.cts))
        fint = interp1d(cidxs[tidxs], self.cts[tidxs], kind='linear')
        self.cts[~tidxs] = fint(cidxs[~tidxs])

    def set_detrend(self, method='none', nbins=None, order=None, knee=None):
        """
        A method allowing the detrending parameters for the light curve to be changed.

        Parameters
        ----------
        method : string
           The detrending method. Can be 'savitzkygolay', 'runningmedian', or,
           'highpassfilter'.
        nbins : int
           The length of the detrending window, in bins.
        order : int
           The order of the detrending filter.

        See also
        --------
        detrend

        """
        self.detrend_method=method
        self.detrend_length=nbins
        self.detrend_order=order
        self.detrend_knee=knee

    def detrend(self, method='none', nbins=None, order=None, knee=None):
        """
        A method to detrend the light curve using a Savitsky-Golay filter (:func:`.savitzky_golay`),
        a running median filter (:func:`.running_median`), or a high-pass filter
        (:func:`.highpass_filter_lightcurve`).

        Parameters
        ----------
        method : string, default: 'none'
           The detrending method. Either 'savitzkygolay', 'runningmedian', or
           'highpassfilter'.
        nbins : int, default: None
           The number of bins in the Savitsky-Golay, or running median detrend window
        order : int, default: None
           The polynomial order of the Savitsky-Golay detrending fit.
        knee : float, default: None
           The high-pass filter knee frequency (Hz).

        """

        self.set_detrend(method=method, nbins=nbins, order=order, knee=knee)
        self.detrended = True

        # store un-detrending light curve
        self.ulc = np.copy(self.clc)

        if method == 'savitzkygolay':
            if nbins is None or order is None:
                raise ValueError("Number of bins, or polynomial order, for Savitsky-Golay filter not set")

            ffit = bf.savitzky_golay(self.clc[~self.datamask], nbins, order)
            self.clc[~self.datamask] = (self.clc[~self.datamask] - ffit)
            self.detrend_fit = np.copy(ffit)
        elif method == 'runningmedian':
            if nbins is None:
                raise ValueError("Number of bins for running median filter not set")

            ffit = bf.running_median(self.clc[~self.datamask], nbins)
            self.clc[~self.datamask] = (self.clc[~self.datamask] - ffit)
            self.detrend_fit = np.copy(ffit)
        elif method == 'highpassfilter':
            if knee is None:
                raise ValueError("Knee frequency for high-pass filter not set.")

            dlc = bf.highpass_filter_lightcurve(self, knee=knee)
            self.clc = np.copy(dlc.clc)
        else:
            raise ValueError("No detrend method set")

    def plot(self, figsize=(10,3)):
        """
        Method to produce a plot of the light curve.

        Parameters
        ----------
        figsize : tuple
           The size of the output plot.

        """
        fig, ax = pl.subplots(1)
        pl.title('Lightcurve for KIC'+str(self.id))
        self.trace = ax.plot(self.cts/(24*3600.0), self.clc)
        fig.autofmt_xdate()
        pl.xlabel('Time [days]')
        pl.ylabel('Luminosity')
        pl.show()

