from bayesflare import Lightcurve
import bayesflare as pf
from math import *
import numpy as np
from random import random
__all__ = ["SimLightcurve", "simulate_single"]

class SimLightcurve(Lightcurve):
    """
    Contains methods to simulate various light curves; builds upon the framework of the :class:`Lightcurve` class.

    Parameters
    ----------
    dt : float, optional
       The time separating each data point, in seconds. Defaults to 1765.55929 seconds; the spacing in Kepler Q1
       long cadence data.
    length : float, optional
       The length of the light curve, in days. Defaults to 33.5 days, the length of a Kepler Q1 light curve.
    sigma : float, optional
       The standard deviation of the noise to be simulated. Defaults to 0.5.
    mean : float, optional
       The mean of the noise to be simulated. Defaults to 1.
    cadence : {'long', 'short'}, optional
       The Kepler cadence which is being simulated. Defaults to 'long'.

    """
    original = []

    def __init__(self, dt=1765.55929,  length=33.5, sigma=0.5, mean=1, cadence='long'):
        self.phi    = random()
        self.frq    = 0.000005*random()
        self.va     = 15*random()
        self.sdt    = dt
        self.length = length
        self.sigma  = sigma
        self.mean   = mean
        self.cadence= cadence
        self.lc     = []
        self.generate_curve()

    def __str__(self):
        return "<BayesFlare Simulated Lightcurve with s="+str(self.sigma)+">"


    def snr(self):
        """
        Attempts to calculate the signal-to-noise ratio of the light curve.

        Returns
        -------
        float
           The estimated signal-to-noise ratio (SNR).

        """
        signalarea = np.sum(self.original)
        noise      = pf.estimate_noise(self)[0]
        return np.sqrt( signalarea**2 / noise**2)

    def generate_curve(self):
         """
         Generates the light curve according to the parameters provided to the initiator.

         """

         hours = 3600
         days = 86400

         x = np.arange(0, self.length, self.sdt)   # create the time stamps
         z = np.zeros_like(x)                # create the data array
         # add low frequency variation
         z   = self.va * np.sin(2*np.pi*self.frq*x + self.phi);

         # Add Gaussian noise
         self.lc.append(pf.addNoise(z, self.mean, self.sigma))
         self.ts.append(x)
         self.le.append(np.zeros_like(x))
         self.original.append(np.zeros_like(x))
         self.combine()
         self.detrend()

    def inject_model(self, model, instance):
        """
        Injects a model into the light curve.

        Parameters
        ----------
        model : BayesFlare Model instance
           An object describing the model to be injected.
        instance : int
           The specific model instance (i.e. combination of parameters) to be injected.

        Returns
        -------
        BayesFlare LightCurve
           The light curve containing an injected model.

        """
        return pf.inject_model(self, model, instance)

    def detrend(self, nbins=101, order=3):
        """
        Detrends the simulated light curve using the Savitsky-Golay filter.

        Parameters
        ----------
        nbins : int
           The number of bins used for the filter window width.
        order : int
           The polynomial order of the filter.

        """

        self.clc = (self.clc - pf.savitzky_golay(self.clc, nbins, order))

def simulate_single( dt = 1765.55929, length=33.5, sigma=0.5, mean=1, amp = 1.5):
    """
    Produce a timeseries of simulated data containing randomly
    generated noise and a single flare.

    Parameters
    ----------
    dt : float, optional
       The sample time of the required data in seconds.
       Default is 1765.55929, the sample time of the quarter 1
       *Kepler* data
    length : float, optional
       The number of days long the required data should be.
       Default is 33.5, the length of the quarter 1 *Kepler*
       data
    sigma : float, optional
       The standard deviation of the noise in the time series.
       Default is 0.5
    mean  : float, optional
       The mean of the noise in the time series.
       The default is 1.

    Returns
    -------
    x : np.ndarray
       An array of times
    z : np.array
       An array containing the time series data, including noise
    o : np.array
       An array containing only the flares, without noise or sinusoidal
       variations.
    n : int
       The number of flares injected.

    See also
    --------
    simulate, simulate_single_chunks

    """

    hours = 3600
    days = 86400
    dt = 1765.55929                     # sample interval (sec)
    x = np.arange(0, length*days, dt)   # create the time stamps
    z = np.zeros_like(x)                # create the data array
    o = np.zeros_like(x)                # create clean data array
    # add low frequency variation
    va  = 15*random()                   # amplitude of low frequency variation
    phi = random()                      # initial phase of low frequency variation
    frq = 0.000005*random()             # frequency of variation (Hz)
    z   = va * np.sin(2*np.pi*frq*x + phi);

    # add random flare
    #amp = 3*sigma                       # amplitude of flare
    tau1= np.floor(random()*10)/2
    tau2= 0
    while tau2 < tau1:
        tau2= np.floor(random()*10)/2
    tau = [tau1*hours, tau2*hours]      # decay consts of flare
    pos = floor(random()*len(x))        # random position
    t0  = x[floor(len(x)/2)]            # position of flare peak
    z  += pf.flare(amp, tau, x, t0)
    o  += pf.flare(amp, tau, x, t0)
    # Add Gaussian noise
    z = pf.addNoise(z, mean, sigma)
    x = x/86400

    ze = np.zeros_like(x)

    a = SimLightcurve()
    a.sigma = sigma
    a.ts.append(x)
    a.lc.append(z)
    a.le.append(ze)
    a.original.append(o)
    a.combine()
    return a
