from flare import ModelCurve
import numpy as np
from math import *
import bayesflare as pf

class Gaussian():
    """
    A generic Gaussian model.
    """

    amp = 0
    ts  = []
    t0  = 0
    f   = []
    sigma = np.linspace(1, 7, 6)*3600 # standard deviation of Gaussian dip/rise profile

    parameters = []
    ranges = [sigma]

    priors = []

    shape = [len(sigma)]

    def __init__(self, ts, amp=1, t0=None):
        """
        Creates a Gaussian model
        amp                        -- initial amplitude
        ts                         -- a vector containing times
        t0                         -- time of peak

        Returns
        f  --  an array containing the Gaussian model
        """

        if t0 == None:
            t0 = ts[floor(len(ts)/2)]

        self.amp = amp
        self.t0  = t0
        self.ts  = ts
        self.parameters_refresh()
        self.update_priors()

    def __str__(self):
        return "<pyFlare Gaussian model containing "+str(len(self.sigma))+" variants>"

    def __repr__(self):
        return self.__str__()

    def identity_string(self):
        return "gaussian_sigma_"+str(self.ranges[0][0])+"_"+str(self.ranges[0][-1])

    def identity_type(self):
        return "gaussian"

    def set_sigma(self, low, high, number=None):
        if number==None:
            number = floor(high - low)/3600
        self.sigma = np.linspace(low, high, number)
        self.parameters_refresh()
        self.ranges = [self.sigma]
        self.shape = [len(self.sigma)]
        self.update_priors()

    def update_priors(self):
        # only use parameter space for which tauf + 2*sigmagcutoff*sigmag < maxduration
        # sigmagcutoff is the number of sigmag's to use in the width of the transit within
        # the maxduration range
        self.lnsigmaprior = -np.log(self.sigma[-1] - self.sigma[0])

        self.priors = [self.lnsigmaprior]

    def parameters_refresh(self):
        self.parameters = np.zeros(len(self.sigma), dtype=list)
        for i in np.arange(len(self.sigma)):
            self.parameters[i] = [self.sigma[i]]

    def model(self, amp, sigma, t0, ts):
        # the Gaussian model for given parameters
        if sigma == 0: # if sigma is 0 then have delta function at point closest to t0
            f = np.zeros(len(ts))
            tm0 = ts-t0
            f[np.amin(tm0) == tm0] = amp
        else:
            f = amp*np.exp(-(ts - t0)**2/(2*float(sigma)**2))

        return f

    def filter_model(self, m, nbins=101, order=3):
      """ use the Savitzky-Golay smoothing to high-pass filter the model m:
            -- nbins - an odd integer width (in bins) for the filtering
            -- order - the polynomial order for the filtering
      """
      return (m - pf.savitzky_golay(m, nbins, order))

    def output_model(self, i, ts=None, filt=False, nbins=101, order=3):
        if ts == None:
            ts = self.ts

        f = self.model(self.amp, self.sigma[i], self.t0, ts)

        if filt:
            f = self.filter_model(f, nbins, order)

        m = ModelCurve(ts, f)
        return m

    def __call__(self, i, ts=None, filt=False, nbins=101, order=3):
        if ts == None:
            ts = self.ts

        return self.output_model(i , ts, filt, nbins, order)
