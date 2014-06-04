from flare import ModelCurve
import numpy as np
from math import floor, sqrt
import bayesflare as pf

class Expdecay():
    """
    Creates an exponentially decaying flare model with a sudden rise.

    Parameters
    ----------
    amp : float, optional
       The amplitude of the flare. Defaults to 1.
    ts : ndarray
       A vector containing time stamps.
    t0 : float, optional
       The time of peak. Defaults to the centre of ``ts``.
    reverse : bool, optional
       A boolean flag. Set this to reverse the flare shape.

    Returns
    -------
    f : ndarray
       An array containing the flare model.
    """

    amp = 0
    ts  = []
    t0  = 0
    f   = []
    tausExp   = np.linspace(1,10,10)*3600

    parameters = []
    paramnames = ['t0', 'tauexp', 'amp'] # names of valid parameters
    ranges = [tausExp]

    shape = [len(tausExp)]

    timeseries = []
    reverse=False

    def __init__(self, ts, amp=1, t0=None, reverse=False):
        """
        Creates an exponentially decaying flare model with a Gaussian rise
        amp                        -- initial amplitude
        ts                         -- a vector containing times
        t0                         -- time of peak
        reverse                    -- set to reverse the flare shape

        Returns
        f  --  an array containing the flare model
        """

        if t0 == None:
            t0 = ts[floor(len(ts)/2)]

        self.amp = amp
        self.t0  = t0
        self.ts  = ts
        self.parameters_refresh()
        self.update_priors()
        self.reverse = reverse

    def __str__(self):
        return "<pyFlare Flare model containing "+str(len(self.tausExp))+" variants>"

    def __repr__(self):
        return self.__str__()

    def identity_type(self):
        """
        Returns the type of model represented by the object.

        Returns
        -------
        str
           A string containing the type of the model the object represents.
        """
        return "expdecay"

    def identity_string(self):
        """
        Returns a string which identifies the model.

        Returns
        -------
        str
           A string identifying the model and its parameters.
        """
        return "expdecay_te"

    def set_taus_exp(self, low, high=None, number=None):
        """
        Allows the ``tau_exp`` parameter of the model to be set.

        Parameters
        ----------
        low : float
           The lowest value of ``tau_exp`` to be used.
        high : float, optional
           The highest value of ``tau_exp`` to be used.
        number : int, optional
           The number of intervals to be created between the low and high values of ``tau_exp``.

        """
        self.tausExp = np.array([low]) # default to just low value

        if high != None:
            if number==None:
                number = floor(high - low)
            if low < high and number > 1:
                self.tausExp = np.linspace(low, high, number)
        self.parameters_refresh()
        self.shape = [len(self.tausExp)]
        self.ranges = [self.tausExp]
        self.update_priors()

    def update_priors(self):
        """
        Updates the priors to reflect the models.
        """
        self.lntauExpprior = 0.

        if len(self.tausExp) > 1:
            self.lntauExpprior = -np.log(self.tausExp[-1] - self.tausExp[0])

        self.priors = [self.lntauExpprior]

    def parameters_refresh(self):
        """
        Refreshes the object to reflect new parameters set by ``set_taus_exp`` and ``set_taus_gauss``.
        """
        self.parameters = np.zeros((len(self.tausExp)), dtype=list)
        for j in np.arange(len(self.tausExp)):
            self.parameters[j] = [self.tausExp[j]]

    def model(self, amp, tauExp, t0, ts):
        """
        Private method to generate the flare time series.
        """
        # the exponential decay model for arbitrary parameters
        f = np.zeros(len(ts))
        f[ts == t0] = amp

        if tauExp > 0:
            if self.reverse:
                f[ts < t0] = amp*np.exp((ts[ts < t0] - t0)/float(tauExp))
            else:
                f[ts > t0] = amp*np.exp(-(ts[ts > t0] - t0)/float(tauExp))

        return f

    def modeldict(self, pdict, ts):
        """
        A method to generate a model given a time vector and a dictionary
        of model parameters.

        Parameters
        ----------
        pdict : dictionary of parameters
           A dictionary containing the parameters of the desired model output.
           The valid names for the parameters are contained within ``expdecay.paramnames``

        Note
        -------
        This should probably be moved to the model factory class in a later version rather than being here.
        
        """

        # check dictionary is valid
        for p in self.paramnames:
            try:
                pdict[p]
            except:
                print "Input dictionary does not contain the %s parameter"
                return None

        return self.model(pdict['amp'], pdict['tauexp'], pdict['t0'], ts)

    def filter_model(self, m, nbins=101, order=3):
        """
        Use the Savitzky-Golay smoothing to high-pass filter the model m:

        Parameters
        ----------
        m : ndarray
           An array containing the model.
        nbins : int, optional
           An odd integer width (in bins) for the filtering.
        order : int, optional
           The polynomial order for the filtering.
        
        """
        return (m - pf.savitzky_golay(m, nbins, order))

    def output_model(self, i, ts=None, filt=False, nbins=101, order=3):
        """
        A private class to return a model from the object. Should be used with the ``__call__`` process.

        Parameters
        ----------
        i, j : int
           The locations in the ``tau_exp`` array.
        ts : ndarray, optional
           The time axis to generate the model on.
           Defaults to the normal object's normal time axis.
        filt : bool, optional
           Boolean flag to enable filtering on the output model.
           Defaults to False.
        nbins : int, optional
           The width, in bins, of the filter. Defaults to 101.
        order : int, optional
           The order of the filter.
        """
        if ts == None:
            ts = self.ts

        #i, j = np.unravel_index(q, self.shape)

        f = self.model(self.amp, self.tausExp[i], self.t0, ts)

        if filt:
            f = self.filter_model(f, nbins, order)

        m = ModelCurve(ts, f)
        return m

    def __call__(self, i, ts=None, filt=False, nbins=101, order=3):
        if ts == None:
            ts = self.ts
        return self.output_model(i, ts, filt, nbins, order)
