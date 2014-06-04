from flare import ModelCurve
import numpy as np
from math import floor, sqrt
import bayesflare as pf

class Impulse():
    """
    A delta-function impulse model.

    Parameters
    ----------
    amp : float, optional
       The amplitude of the flare. Defaults to 1.
    ts : ndarray
       A vector containing time stamps.
    t0 : float, optional
       The time of peak. Defaults to the centre of ``ts``.

    Returns
    -------
    f : ndarray
       An array containing the flare model.
    """

    amp = 1
    ts  = []
    t0  = 0

    t0s = np.array([1.])

    parameters = []
    paramnames = ['t0'] # names of valid parameters
    ranges = [t0s]

    shape = [len(t0s)]
    priors = [1.]

    def __init__(self, ts, amp=1, t0=None):
        """
        Creates an exponentially decaying flare model with a Gaussian rise
        amp                        -- initial amplitude
        ts                         -- a vector containing times
        t0                         -- time of peak

        Returns
        f  --  an array containing the flare model
        """

        if t0 == None:
            t0 = ts[floor(len(ts)/2)]

        self.amp = amp
        self.t0  = t0
        self.ts  = ts

    def __str__(self):
        return "<pyFlare Impulse model containing "+str(len(self.t0s))+" variants>"

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
        return "impulse"

    def identity_string(self):
        """
        Returns a string which identifies the model.

        Returns
        -------
        str
           A string identifying the model and its parameters.
        """
        return "impulse"

    def set_t0s(self, low, high=None, number=None):
        """
        Allows the ``t0`` parameter of the model to be set.

        Parameters
        ----------
        low : float
           The lowest value of ``t0`` to be used.
        high : float, optional
           The highest value of ``t0`` to be used.
        number : int, optional
           The number of intervals to be created between the low and high values of ``t0``.

        """
        self.t0s = np.array([low])

        if high != None:
            if number==None:
                number = floor(high - low)
            if low < high and number > 1:
                self.t0s = np.linspace(low, high, number)
        self.parameters_refresh()
        self.ranges = [self.t0s]
        self.shape = [len(self.t0s)]
        self.update_priors()

    def parameters_refresh(self):
        """
        Refreshes the object to reflect new parameters set by ``set_taus_exp`` and ``set_taus_gauss``.
        """
        self.parameters = np.zeros((len(self.t0s)), dtype=list)
        for i in np.arange(len(self.t0s)):
            self.parameters[i] = [self.t0s[i]]

    def update_priors(self):
        """
        Updates the priors to reflect the models.
        """
        self.lnt0prior = 0.

        if len(self.t0s) > 1:
            self.lnt0prior = -np.log(self.t0s[-1]  -self.t0s[0])

        self.priors = [self.lnt0prior]

    def model(self, amp, t0, ts):
        """
        Private method to generate the impulse time series.
        """
        # the impulse (delta-function) model
        f = np.zeros_like(ts)

        # find nearest index to t0 and set to amp value
        idx = np.abs(ts-t0).argmin()
        f[idx] = amp
        return f

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
        i : int
           The location in the ``t0`` array.
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

        f = self.model(self.amp, ts[0]+self.t0s[i], ts)

        if filt:
            f = self.filter_model(f, nbins, order)

        m = ModelCurve(ts, f)
        return m

    def __call__(self, i, ts=None, filt=False, nbins=101, order=3):
        if ts == None:
            ts = self.ts

        return self.output_model(i, ts=ts)

