import numpy as np
from math import floor, sqrt
import bayesflare as pf
from ..stats.general import logminus

class Flare():
    """
    Creates an exponentially decaying flare model with a Gaussian rise.
    
    Parameters
    ----------
    amp : float, optional, default: 1
       The amplitude of the flare.
    ts : :class:`numpy.ndarray`
       A vector containing time stamps.
    t0 : float, optional
       The time of peak. Defaults to the centre of ``ts``.
    reverse : bool, optional, default: False
       A boolean flag. Set this to reverse the flare shape.

    Returns
    -------
    f : :class:`numpy.ndarray`
       An array containing the flare model.
    """

    amp = 0
    ts  = []
    t0  = 0
    f   = []
    tausGauss = np.linspace(0,10,10)*3600
    tausExp   = np.linspace(1,10,10)*3600

    parameters = []
    paramnames = ['t0', 'taugauss', 'tauexp', 'amp'] # names of valid parameters
    ranges = [tausGauss, tausExp]

    shape = [len(tausGauss), len(tausExp)]

    timeseries = []
    reverse=False

    def __init__(self, ts, amp=1, t0=None, reverse=False):

        if t0 == None:
            t0 = ts[floor(len(ts)/2)]

        self.amp = amp
        self.t0  = t0
        self.ts  = ts
        self.parameters_refresh()
        self.update_priors()
        self.reverse = reverse

    def __str__(self):
        return "<pyFlare Flare model containing "+str(len(self.tausGauss)*len(self.tausExp))+" variants>"

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
        return "flare"

    def identity_string(self):
        """
        Returns a string which identifies the model.

        Returns
        -------
        str
           A string identifying the model and its parameters.
        """
        return "flare_tg_"+str(self.ranges[0][0])+"_"+str(self.ranges[0][-1])+"_"+str(len(self.ranges[0]))+"_te_"+str(self.ranges[1][0])+"_"+str(self.ranges[1][-1])+"_"+str(len(self.ranges[1]))

    def set_taus_gauss(self, low, high=None, number=None):
        """
        Allows the ``tau_gauss`` parameter of the model to be set.

        Parameters
        ----------
        low : float
           The lowest value of ``tau_gauss`` to be used.
        high : float, optional
           The highest value of ``tau_gauss`` to be used.
        number : int, optional
           The number of intervals to be created between the low and high values of ``tau_gauss``.

        """
        self.tausGauss = np.array([low]) # default to just low value

        if high != None:
            if number==None:
                number = floor(high - low)
            if low < high and number > 1:
                self.tausGauss = np.linspace(low, high, number)
        self.parameters_refresh()
        self.ranges = [self.tausGauss, self.tausExp]
        self.shape = [len(self.tausGauss), len(self.tausExp)]
        self.update_priors()

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
        self.shape = [len(self.tausGauss), len(self.tausExp)]
        self.ranges = [self.tausGauss, self.tausExp]
        self.update_priors()

    def update_priors(self):
        """
        Updates the priors to reflect the models.
        """
        # we the parameter space for which tauExp > tauGauss
        self.lntauExpprior = 0.
        self.lntauGaussprior = 0.

        if len(self.tausExp) > 1 and len(self.tausGauss) == 1:
            if self.tausGauss[0] < self.tausExp[0]:
                self.lntauExpprior = -np.log(self.tausExp[-1] - self.tausExp[0])
            elif self.tausGauss[0] > self.tausExp[0] and self.tausGauss[0] < self.tausEx[-1]:
                self.lntauExpprior = -np.log(self.tausExp[-1] - self.tausGauss[0])
            else:
                self.lntauExpprior = -np.inf # tauGauss is > tauExp range

        if len(self.tausGauss) > 1 and len(self.tausExp) == 1:
            if self.tausExp[0] > self.tausGauss[-1]:
                self.lntauGaussprior = -np.log(self.tausGauss[-1] - self.tausGauss[0])
            elif self.tausExp[0] > self.tausGauss[0] and self.tausExp[0] < self.tausGauss[0]:
                self.lntauGaussprior = -np.log(self.tauExp[0] - self.tausGauss[0])
            else:
                self.lntauGaussprior = -np.inf # tauExp is < tauGauss range

        if len(self.tausGauss) > 1 and len(self.tausExp) > 1:
            logarea = logminus(np.log(self.tausExp[-1] - self.tausExp[0]) +
                               np.log(self.tausGauss[-1] - self.tausGauss[0]),
                               2.*np.log(self.tausGauss[-1] - self.tausExp[0]) - np.log(2.))
            # set priors as sqrt of 1/area, so they are recombined properly later
            self.lntauGaussprior = -0.5*logarea
            self.lntauExpprior = -0.5*logarea

        self.priors = [self.lntauGaussprior, self.lntauExpprior]

    def parameters_refresh(self):
        """
        Refreshes the object to reflect new parameters set by ``set_taus_exp`` and ``set_taus_gauss``.
        """
        self.parameters = np.zeros((len(self.tausGauss), len(self.tausExp)), dtype=list)
        for i in np.arange(len(self.tausGauss)):
            for j in np.arange(len(self.tausExp)):
               self.parameters[i,j] = [self.tausGauss[i],self.tausExp[j]]

    def model(self, amp, tauGauss, tauExp, t0, ts):
        """
        Private method to generate the flare time series.
        """
        # the flare model for arbitrary parameters
        f = np.zeros(len(ts))
        f[ts == t0] = amp

        # avoid division by zero errors
        if tauGauss > 0:
            if self.reverse:
                f[ts > t0] = amp*np.exp(-(ts[ts > t0] - t0)**2 / (2*float(tauGauss)**2))
            else:
                f[ts < t0] = amp*np.exp(-(ts[ts < t0] - t0)**2 / (2*float(tauGauss)**2))

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
           The valid names for the parameters are contained within ``flare.paramnames``
        
        """

        # check dictionary is valid
        for p in self.paramnames:
            try:
                pdict[p]
            except:
                print "Input dictionary does not contain the %s parameter"
                return None

        return self.model(pdict['amp'], pdict['taugauss'], pdict['tauexp'], pdict['t0'], ts)

    def filter_model(self, m, nbins=101, order=3):
        """
        Use the Savitzky-Golay smoothing (:func:`.savitzky_golay`) to high-pass filter the model m.

        Parameters
        ----------
        m : :class:`numpy.ndarray`
           An array containing the model.
        nbins : int, optional, default: 101
           An odd integer width (in bins) for the filtering.
        order : int, optional, default: 3
           The polynomial order for the filtering.
        
        """
        return (m - pf.savitzky_golay(m, nbins, order))

    def output_model(self, i, j, ts=None, filt=False, nbins=101, order=3):
        """
        A private class to return a model from the object. Should be used with the ``__call__`` process.

        Parameters
        ----------
        i, j : int
           The locations in the ``tau_gauss`` and ``tau_exp`` arrays.
        ts : :class:`numpy.ndarray`, optional
           The time axis to generate the model on.
           Defaults to the normal object's normal time axis.
        filt : bool, optional, default: False
           Boolean flag to enable filtering on the output model.
        nbins : int, optional, default: 101
           The width, in bins, of the filter.
        order : int, optional, default: 3
           The polynomial order of the filter.
        """
        if ts == None:
            ts = self.ts

        #i, j = np.unravel_index(q, self.shape)
        if (self.tausGauss[i] > self.tausExp[j]):
            return None

        f = self.model(self.amp, self.tausGauss[i], self.tausExp[j], self.t0, ts)

        if filt:
            f = self.filter_model(f, nbins, order)

        m = ModelCurve(ts, f)
        return m

    def __call__(self, q, ts=None, filt=False, nbins=101, order=3):
        if ts == None:
            ts = self.ts
        i, j = np.unravel_index(q, self.shape)
        return self.output_model(i, j, ts, filt, nbins, order)


class ModelCurve():

    def __init__(self, ts, lc):
        self.clc = lc
        self.cts = ts

    def dt(self):
        """
        Calculates the time interval of the time series.

        Returns
        -------
        float
           The time interval.
        """
        return self.cts[1] - self.cts[0]

    def fs(self):
        """
        Calculates the sample frequency of the time series.

        Returns
        -------
        float
           The sample frequency.
        """
        return 1.0/self.dt()
