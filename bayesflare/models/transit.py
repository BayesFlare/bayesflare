from flare import ModelCurve
import numpy as np
from math import *
import bayesflare as pf

class Transit():
    """
    A transit model with a half-Gaussian dip and rise and flat bottom profile.

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
       An array containing the transit model.
    """

    amp = 0
    ts  = []
    t0  = 0
    f   = []
    sigmag = np.linspace(1, 7, 6)*3600 # standard deviation of Gaussian dip/rise profile
    tauf = np.linspace(0, 7, 7)*3600   # halfwidth of flat bottom of profile

    maxduration = 12*3600 # maximum duration of the transit - 12 hours is default

    parameters = []
    paramnames = ['t0', 'sigmag', 'tauf', 'amp']
    ranges = [sigmag, tauf]

    priors = []

    shape = [len(sigmag), len(tauf)]

    def __init__(self, ts, amp=1, t0=None):
        """
        Creates an transit model with a half-Gaussian dip and rise and a flat bottom
        amp                        -- initial amplitude
        ts                         -- a vector containing times
        t0                         -- time of peak

        Returns
        f  --  an array containing the transit model
        """

        if t0 == None:
            t0 = ts[floor(len(ts)/2)]

        self.amp = amp
        self.t0  = t0
        self.ts  = ts
        self.parameters_refresh()
        self.update_priors()

    def __str__(self):
        return "<pyFlare Transit model containing "+str(len(self.sigmag)*len(self.tauf))+" variants>"

    def __repr__(self):
        return self.__str__()

    def identity_string(self):
        """
        Returns a string which identifies the model.

        Returns
        -------
        str
           A string identifying the model and its parameters.
        """
        return "transit_tauf_"+str(self.ranges[0][0])+"_"+str(self.ranges[0][-1])+"_"+str(len(self.ranges[0]))+"_sigmag_"+str(self.ranges[1][0])+"_"+str(self.ranges[1][-1])+"_"+str(len(self.ranges[1]))

    def identity_type(self):
        """
        Returns the type of model represented by the object.

        Returns
        -------
        str
           A string containing the type of the model the object represents.
        """
        return "transit"

    def set_sigmag(self, low, high=None, number=None):
        """
        Allows the ``sigmag`` parameter of the model to be set.

        Parameters
        ----------
        low : float
           The lowest value of ``sigmag`` to be used.
        high : float, optional
           The highest value of ``sigmag`` to be used.
        number : int, optional
           The number of intervals to be created between the low and high values of ``sigmag``.

        """
        self.sigmag = np.array([low])

        if high != None:
            if number==None:
                number = floor(high - low)/3600
            if low < high and number > 1:
                self.sigmag = np.linspace(low, high, number)
        self.parameters_refresh()
        self.ranges = [self.sigmag, self.tauf]
        self.shape = [len(self.sigmag), len(self.tauf)]
        self.update_priors()

    def set_tauf(self, low, high=None, number=None):
        """
        Allows the ``tauf`` parameter of the model to be set.

        Parameters
        ----------
        low : float
           The lowest value of ``tauf`` to be used.
        high : float, optional
           The highest value of ``tauf`` to be used.
        number : int, optional
           The number of intervals to be created between the low and high values of ``tauf``.

        """
        self.tauf = np.array([low]) # default to low value

        if high != None:
            if number==None:
                number = floor(high - low)/3600
            if low < high and number > 1:
                self.tauf = np.linspace(low, high, number)
        self.parameters_refresh()
        self.ranges = [self.sigmag, self.tauf]
        self.shape = [len(self.sigmag), len(self.tauf)]
        self.update_priors()

    def set_maxduration(self, maxd):
        """
        Sets the ``maxdur`` parameter.

        Parameters
        ----------
        maxd : float
           The maximum duration, in hours.
        
        """
        # enter maxd in hours, but store in seconds

        self.maxduration = maxd * 3600.
        self.update_priors()

    def update_priors(self, sigmagcutoff=2):
        """
        Updates the priors to reflect the models.
        """
        # only use parameter space for which tauf + 2*sigmagcutoff*sigmag < maxduration
        # sigmagcutoff is the number of sigmag's to use in the width of the transit within
        # the maxduration range
        self.lnsigmagprior = 0.
        self.lntaufprior = 0.

        maxtf = self.tauf[-1]
        mintf = self.tauf[0]
        maxsg = self.sigmag[-1]
        minsg = self.sigmag[0]

        self.sigmagcutoff = sigmagcutoff

        # get the sigmag and tauf values at which the above exclusion region intersects the
        # maximum and minimum tauf values
        sigmag1 = ( self.maxduration - mintf ) / ( 2.*sigmagcutoff )
        sigmag2 = ( self.maxduration - maxtf ) / ( 2.*sigmagcutoff )
        tauf1 = self.maxduration - ( 2.*sigmagcutoff ) * minsg
        tauf2 = self.maxduration - ( 2.*sigmagcutoff ) * maxsg
        dsigmag = maxsg - minsg
        dtauf = maxtf - mintf

        self.lntaufprior = 0.

        if sigmag1 < minsg and ( sigmag2 > minsg and sigmag2 <= maxsg ):
            # prior just covers a triangle
            self.lnsigmagprior = -np.log( (sigmag2 - minsg) * (tauf1 - mintf) / 2. )
        elif ( sigmag1 > minsg and sigmag1 <= maxsg ) and ( sigmag2 > minsg and sigmag2 <= maxsg ):
            # prior covers a trapezium
            self.lnsigmagprior = -np.log( (dtauf/2.) * (sigmag2 + sigmag1 - 2.*minsg) )
        elif ( tauf1 > mintf and tauf1 < maxtf ) and ( tauf2 > mintf and tauf2 < maxtf ):
            # prior covers a trapezium
            self.lnsigmagprior = -np.log( (dsigmag/2.) * (tauf2 + tauf1 - 2.*mintf) )
        elif ( sigmag1 > minsg and sigmag1 <= maxsg ) and sigmag2 > maxsg:
            # prior covers a Pentagon
            self.lnsigmagprior = -np.log( dtauf * (sigmag1 - minsg) + ( maxsg - sigmag1 ) *
              ( dtauf + (tauf2 - mintf) ) / 2. )
        else:
            if len(self.sigmag) > 1:
                self.lnsigmagprior = -np.log(self.sigmag[-1] - self.sigmag[0])
            if len(self.tauf) > 1:
                self.lntaufprior = -np.log(self.tauf[-1] - self.tauf[0])

        self.priors = [self.lnsigmagprior, self.lntaufprior]

    def parameters_refresh(self):
        """
        Refreshes the object to reflect new parameters set by ``set_taus_exp`` and ``set_taus_gauss``.
        """
        self.parameters = np.zeros((len(self.sigmag), len(self.tauf)), dtype=list)
        for i in np.arange(len(self.sigmag)):
            for j in np.arange(len(self.tauf)):
               self.parameters[i,j] = [self.sigmag[i],self.tauf[j]]

    def model(self, amp, sigmag, tauf, t0, ts):
        """
        Private method to generate the impulse time series.
        """
        # the transit model for given parameters
        f = -1*amp*np.ones(len(ts))
        #f[ts < t0] = -1*amp*np.exp(-(ts[ts < t0] - (t0))**2/(2*float(sigmag)**2));
        if sigmag > 0:
            f[ts < t0-tauf] = -1*amp*np.exp(-(ts[ts < t0-tauf] - (t0-tauf))**2/(2*float(sigmag)**2))
            f[ts > t0+tauf] = -1*amp*np.exp(-(ts[ts > t0+tauf] - (t0+tauf))**2/(2*float(sigmag)**2))
        else:
            f[ts < t0-tauf] = 0
            f[ts > t0+tauf] = 0
        return f

    def modeldict(self, pdict, ts):
        """
        A method to generate a model given a time vector and a dictionary
        of model parameters.

        Parameters
        ----------
        pdict : dictionary of parameters
           A dictionary containing the parameters of the desired model output.
           The valid names for the parameters are contained within ``transit.paramnames``
        
        """

        # check dictionary is valid
        for p in self.paramnames:
            try:
                pdict[p]
            except:
                print "Input dictionary does not contain the %s parameter"
                return None

        return self.model(pdict['amp'], pdict['sigmag'], pdict['tauf'], pdict['t0'], ts)

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

    def output_model(self, i, j, ts=None, filt=False, nbins=101, order=3):
        """
        A private class to return a model from the object. Should be used with the ``__call__`` process.

        Parameters
        ----------
        i,j : int
           The locations in the ``sigmag`` and ``tauf`` arrays.
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

        # check model is within prior ranges otherwise return None
        sigmag1 = (self.maxduration - self.tauf[-1]) / ( 2.*self.sigmagcutoff )
        sigmag2 = ( self.maxduration - self.tauf[0] ) / ( 2.*self.sigmagcutoff )
        tauftop = self.maxduration - ( 2.*self.sigmagcutoff ) * self.sigmag[i]

        if self.sigmag[i] > sigmag1 and self.sigmag[i] < sigmag2:
            if self.tauf[i] > tauftop:
                return None

        f = self.model(self.amp, self.sigmag[i], self.tauf[j], self.t0, ts)

        if filt:
            f = self.filter_model(f, nbins, order)

        m = ModelCurve(ts, f)
        return m

    def __call__(self, q, ts=None, filt=False, nbins=101, order=3):
        if ts == None:
            ts = self.ts

        i, j = np.unravel_index(q, self.shape)
        return self.output_model(i, j, ts, filt, nbins, order)
