import numpy as np
from math import floor, sqrt
import bayesflare as bf
from ..stats.general import logminus


"""
  The names of the parameters for each allowed model.

  If adding a new model add the required parameter names here. Models should have an
  amplitude parameter, 'amp', and a central time parameter, 't0'.
"""
modelparamnames = {
  'flare':    ['t0' 'tauexp', 'taugauss', 'amp'], # flare model parameters
  'transit':  ['t0', 'sigmag', 'tauf', 'amp'],    # transit model parameters
  'impulse':  ['t0', 'amp'],                      # impulse model parameters
  'expdecay': ['t0', 'tauexp', 'amp'],            # exponential decay parameters
  'gaussian': ['t0', 'sigma', 'amp'],             # Gaussian model parameters
  'step':     ['t0', 'amp']                       # step function model parameters
}

"""
  The default parameter ranges for each model. For each parameter the required ranges should
  be able tuple with the low end of the range,the high end and the number of grid points.
  If the tuple contains a single value then that will be the fixed value of the parameter.
  The t0 parameter defined is centred around the value input when initialising the model
  class.

  If adding a new model add the default parameter ranges here.
"""
modelparamranges = {
   'flare': {
     't0':       (0.,), # default to 0. which sets it to the centre of the time series
     'tauexp':   (0., 10.*3600., 10),
     'taugauss': (0., 10.*3600., 10),
     'amp':      (1.,)
   },
   'transit': {
     't0':       (0.,), # default to 0. which sets it to the centre of the time series
     'sigmag':   (1.*3600., 7.*3600., 10),
     'tauf':     (0., 7.*3600., 10),
     'amp':      (1.,)
   }
   'expdecay': {
     't0':       (0.,),
     'tauexp':   (1.*3600., 10.*3600., 10),
     'amp':      (1.,)
   }
   'impulse':  {
     't0':       (0.,),
     'amp':      (1.,)
   }
   'gaussian': {
     't0':       (0.,),
     'sigma':    (1.*3600., 7.*3600., 6),
     'amp':      (1.,)
   }
   'step':     {
     't0':       (0.,),
     'amp':      (1.,)
   }
}


"""
  A dictionary of model functions.

  If adding a new model add the model function name here.
"""
modelfunctions = {
   'flare':    flaremodel,
   'transit':  transitmodel,
   'expdecay': expdecaymodel,
   'impulse':  impulsemodel,
   'gaussian': gaussianmodel,
   'step': stepmodel
}


"""
  A dictionary of prior functions for each model.

  If adding a new model add the prior function name here.
"""
modelpriors = {
   'flare':    flareprior,
   'transit':  transitprior,
   'expdecay': expdecayprior,
   'impulse':  impulseprior,
   'gaussian': gaussianprior,
   'step':     stepprior
}


class Model():
    """
    A class with methods for a generic model.
    
    Parameters
    ----------
    mtype : string, default: 'flare'
       The model type, currently this can be 'flare', 'transit', 'expdecay', 'impulse', 'gaussian' or 'step'
    ts : :class:`numpy.ndarray`
       A vector containing time stamps.
    amp : float, optional, default: 1
       The amplitude of the model.
    t0 : float, optional
       The central time of the model. Defaults to the centre of ``ts``.
    reverse : bool, optional, default: False
       A boolean flag. Set this to reverse the model shape.

    Returns
    -------
    f : :class:`numpy.ndarray`
       An array containing the flare model.
    """

    amp = 0
    ts  = []
    t0  = None
    f   = []

    parameters = []
    paramnames = [] # names of valid parameters
    ranges = {}

    shape = []

    timeseries = []
    reverse=False

    def __init__(self, ts, mtype='flare', amp=1, t0=None, reverse=False):

        if t0 == None:
            t0 = ts[floor(len(ts)/2)]

        self.mtype = mtype.lower()
        self.paramnames = modelparamnames[self.mtype]
        self.t0 = t0
        self.ts  = ts
        self.reverse = reverse

        # set default ranges
        self.set_params(modelparamranges[self.mtype])
        
    def __str__(self):
        return "<BayesFlare "+self.mtype+" model>"

    def __repr__(self):
        return self.__str__()

    def set_params(self, paramrangedict):
        """
        Set a grid of parameter ranges for the model.
        
        Parameters
        ----------
        paramrangedict : dict
           A dictionary of containing tuples for ranges of each of the parameters
           for the given model.
        
        """
        
        for p in self.paramnames:
            rangetuple = paramrangedict[p]
          
            if len(rangetuple) == 1:
                self.ranges[p] = np.array([rangetuple[0]])
            elif len(rangetuple) == 3:
                self.ranges[p] = np.linspace(rangetuple[0], rangetuple[1], rangetuple[2])
            else:
                raise ValueError("Error... range must either contain 1 or 3 values")
                return
            
            self.shape.append(len(self.ranges[p]))

    def prior(self, pdict):
        """
        Return the prior for a set a parameters.
        
        Parameters
        ----------
        pdict : dict
           A dictionary of the model parameters.
        """
        
        return modelpriors[self.mtype](pdict, self.ranges)

    def parameters_refresh(self):
        """
        Refreshes the object to reflect new parameters set by ``set_taus_exp`` and ``set_taus_gauss``.
        """
        self.parameters = np.zeros((len(self.tausGauss), len(self.tausExp)), dtype=list)
        for i in np.arange(len(self.tausGauss)):
            for j in np.arange(len(self.tausExp)):
               self.parameters[i,j] = [self.tausGauss[i],self.tausExp[j]]

    def model(self, pdict, ts):
        """
        A method to generate a model given a time vector and a dictionary
        of model parameters.

        Parameters
        ----------
        pdict : dictionary of parameters
           A dictionary containing the parameters of the desired model output.
           The valid names for the parameters are contained within ``flare.paramnames``
        ts : :class:`numpy.ndarray`
           A 1D array of times.
        
        """

        return modelfunctions[self.mtype](pdict, ts, reverse=self.reverse)

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
        return (m - bf.savitzky_golay(m, nbins, order))

    def output_model(self, pdict, ts=None, filt=False, nbins=101, order=3):
        """
        A private class to return a model from the object. Should be used with the ``__call__`` process.

        Parameters
        ----------
        pdict : int
           The dictionary of model parameter values.
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
        
        # remove checks like this and instead have it in the prior function
        if (self.tausGauss[i] > self.tausExp[j]):
            return None

        f = self.model(pdict, ts)

        if filt:
            f = self.filter_model(f, nbins, order)

        m = ModelCurve(ts, f)
        return m

    def __call__(self, q, ts=None, filt=False, nbins=101, order=3):
        if ts == None:
            ts = self.ts
        #i, j = np.unravel_index(q, self.shape)
        idxtuple = np.unravel_index(q, self.shape)
        
        pdict = {}
        for i, p in enumerate(paramnames):
            pdict[p] = self.ranges[p][idxtuple[i]]
        
        return self.output_model(pdict, ts, filt, nbins, order)


class Flare(Model):
    """
    Creates an exponentially decaying flare model with a Gaussian rise.
    """

    def __init__(self), ts, amp=1, t0=None, reverse=False):
        Model.__init__(self, ts, mtype='flare', amp=amp, t0=t0, reverse=reverse)

        
class Transit(Model):
    """
    Creates a transit model.
    """

    def __init__(self), ts, amp=1, t0=None, reverse=False):
        Model.__init__(self, ts, mtype='transit', amp=amp, t0=t0, reverse=reverse)


class Expdecay(Model):
    """
    Creates an exponentially decaying model.
    """

    def __init__(self), ts, amp=1, t0=None, reverse=False):
        Model.__init__(self, ts, mtype='transit', amp=amp, t0=t0, reverse=reverse)

        
class Impulse(Model):
    """
    Creates a delta-function impulse model.
    """

    def __init__(self), ts, amp=1, t0=None, reverse=False):
        Model.__init__(self, ts, mtype='impulse', amp=amp, t0=t0, reverse=reverse)


class Gaussian(Model):
    """
    Creates a Gaussian profile model.
    """

    def __init__(self), ts, amp=1, t0=None, reverse=False):
        Model.__init__(self, ts, mtype='gaussian', amp=amp, t0=t0, reverse=reverse)
        
        
class Step(Model):
    """
    Creates a step function model. 
    """

    def __init__(self), ts, amp=1, t0=None, reverse=False):
        Model.__init__(self, ts, mtype='step', amp=amp, t0=t0, reverse=reverse)


def flaremodel(pdict, ts, **kwargs):
    """
    The flare model.
    
    Parameters
    ----------
    pdict : dict,
       A dictionary of the flare model parameters ('t0', 'amp', 'taugauss', 'tauexp').
    ts : :class:`numpy.ndarray`, required
       A 1D time series array for evaluating the model.
    
    Keyword arguments
    -----------------
    reverse : bool, default: False
       Reverse the shape of the model.
       
    Returns
    -------
    f : :class:`numpy.ndarray`
       A 1D time series of values of the model evaluated at the set of parameters.
    """
    
    # check input values
    if not pdict.has_key('t0'):
        raise ValueError("Error... no 't0' value in dictionary!")
    if not pdict.has_key('amp'):
        raise ValueError("Error... no 'amp' value in dictionary!")
    if not pdict.has_key('taugauss'):
        raise ValueError("Error... no 'taugauss' value in dictionary!")
    if not pdict.has_key('tauexp'):
        raise ValueError("Error... no 'tauexp' value in dictionary!")
    
    t0 = pdict['t0']
    amp = pdict['amp']
    tauGauss = pdict['taugauss']
    tauExp = pdict['tauexp']

    f = np.zeros(len(ts))
    f[ts == t0] = amp

    reverse = kwargs.get('reverse', False) # get reverse (default to False)
    
    # avoid division by zero errors
    if tauGauss > 0:
        if reverse:
            f[ts > t0] = amp*np.exp(-(ts[ts > t0] - t0)**2 / (2*float(tauGauss)**2))
        else:
            f[ts < t0] = amp*np.exp(-(ts[ts < t0] - t0)**2 / (2*float(tauGauss)**2))

    if tauExp > 0:
        if reverse:
            f[ts < t0] = amp*np.exp((ts[ts < t0] - t0)/float(tauExp))
        else:
            f[ts > t0] = amp*np.exp(-(ts[ts > t0] - t0)/float(tauExp))

    return f

    
def transitmodel(pdict, ts, **kwargs):
    """
    The transit model.

    Parameters
    ----------
    pdict : dict,
       A dictionary of the transit model parameters ('t0', 'amp', 'sigmag', 'tauf').
    ts : :class:`numpy.ndarray`
       A 1D time series array for evaluating the model.
       
    Returns
    -------
    f : :class:`numpy.ndarray`
       A 1D time series of values of the model evaluated at the set of parameters.
    """
    
    # check input values
    if not pdict.has_key('t0'):
        raise ValueError("Error... no 't0' value in dictionary!")
    if not pdict.has_key('amp'):
        raise ValueError("Error... no 'amp' value in dictionary!")
    if not pdict.has_key('sigmag'):
        raise ValueError("Error... no 'sigmag' value in dictionary!")
    if not pdict.has_key('tauf'):
        raise ValueError("Error... no 'tauf' value in dictionary!")
    
    t0 = pdict['t0']
    amp = pdict['amp']
    sigmag = pdict['sigmag']
    tauf = pdict['tauf']
    
    # the transit model for given parameters
    f = -1.*amp*np.ones(len(ts))
    #f[ts < t0] = -1*amp*np.exp(-(ts[ts < t0] - (t0))**2/(2*float(sigmag)**2));
    if sigmag > 0:
        f[ts < t0-tauf] = -1*amp*np.exp(-(ts[ts < t0-tauf] - (t0-tauf))**2/(2*float(sigmag)**2))
        f[ts > t0+tauf] = -1*amp*np.exp(-(ts[ts > t0+tauf] - (t0+tauf))**2/(2*float(sigmag)**2))
    else:
        f[ts < t0-tauf] = 0
        f[ts > t0+tauf] = 0
    return f
    

def gaussianmodel(pdict, ts, **kwargs):
    """
    The Gaussian model.
    
    Parameters
    ----------
    pdict : dict,
       A dictionary of the Gaussian model parameters ('t0', 'amp', 'sigma').
    ts : :class:`numpy.ndarray`
       A 1D time series array for evaluating the model.
       
    Returns
    -------
    f : :class:`numpy.ndarray`
       A 1D time series of values of the model evaluated at the set of parameters.
    """
    
    # check input values
    if not pdict.has_key('t0'):
        raise ValueError("Error... no 't0' value in dictionary!")
    if not pdict.has_key('amp'):
        raise ValueError("Error... no 'amp' value in dictionary!")
    if not pdict.has_key('sigma'):
        raise ValueError("Error... no 'sigma' value in dictionary!")
    
    t0 = pdict['t0']
    amp = pdict['amp']
    sigma = pdict['sigma']
    
    # the Gaussian model for given parameters
    if sigma == 0: # if sigma is 0 then have delta function at point closest to t0
        f = np.zeros(len(ts))
        tm0 = ts-t0
        f[np.amin(tm0) == tm0] = amp
    else:
        f = amp*np.exp(-(ts - t0)**2/(2*float(sigma)**2))

    return f

    
def expdecaymodel(pdict, ts, **kwargs):
    """
    The exponential decay model.
    
    Parameters
    ----------
    pdict : dict,
       A dictionary of the exponential decay model parameters ('t0', 'amp', 'tauexp').
    ts : :class:`numpy.ndarray`
       A 1D time series array for evaluating the model.
    
    Keyword arguments
    -----------------
    reverse : bool, default: False
       Reverse the shape of the model.
    
    Returns
    -------
    f : :class:`numpy.ndarray`
       A 1D time series of values of the model evaluated at the set of parameters.
    """
    
    # check input values
    if not pdict.has_key('t0'):
        raise ValueError("Error... no 't0' value in dictionary!")
    if not pdict.has_key('amp'):
        raise ValueError("Error... no 'amp' value in dictionary!")
    if not pdict.has_key('tauexp'):
        raise ValueError("Error... no 'tauexp' value in dictionary!")
  
    t0 = pdict['t0']
    amp = pdict['amp']
    tauExp = pdict['tauexp']
  
    f = np.zeros(len(ts))
    f[ts == t0] = amp

    reverse = kwargs.get('reverse', False) # get reverse (default to False)
    
    if tauExp > 0:
        if reverse:
            f[ts < t0] = amp*np.exp((ts[ts < t0] - t0)/float(tauExp))
        else:
            f[ts > t0] = amp*np.exp(-(ts[ts > t0] - t0)/float(tauExp))

    return f

    
def impulsemodel(pdict, ts, **kwargs):
    """
    The impulse model.
    
    Parameters
    ----------
    pdict : dict,
       A dictionary of the exponential decay model parameters ('t0', 'amp').
    ts : :class:`numpy.ndarray`
       A 1D time series array for evaluating the model.
    
    Returns
    -------
    f : :class:`numpy.ndarray`
       A 1D time series of values of the model evaluated at the set of parameters.
    """
    
    # check input values
    if not pdict.has_key('t0'):
        raise ValueError("Error... no 't0' value in dictionary!")
    if not pdict.has_key('amp'):
        raise ValueError("Error... no 'amp' value in dictionary!")
  
    t0 = pdict['t0']
    amp = pdict['amp']
  
    # the impulse (delta-function) model
    f = np.zeros_like(ts)

    # find nearest index to t0 and set to amp value
    idx = np.abs(ts-t0).argmin()
    f[idx] = amp
    return f

    
def stepmodel(pdict, ts, **kwargs):
    """
    The step function model. A step from zero to amp.
    
    Parameters
    ----------
    pdict : dict,
       A dictionary of the exponential decay model parameters ('t0', 'amp').
    ts : :class:`numpy.ndarray`
       A 1D time series array for evaluating the model.

    Keyword arguments
    -----------------
    reverse : bool, default: False
       Reverse the shape of the model.

    Returns
    -------
    f : :class:`numpy.ndarray`
       A 1D time series of values of the model evaluated at the set of parameters.
    """
    
    # check input values
    if not pdict.has_key('t0'):
        raise ValueError("Error... no 't0' value in dictionary!")
    if not pdict.has_key('amp'):
        raise ValueError("Error... no 'amp' value in dictionary!")
  
    t0 = pdict['t0']
    amp = pdict['amp']
  
    reverse = kwargs.get('reverse', False) # get reverse (default to False)
  
    # the impulse (delta-function) model
    f = np.zeros_like(ts)

    if reverse:
        f[ts < t0] = amp
    else:
        f[ts > t0] = amp

    return f


def flareprior(pdict, ranges):
    """
    The prior function for the flare model parameters. This is a flat prior
    over the parameter ranges, but with :math:`\tau_e \geq \tau_g`.
    
    Parameters
    ----------
    pdict : dict
       A dictionary of the flare model parameters.
    ranges : dict
       A dictionary of the parameter ranges.
       
    Returns
    -------
    The log of the prior function.
    """
    
    # check input values
    if not pdict.has_key('t0'):
        raise ValueError("Error... no 't0' value in dictionary!")
    if not pdict.has_key('amp'):
        raise ValueError("Error... no 'amp' value in dictionary!")
    if not pdict.has_key('taugauss'):
        raise ValueError("Error... no 'taugauss' value in dictionary!")
    if not pdict.has_key('tauexp'):
        raise ValueError("Error... no 'tauexp' value in dictionary!")
    
    t0 = pdict['t0']
    amp = pdict['amp']
    tauGauss = pdict['taugauss']
    tauExp = pdict['tauexp']
    
    t0range = ranges['t0']
    amprange = ranges['amp']
    taugrange = ranges['taugauss']
    tauerange = ranges['tauexp']
    
    t0prior = 0.
    if len(t0range) > 1:
        t0prior = -np.log(t0range[-1] - t0range[0])
    
    ampprior = 0.
    if len(amprange) > 1:
        ampprior = -np.log(amprange[-1] - amprange[0])
    
    # we the parameter space for which tauExp > tauGauss
    tauprior = 0.
    
    if tauGauss > tauExp or tauGauss > tauerange[-1]:
        tauprior = -np.inf # set prior to 0
    else:
        if len(tauerange) == 1 and len(taugrange) == 1:
            tauprior = 0
        elif len(tauerange) == 1 and len(taugrange) > 1:
            if tauerange[0] < taugrange[-1]:
                tauprior = -np.log(tauerange[0] - taugrange[-1])
            else:
                tauprior = -np.log(taugrange[-1] - taugrange[0])
        elif len(taugrange) == 1 and len(tauerange) > 1:
            tauprior = -np.log(tauerange[-1] - tauerange[0])
        else:
            area = (tauerange[-1] - tauerange[0])*(taugrange[-1] - taugrange[0])
            
            if 
            
            # check area from transit model
            area 
            
            tauprior = 
    
    
    return (ampprior + t0prior + tauprior) 


def expdecayprior(pdict, ranges):
    """
    The prior function for the exponential decay model parameters. This is a flat prior
    over the parameter ranges.
    
    Parameters
    ----------
    pdict : dict
       A dictionary of the exponential decay model parameters.
    ranges : dict
       A dictionary of the parameter ranges.
       
    Returns
    -------
    The log of the prior function.
    """

    # check input values
    if not pdict.has_key('t0'):
        raise ValueError("Error... no 't0' value in dictionary!")
    if not pdict.has_key('amp'):
        raise ValueError("Error... no 'amp' value in dictionary!")
    if not pdict.has_key('tauexp'):
        raise ValueError("Error... no 'tauexp' value in dictionary!")

    t0 = pdict['t0']
    amp = pdict['amp']
    tauExp = pdict['tauexp']

    t0range = ranges['t0']
    amprange = ranges['amp']
    tauerange = ranges['tauexp']
    
    t0prior = 0.
    if len(t0range) > 1:
        t0prior = -np.log(t0range[-1] - t0range[0])
    
    ampprior = 0.
    if len(amprange) > 1:
        ampprior = -np.log(amprange[-1] - amprange[0])
    
    taueprior = 0.
    if len(tauerange) > 1:
        taueprior = -np.log(tauerange[-1] - tauerange[0])

    return (ampprior + t0prior + taueprior) 


def impulseprior(pdict, ranges):
    """
    The prior function for the impulse model parameters. This is a flat prior
    over the parameter ranges.
    
    Parameters
    ----------
    pdict : dict
       A dictionary of the impulse model parameters.
    ranges : dict
       A dictionary of the parameter ranges.
       
    Returns
    -------
    The log of the prior function.
    """
    
    # check input values
    if not pdict.has_key('t0'):
        raise ValueError("Error... no 't0' value in dictionary!")
    if not pdict.has_key('amp'):
        raise ValueError("Error... no 'amp' value in dictionary!")
    
    t0 = pdict['t0']
    amp = pdict['amp']
    
    t0range = ranges['t0']
    amprange = ranges['amp']
    
    t0prior = 0.
    if len(t0range) > 1:
        t0prior = -np.log(t0range[-1] - t0range[0])
    
    ampprior = 0.
    if len(amprange) > 1:
        ampprior = -np.log(amprange[-1] - amprange[0])
    
    return (t0prior + ampprior)

    
def impulseprior(pdict, ranges):
    """
    The prior function for the step function model parameters. This is a flat prior
    over the parameter ranges.
    
    Parameters
    ----------
    pdict : dict
       A dictionary of the impulse model parameters.
    ranges : dict
       A dictionary of the parameter ranges.
       
    Returns
    -------
    The log of the prior function.
    """
    
    # check input values
    if not pdict.has_key('t0'):
        raise ValueError("Error... no 't0' value in dictionary!")
    if not pdict.has_key('amp'):
        raise ValueError("Error... no 'amp' value in dictionary!")
    
    t0 = pdict['t0']
    amp = pdict['amp']
    
    t0range = ranges['t0']
    amprange = ranges['amp']
    
    t0prior = 0.
    if len(t0range) > 1:
        t0prior = -np.log(t0range[-1] - t0range[0])
    
    ampprior = 0.
    if len(amprange) > 1:
        ampprior = -np.log(amprange[-1] - amprange[0])
    
    return (t0prior + ampprior)
    


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
