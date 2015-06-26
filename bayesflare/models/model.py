import numpy as np
from math import floor, sqrt
import bayesflare as bf
from ..stats.general import logminus


class Model():
    """
    A class with methods for a generic model.

    Parameters
    ----------
    mtype : string
       The model type, currently this can be 'flare', 'transit', 'expdecay', 'impulse', 'gaussian' or 'step'
    ts : :class:`numpy.ndarray`
       A vector containing time stamps.
    amp : float, optional, default: 1
       The amplitude of the model.
    t0 : float, optional
       The central time of the model. Defaults to the centre of ``ts``.
    reverse : bool, optional, default: False
       A boolean flag. Set this to reverse the model shape.
    paramnames : list of strings
       A list with the names of each model parameter.
    paramranges : dict of tuples
       A dictionary of tuples defining the model parameter ranges.

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

    modelname=None

    def __init__(self, ts, mtype, amp=1, t0=None, reverse=False, paramnames=None, paramranges=None):

        if t0 == None:
            t0 = ts[floor(len(ts)/2)]

        self.mtype = mtype.lower()
        self.paramnames = paramnames
        self.t0 = t0
        self.ts  = ts
        self.reverse = reverse
        self.shape = []
        self.ranges = {}

        # set default ranges
        if paramranges != None:
            self.set_params(paramranges)

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

    def filter_model(self, m, filtermethod='savitzkygolay', nbins=101, order=3, filterknee=(1./(0.3*86400.))):
        """
        Use the Savitzky-Golay smoothing (:func:`.savitzky_golay`) to high-pass filter the model m.

        Parameters
        ----------
        m : :class:`numpy.ndarray`
           An array containing the model.
        filtermethod : string, default: 'savitzkygolay'
           The method for filtering/detrending the model function. The default is
           the Savitzky-Golay method, but this can also be 'runningmedian' to use
           a running median detrending, or 'highpass' for a high-pass 3rd order Butterworth
           filter.
        nbins : int, optional, default: 101
           An odd integer width (in bins) for the Savitzky-Golay, or running median, filtering.
        order : int, optional, default: 3
           The polynomial order for the Savitzky-Golay filtering.
        filterknee : float, default: 1/(0.3*86400) Hz
           The filter knee frequency (in Hz) for the high-pass filter method.

        Returns
        -------
        The filtered model time series

        """
        if filtermethod == 'savitzkygolay':
            return (m - bf.savitzky_golay(m, nbins, order))
        elif filtermethod == 'runningmedian':
            return (m - bf.running_median(m, nbins))
        elif filtermethod == 'highpass':
            ml = bf.Lightcurve()
            ml.clc = np.copy(m)
            ml.cts = np.copy(self.ts)
            filtm = bf.highpass_filter_lightcurve(ml, knee=filterknee)
            del ml
            return filtm
        else:
            raise ValueError('Unrecognised filter method (%s) given' % filtermethod)

    def __call__(self, q, ts=None, filt=False, filtermethod='savitzkygolay', nbins=101, order=3, filterknee=(1./(0.3*86400.))):
        if ts == None:
            ts = self.ts
        #i, j = np.unravel_index(q, self.shape)
        idxtuple = np.unravel_index(q, self.shape)

        pdict = {}
        for i, p in enumerate(self.paramnames):
            pdict[p] = self.ranges[p][idxtuple[i]]

        if ts == None:
            ts = self.ts

        f = self.model(pdict, ts=ts)

        if filt:
            f = self.filter_model(f, filtermethod=filtermethod, nbins=nbins, order=order, filterknee=filterknee)

        m = ModelCurve(ts, f)

        return m


class Flare(Model):
    """
    Creates an exponentially decaying flare model with a Gaussian rise.

    Parameters
    ----------
    ts : :class:`numpy.ndarray`, required
       A 1D array containing the times
    paramranges : dict, default: None
       A dictionary of the flare model parameter ranges. The default has no
       parameter grid set up.
    amp : float, default: 1
       The amplitude of the model
    t0 : float
       The central time of the model
    reverse : bool, default=False
       Reverse the model shape
    paramnames : list, default: ['t0' 'tauexp', 'taugauss', 'amp']
       The names of the flare model parameters

    Examples
    --------
    The flare model could be set up with the following parameter ranges (note
    that for the 't0' parameter a value of inf will just default to the centre of the
    time series):

       >>> ts = np.linspace(0., 30.*86400., 1500) # a time series (in seconds)
       >>> paramranges = { 't0': (np.inf,), ...
       >>>   'tauexp': (0., 10.*3600., 10), ...
       >>>   'taugauss': (0., 10.*3600., 10), ...
       >>>   'amp': (1.,)}
       >>> flare = Flare(ts, paramranges)
    """

    def __init__(self, ts, paramranges=None, amp=1, t0=None, reverse=False):
        Model.__init__(self, ts, mtype='flare', amp=amp, t0=t0, reverse=reverse,
                       paramnames=['t0', 'tauexp', 'taugauss', 'amp'],
                       paramranges=paramranges)

        self.modelname = 'flare'

    def model(self, pdict, ts=None):
        """
        The flare model.

        Parameters
        ----------
        pdict : dict,
           A dictionary of the flare model parameters ('t0', 'amp', 'taugauss', 'tauexp').
        ts : :class:`numpy.ndarray`, default: None
           A 1D set of time stamps (if 'None' the value of ts defined in the model is used).

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

        if ts == None:
            ts = self.ts

        t0 = pdict['t0']
        amp = pdict['amp']
        tauGauss = pdict['taugauss']
        tauExp = pdict['tauexp']

        # if t0 is inf then set it to the centre of the time series
        if t0 == np.inf:
            t0 = ts[int(len(ts)/2.)]

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

    def prior(self, pdict):
        """
        The prior function for the flare model parameters. This is a flat prior
        over the parameter ranges, but with :math:`\\tau_e \geq \\tau_g`.

        Parameters
        ----------
        pdict : dict
           A dictionary of the flare model parameters.


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

        t0range = self.ranges['t0']
        amprange = self.ranges['amp']
        taugrange = self.ranges['taugauss']
        tauerange = self.ranges['tauexp']

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
            # get area

            taugmin = taugrange[0]
            taugmax = taugrange[-1]
            tauemin = tauerange[0]
            tauemax = tauerange[-1]

            dtaug = taugmax-taugmin
            dtaue = tauemax-tauemin

            if taugmin <= tauemin and taugmax <= tauemax:
                # get rectangle area and subtract the lower triangle
                parea = dtaue * dtaug - 0.5*(taugmax-tauemin)**2
            elif taugmin > tauemin and taugmax > tauemax:
                # get upper triangle area
                parea = 0.5*(tauemax-taugmin)**2
            elif taugmin > tauemin and taugmax < tauemax:
                # get upper trapezium area
                parea = 0.5*dtaug*((tauemax-taugmin)+(tauemax-taugmax))
            elif taugmin < tauemin and taugmax > tauemax:
                # get lower trapezium area
                parea = 0.5*dtaue*((tauemin-taugmin)+(tauemax-taugmin))

            tauprior = -np.log(parea)

        return (ampprior + t0prior + tauprior)


class Transit(Model):
    """
    Creates a parameterised transit model with Gaussian wings and a flat bottom.

    Parameters
    ----------
    ts : :class:`numpy.ndarray`, required
       A 1D array containing the times
    paramranges : dict, default: None
       A dictionary of the transit model parameter ranges. The default has no
       parameter grid set up.
    amp : float, default: 1
       The amplitude of the model
    t0 : float
       The central time of the model
    paramnames : list, default: ['t0' 'sigmag', 'tauf', 'amp']
       The names of the transit model parameters

    Examples
    --------
    The transit model could be set up with the following parameter ranges (note
    that for the 't0' parameter a value of inf represents the centre of the
    time series):

       >>> ts = np.linspace(0., 30.*86400., 1500) # a time series (in seconds)
       >>> paramranges = { 't0': (np.inf,), ...
       >>>   'sigmag': (1.*3600., 7.*3600., 10), ...
       >>>   'tauf': (0., 7.*3600., 10), ...
       >>>   'amp': (1.,)}
       >>> transit = Transit(ts, paramranges)
    """

    maxdur = np.inf # the maximum transit duration

    def __init__(self, ts, amp=1, t0=None, paramranges=None):
        Model.__init__(self, ts, mtype='transit', amp=amp, t0=t0,
                       paramnames=['t0', 'sigmag', 'tauf', 'amp'],
                       paramranges=paramranges)

        self.modelname = 'transit'

    def set_max_duration(self, maxdur=np.inf):
        """
        Set the maximum duration of the transit. This will be used when
        constructing the prior to limit the maximum duration of the transit
        model.

        Parameters
        ----------
        maxdur : float, default: infinity
           The maximum transit duration.
        """

        self.maxdur = maxdur

    def model(self, pdict, ts=None):
        """
        The transit model.

        Parameters
        ----------
        pdict : dict,
           A dictionary of the transit model parameters ('t0', 'amp', 'sigmag', 'tauf').
        ts : :class:`numpy.ndarray`, default: None
           A 1D set of time stamps (if 'None' the value of ts defined in the model is used).

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

        if ts == None:
            ts = self.ts

        t0 = pdict['t0']
        amp = pdict['amp']
        sigmag = pdict['sigmag']
        tauf = pdict['tauf']

        # if t0 is inf then set it to the centre of the time series
        if t0 == np.inf:
            t0 = ts[int(len(ts)/2.)]

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

    def prior(self, pdict, sigmacutoff=3.):
        """
        The prior function for the transit model parameters. This is a flat prior
        over the parameter ranges, but with tauf + 2*sigmagcutoff*sigmag < maxduration.

        Parameters
        ----------
        pdict : dict
           A dictionary of the transit model parameters.
        sigmacutoff : float, default: 3.0
           A cut-off number of sigma for the Gaussian wings with which
           to calculate the transit duration.

        Returns
        -------
        The log of the prior function.
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

        t0range = self.ranges['t0']
        amprange = self.ranges['amp']
        sigmagrange = self.ranges['sigmag']
        taufrange = self.ranges['tauf']

        if pdict['tauf'] + 2.*sigmagcutoff*pdict['sigmag'] > self.maxdur:
            return -np.inf

        t0prior = 0.
        if len(t0range) > 1:
            t0prior = -np.log(t0range[-1] - t0range[0])

        ampprior = 0.
        if len(amprange) > 1:
            ampprior = -np.log(amprange[-1] - amprange[0])

        maxtf = taufrange[-1]
        mintf = taufrange[0]
        maxsg = sigmagrange[-1]
        minsg = sigmagrange[0]

        # get the sigmag and tauf values at which the above exclusion region intersects the
        # maximum and minimum tauf values
        sigmag1 = ( self.maxdur - mintf ) / ( 2.*sigmagcutoff )
        sigmag2 = ( self.maxdur - maxtf ) / ( 2.*sigmagcutoff )
        tauf1 = self.maxdur - ( 2.*sigmagcutoff ) * minsg
        tauf2 = self.maxdur - ( 2.*sigmagcutoff ) * maxsg
        dsigmag = maxsg - minsg
        dtauf = maxtf - mintf

        lntaufprior = 0.

        if sigmag1 < minsg and ( sigmag2 > minsg and sigmag2 <= maxsg ):
            # prior just covers a triangle
            lnsigmagprior = -np.log( (sigmag2 - minsg) * (tauf1 - mintf) / 2. )
        elif ( sigmag1 > minsg and sigmag1 <= maxsg ) and ( sigmag2 > minsg and sigmag2 <= maxsg ):
            # prior covers a trapezium
            lnsigmagprior = -np.log( (dtauf/2.) * (sigmag2 + sigmag1 - 2.*minsg) )
        elif ( tauf1 > mintf and tauf1 < maxtf ) and ( tauf2 > mintf and tauf2 < maxtf ):
            # prior covers a trapezium
            lnsigmagprior = -np.log( (dsigmag/2.) * (tauf2 + tauf1 - 2.*mintf) )
        elif ( sigmag1 > minsg and sigmag1 <= maxsg ) and sigmag2 > maxsg:
            # prior covers a Pentagon
            lnsigmagprior = -np.log( dtauf * (sigmag1 - minsg) + ( maxsg - sigmag1 ) *
              ( dtauf + (tauf2 - mintf) ) / 2. )
        else:
            if len(sigmagrange) > 1:
                lnsigmagprior = -np.log(sigmagrange[-1] - sigmagrange[0])
            if len(taufrange) > 1:
                lntaufprior = -np.log(taufrange[-1] - taufranges[0])

        return (ampprior + t0prior + lntaufprior + lnsigmagprior)


class Expdecay(Model):
    """
    Creates an exponential decay model.

    Parameters
    ----------
    ts : :class:`numpy.ndarray`, required
       A 1D array containing the times
    paramranges : dict, default: None
       A dictionary of the exponential decay model parameter ranges. The default has no
       parameter grid set up.
    amp : float, default: 1
       The amplitude of the model
    t0 : float
       The central time of the model
    reverse : bool, default=False
       Reverse the model shape
    paramnames : list, default: ['t0', 'amp', 'tauexp']
       The names of the exponential decay model parameters

    Examples
    --------
    The exponential decay model could be set up with the following parameter ranges (note
    that for the 't0' parameter a value of inf represents the centre of the
    time series):

       >>> ts = np.linspace(0., 30.*86400., 1500) # a time series (in seconds)
       >>> paramranges = { 't0': (np.inf,), ...
       >>>   'tauexp': (0., 2.*3600., 10), ...
       >>>   'amp': (1.,)}
       >>> expdecay = Expdecay(ts, paramranges)
    """

    def __init__(self, ts, amp=1, t0=None, reverse=False, paramranges=None):
        Model.__init__(self, ts, mtype='expdecay', amp=amp, t0=t0, reverse=reverse,
                       paramnames=['t0', 'amp', 'tauexp'],
                       paramranges=paramranges)

        self.modelname = 'expdecay'

    def model(self, pdict, ts=None):
        """
        The exponential decay model.

        Parameters
        ----------
        pdict : dict,
           A dictionary of the exponential decay model parameters ('t0', 'amp', 'tauexp').
        ts : :class:`numpy.ndarray`, default: None
           A 1D set of time stamps (if 'None' the value of ts defined in the model is used).

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

        if ts == None:
            ts = self.ts

        t0 = pdict['t0']
        amp = pdict['amp']
        tauExp = pdict['tauexp']

        # if t0 is inf then set it to the centre of the time series
        if t0 == np.inf:
            t0 = ts[int(len(ts)/2.)]

        f = np.zeros(len(ts))
        f[ts == t0] = amp

        reverse = self.reverse # get reverse (default to False)

        if tauExp > 0:
            if reverse:
                f[ts < t0] = amp*np.exp((ts[ts < t0] - t0)/float(tauExp))
            else:
                f[ts > t0] = amp*np.exp(-(ts[ts > t0] - t0)/float(tauExp))

        return f

    def prior(self, pdict):
        """
        The prior function for the exponential decay model parameters. This is a flat prior
        over the parameter ranges.

        Parameters
        ----------
        pdict : dict
           A dictionary of the transit model parameters.

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

        t0range = self.ranges['t0']
        amprange = self.ranges['amp']
        tauexprange = self.ranges['tauexp']

        t0prior = 0.
        if len(t0range) > 1:
            t0prior = -np.log(t0range[-1] - t0range[0])

        ampprior = 0.
        if len(amprange) > 1:
            ampprior = -np.log(amprange[-1] - amprange[0])

        tauexpprior = 0.
        if len(tauexprange) > 1:
            tauexpprior = -np.log(tauexprange[-1] - tauexprange[0])

        return (t0prior + ampprior + tauexpprior)


class Impulse(Model):
    """
    Creates a delta-function impulse model.

    Parameters
    ----------
    ts : :class:`numpy.ndarray`, required
       A 1D array containing the times
    paramranges : dict, default: None
       A dictionary of the delta-function model parameter ranges. The default has no
       parameter grid set up.
    amp : float, default: 1
       The amplitude of the model
    t0 : float
       The central time of the model
    reverse : bool, default: False
       Reverse the model shape
    paramnames : list, default: ['t0', 'amp']
       The names of the delta-function model parameters

    Examples
    --------
    The delta-function impulse model could be set up with the following parameter ranges (note
    that for the 't0' parameter a value of inf represents the centre of the
    time series):

       >>> ts = np.linspace(0., 30.*86400., 1500) # a time series (in seconds)
       >>> paramranges = { 't0': (np.inf,), ...
       >>>   'amp': (1.,)}
       >>> impulse = Impulse(ts, paramranges)
    """

    def __init__(self, ts, amp=1, t0=None, paramranges=None):
        Model.__init__(self, ts, mtype='impulse', amp=amp, t0=t0,
                       paramnames=['t0', 'amp'],
                       paramranges=paramranges)

        self.modelname = 'impulse'

    def model(self, pdict, ts=None):
        """
        The impulse model.

        Parameters
        ----------
        pdict : dict,
           A dictionary of the exponential decay model parameters ('t0', 'amp').
        ts : :class:`numpy.ndarray`, default: None
           A 1D set of time stamps (if 'None' the value of ts defined in the model is used).

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

        if ts == None:
            ts = self.ts

        t0 = pdict['t0']
        amp = pdict['amp']

        # if t0 is inf then set it to the centre of the time series otherwise t0 is assumed to
        # be the time from the start of the time series
        if t0 == np.inf:
            t0 = ts[int(len(ts)/2.)]
        else:
            t0 = ts[0]+t0

        # the impulse (delta-function) model
        f = np.zeros_like(ts)

        # find nearest index to t0 and set to amp value
        idx = np.abs(ts-t0).argmin()
        f[idx] = amp
        return f

    def prior(self, pdict):
        """
        The prior function for the impulse model parameters. This is a flat prior
        over the parameter ranges.

        Parameters
        ----------
        pdict : dict
           A dictionary of the impulse model parameters.

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

        t0range = self.ranges['t0']
        amprange = self.ranges['amp']

        t0prior = 0.
        if len(t0range) > 1:
            t0prior = -np.log(t0range[-1] - t0range[0])

        ampprior = 0.
        if len(amprange) > 1:
            ampprior = -np.log(amprange[-1] - amprange[0])

        return (t0prior + ampprior)


class Gaussian(Model):
    """
    Creates a Gaussian profile model.

    Parameters
    ----------
    ts : :class:`numpy.ndarray`, required
       A 1D array containing the times
    paramranges : dict, default: None
       A dictionary of the delta-function model parameter ranges. The default has no
       parameter grid set up.
    amp : float, default: 1
       The amplitude of the model
    t0 : float
       The central time of the model
    paramnames : list, default: ['t0', 'amp', 'sigma']
       The names of the Gaussian model parameters

    Examples
    --------
    The Gaussian profile model could be set up with the following parameter ranges (note
    that for the 't0' parameter a value of inf represents the centre of the
    time series):

       >>> ts = np.linspace(0., 30.*86400., 1500) # a time series (in seconds)
       >>> paramranges = { 't0': (np.inf,), ...
       >>>   'sigma': (0., 3.*3600., 10), ...
       >>>   'amp': (1.,)}
       >>> gaussian = Gaussian(ts, paramranges)
    """

    def __init__(self, ts, amp=1, t0=None, paramranges=None):
        Model.__init__(self, ts, mtype='gaussian', amp=amp, t0=t0,
                       paramnames=['t0', 'sigma', 'amp'],
                       paramranges=paramranges)

        self.modelname = 'gaussian'

    def model(self, pdict, ts=None):
        """
        The Gaussian model.

        Parameters
        ----------
        pdict : dict,
           A dictionary of the Gaussian model parameters ('t0', 'amp', 'sigma').
        ts : :class:`numpy.ndarray`, default: None
           A 1D set of time stamps (if 'None' the value of ts defined in the model is used).

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

        if ts == None:
            ts = self.ts

        t0 = pdict['t0']
        amp = pdict['amp']
        sigma = pdict['sigma']

        # if t0 is inf then set it to the centre of the time series
        if t0 == np.inf:
            t0 = ts[int(len(ts)/2.)]

        # the Gaussian model for given parameters
        if sigma == 0: # if sigma is 0 then have delta function at point closest to t0
            f = np.zeros(len(ts))
            tm0 = ts-t0
            f[np.amin(tm0) == tm0] = amp
        else:
            f = amp*np.exp(-(ts - t0)**2/(2*float(sigma)**2))

        return f

    def prior(self, pdict):
        """
        The prior function for the Gaussian function model parameters. This is a flat prior
        over the parameter ranges.

        Parameters
        ----------
        pdict : dict
           A dictionary of the impulse model parameters.

        Returns
        -------
        The log of the prior function.
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

        t0range = self.ranges['t0']
        amprange = self.ranges['amp']
        sigmarange = self.ranges['sigma']

        t0prior = 0.
        if len(t0range) > 1:
            t0prior = -np.log(t0range[-1] - t0range[0])

        ampprior = 0.
        if len(amprange) > 1:
            ampprior = -np.log(amprange[-1] - amprange[0])

        sigmaprior = 0.
        if len(sigmaprior) > 1:
            sigmaprior = -np.log(sigmaprior[-1] - sigmaprior[0])

        return (t0prior + ampprior + sigmaprior)


class Step(Model):
    """
    Creates a step function model.

    Parameters
    ----------
    ts : :class:`numpy.ndarray`, required
       A 1D array containing the times
    paramranges : dict, default: None
       A dictionary of the step function model parameter ranges. The default has no
       parameter grid set up.
    amp : float, default: 1
       The amplitude of the model
    t0 : float
       The central time of the model
    paramnames : list, default: ['t0', 'amp']
       The names of the delta-function model parameters

    Examples
    --------
    The step function model could be set up with the following parameter ranges (note
    that for the 't0' parameter a value of inf represents the centre of the
    time series):

       >>> ts = np.linspace(0., 30.*86400., 1500) # a time series (in seconds)
       >>> paramranges = { 't0': (np.inf,), ...
       >>>   'amp': (1.,)}
       >>> step = Step(ts, paramranges)
    """

    def __init__(self, ts, amp=1, t0=None, reverse=False, paramranges=None):
        Model.__init__(self, ts, mtype='step', amp=amp, t0=t0,
                       paramnames=['t0', 'amp'],
                       paramranges=paramranges)

        self.modelname = 'step'

    def model(self, pdict, ts=None):
        """
        The step function model. A step from zero to amp.

        Parameters
        ----------
        pdict : dict,
           A dictionary of the exponential decay model parameters ('t0', 'amp').
        ts : :class:`numpy.ndarray`, default: None
           A 1D set of time stamps (if 'None' the value of ts defined in the model is used).

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

        if ts == None:
            ts = self.ts

        t0 = pdict['t0']
        amp = pdict['amp']

        # if t0 is inf then set it to the centre of the time series
        if t0 == np.inf:
            t0 = ts[int(len(ts)/2.)]

        reverse = self.reverse

        # the step model
        f = np.zeros_like(ts)
        f[ts > t0] = amp

        return f

    def prior(self, pdict):
        """
        The prior function for the step function model parameters. This is a flat prior
        over the parameter ranges.

        Parameters
        ----------
        pdict : dict
           A dictionary of the step model parameters.

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

        t0range = self.ranges['t0']
        amprange = self.ranges['amp']

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

