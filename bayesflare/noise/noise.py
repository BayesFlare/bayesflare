import numpy as np
import scipy.signal as signal
from scipy.special import erf
from scipy.interpolate import interp1d
import matplotlib.mlab as ml
from ..misc import nextpow2
from ..simulate import SimLightcurve
from math import factorial

def estimate_noise_ps(lightcurve, estfrac=0.5):
    """
    Use the high frequency part of the power spectrum of a light curve
    to estimate the time domain noise standard deviation of the
    data. This avoids the estimate being contaminated by low-frequency lines
    and flare signals.

    Parameters
    ----------
    lightcurve : :class:`.Lightcurve`
       A :class:`.Lightcurve` instance containing the time series data.
    estfrac : float, optional, default: 0.5
       The fraction of the spectrum (from the high frequency end)
       with which to estimate the noise. The default is 0.5 i.e. use
       the final half of the spectrum for the estimation.

    Returns
    -------
    sqrt(sk) : float
        The noise standard deviation
    sk : float
        The noise variance
    noise_v : :class:`numpy.array`
        A vector of noise variance values
    """
    l = len(lightcurve.clc)
    # get the power spectrum of the lightcurve data
    sk, f = lightcurve.psd()
    # get the mean of the final quarter of the data
    sk = np.mean(sk[np.floor((1.-estfrac)*len(sk)):])
    # scale to give noise variance
    sk = sk * lightcurve.fs() / 2.
    noise_v = np.ones(nextpow2(2*len(lightcurve.clc)-1)) * sk

    return np.sqrt(sk), sk, noise_v


def estimate_noise_tv(d, sigma=1.0):
    """
    A method of estimating the noise, whilst ignoring large outliers.

    This uses the cumulative distribution of the data point and uses the probability
    contained within a Gaussian range (defined by sigma) to work out what the
    standard deviation is (i.e. it doesn't use tails of the distribution that
    contain large outliers, although the larger the sigma value to more outliers
    will effect the result.) This is mainly suitable to data in which the
    underlying noise is Gaussian.

    Parameters
    ----------
    d : array-like
        The time series of data (either a :class:`numpy.array` or a list).
    sigma: float
        The number of standard deviations giving the cumulative probability
        to be included in the noise calculation e.g. if sigma=1 then the central
        68% of the cumulative probability distribution is used.

    Returns
    -------
    std: float
        The noise standard deviation
    mean: float
        The value at the middle of the distribution
    """

    ld = len(d)

    # get normalised histogram
    n, bins = np.histogram(d, bins=ld, density=True)
    bincentres = (bins[:-1] + bins[1:])/2. # bin centres

    # get the cumulative probability distribution
    cs = np.cumsum(n*(bins[1]-bins[0]))

    # get unique values (helps with interpolation)
    csu, idx = np.unique(cs, return_index=True)
    binsu = bincentres[idx]

    # get the cumulative % probability covered by sigma
    cp = erf(sigma/np.sqrt(2.))

    interpf = interp1d(csu, binsu) # interpolation function

    # get the upper and lower interpolated data values that bound the range
    lowS = interpf(0.5 - cp/2.);
    highS = interpf(0.5 + cp/2.);

    # get the value at the middle of the distribution
    m = interpf(0.5);

    # get the standard deviation estimate
    std = (highS - lowS)/(2.*sigma)

    return std, m


def addNoise(z, mean, stdev):
    import random
    """
    Adds gaussian noise to the time series

    Parameters
    ----------
    z : :class:`numpy.ndarray`
       An array containing a time series.
    mean : float
       The mean of the desired noise.
    stdev : float
       The standard deviation of the desired noise.

    Returns
    -------
    z : :class:`numpy.ndarray`
       The input time series with added Gaussian noise.

    """
    z += [random.gauss(mean,stdev) for _ in xrange(len(z))]
    return z

def make_noise_lightcurve(dt = 1765.55929, length=33.5, sigma=0.5, mean=1):
    """
    Produce a time series of gaussian noise, which can be used to
    estimate confidence thresholds.

    Parameters
    ----------
    dt : float, optional
       The sample time of the required data in seconds.
       Default is 1765.55929, the sample time of the quarter 1
       *Kepler* data
    days : float, optional
       The number of days long the required data should be.
       Default is 33.5, the length of the quarter 1 *Kepler*
       data
    sigma : float, optional
       The standard deviation of the noise in the time series.
       Default is 0.5
    mean : float, optional
       The mean of the noise in the time series.
       The default is 1.

    Returns
    -------
    a : :class:`.Lightcurve`
       The generated light curve.

    """

    days = 86400
    dt = 1765.55929                     # sample interval (sec)
    x = np.arange(0, length*days, dt)   # create the time stamps
    z = np.zeros_like(x)                # create the data array
    ze = np.zeros_like(x)
    # Add Gaussian noise
    z = addNoise(z, mean, sigma)
    x = x/86400

    a = SimLightcurve()
    a.sigma = sigma
    a.ts.append(x)
    a.lc.append(z)
    a.le.append(ze)
    a.combine()
    return a


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
      Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
      The Savitzky-Golay filter removes high frequency noise from data.
      It has the advantage of preserving the original shape and
      features of the signal better than other types of filtering
      approaches, such as moving averages techniques. This implementation is
      taken from [3]_.

      Parameters
      ----------
      y : array_like, shape (N,)
          The values of the time history of the signal.
      window_size : int
          The length of the window. Must be an odd integer number.
      order : int
          The order of the polynomial used in the filtering.
          Must be less then `window_size` - 1.
      deriv: int, default: 0
          the order of the derivative to compute (default = 0 means only smoothing)

      Returns
      -------
      ys : :class:`numpy.ndarray`, shape (N)
          the smoothed signal (or it's n-th derivative).

      Notes
      -----
      The Savitzky-Golay is a type of low-pass filter, particularly
      suited for smoothing noisy data. The main idea behind this
      approach is to make for each point a least-square fit with a
      polynomial of high order over a odd-sized window centered at
      the point.

      Examples
      --------
      >>> t = np.linspace(-4, 4, 500)
      >>> y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
      >>> ysg = savitzky_golay(y, window_size=31, order=4)
      >>> import matplotlib.pyplot as plt
      >>> plt.plot(t, y, label='Noisy signal')
      >>> plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
      >>> plt.plot(t, ysg, 'r', label='Filtered signal')
      >>> plt.legend()
      >>> plt.show()

      References
      ----------
      .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
      .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
      .. [3] http://wiki.scipy.org/Cookbook/SavitzkyGolay
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def highpass_filter_lightcurve(lightcurve, knee=(1./(0.3*86400.))):
    """
    Detrends a light curve by high-pass filtering it using a third order Butterworth
    filter (:func:`scipy.signal.butter`).

    Parameters
    -----------
    x : :class:`numpy.ndarray`
       An array of time stamps
    z : :class:`numpy.ndarray`
       An array containing the time series data
    knee : float, optional, default: 3.858e-05
       The high-pass filter knee frequency in Hz (default is 3.858e-05 Hz or (1/0.3)/day).

    Returns
    -------
    z : :class:`numpy.ndarray`
       An array which contains a time series which has been smoothed.

    """

    x  = lightcurve.cts
    z  = lightcurve.clc

    dt = lightcurve.dt()
    if dt <= 0:
        raise(NameError("[ERROR] Sample time of 0 detected. Halting."))
    fs = lightcurve.fs()
    highcut = knee/(1./(2.*dt))
    zr = z[::-1]               # Reverse the timeseries to remove phase offset
    zd = np.concatenate((zr, z))
    b, a = signal.butter(3, highcut, btype='highpass')
    y = signal.lfilter(b, a, zd)
    z = y[np.floor(len(y)/2):]
    return z

    
def running_median(y, window):
    """
    A method to subtract a running median for smoothing data.

    Parameters
    ----------
    y : :class:`numpy.ndarray`
       A 1D array containing the data time series.
    window : int
       The number of time bins to use for the running median. At edges
       the window will be shorter.
       
    Returns
    -------
    ffit : :class:`numpy.ndarray`
       A 1D array containing the running median of the data time series.
    """
    
    ffit = np.array([])
    idxs = np.arange(len(y))
    halfwin = int(window/2)
    
    for i in range(len(y)):
        v = (idxs < (idxs[i]+halfwin)) & (idxs > (idxs[i]-halfwin))
        ffit = np.append(ffit, np.median(y[v]))

    return ffit
