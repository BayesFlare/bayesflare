# -*- coding: utf-8 -*-
import cython
cimport cython

import numpy as np
cimport numpy as np
import scipy

import sys

cdef extern from "math.h":
    double exp(double x)
    double sqrt(double x)
    double log(double x)
    int isinf(double x)
    double fabs(double x)

cdef extern from "gsl/gsl_sf_erf.h":
    double gsl_sf_erf(double x) nogil
    double gsl_sf_log_erfc(double x) nogil

cdef extern from "log_marg_amp_full.h":
    double log_marg_amp_full_C(int Nmodels, double *modelModel, double *dataModel, double sigma, int lastHalfRange)
    double log_marg_amp_except_final_C(int Nmodels, double *modelModel, double *dataModel, double sigma)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef double LOG2PI = 1.837877066409345
cdef double LOGPI_2 = 0.451582705289455
cdef double LOG2 = 0.6931471805599453094172321214581766

cpdef log_one_plus_erf(np.ndarray[DTYPE_t, ndim=1] x):
    """
    Calculate :math:`\log{(1+\\textrm{erf}(x))}`. This uses the numerical approximation to erf
    given in [1]_ or [2]_.

    .. note::
        This was implemented to see if it removed -infinities when using ``x << 0``, but due to the
        ``z*z`` component it does not help. It is kept here for posterity, but does not need to be
        used.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        A 1D array of values.

    Returns
    -------
    lome : double
        The value of :math:`\log{(1+\\textrm{erf}(x))}`.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Error_function#Numerical_approximation
    .. [2] Numerical Recipes in C, pp. 221 (http://apps.nrbook.com/c/index.html)
    """
    cdef int i = 0
    cdef loopmax = len(x)

    cdef np.ndarray lome = np.zeros(x.shape[0], dtype=DTYPE)

    cdef double z = 0.
    cdef double t = 0.
    cdef double logtau = 0.
    cdef double logexp = 0.

    for i in range(loopmax):
        z = fabs(x[i])
        t = 1. / ( 1. + 0.5*z )

        logtau = -logplus(0., log(z)-LOG2)
        logexp = -(z*z) - 1.26551223 + t * ( 1.00002368 + t * ( 0.37409196 + t * ( 0.09678418 +
          t * ( -0.18628806 + t * ( 0.27886807 + t * ( -1.13520398 + t * ( 1.48851587 +
          t * ( -0.82215223 + t * ( 0.17087277 ) ) ) ) ) ) ) ) )

        logtau = logtau + logexp
        if x[i] >= 0.:
            lome[i] = logminus(LOG2, logtau)
        else:
            lome[i] = logtau

    return lome


cpdef log_marg_amp(np.ndarray[DTYPE_t, ndim=1] d, np.ndarray[DTYPE_t, ndim=1] m, ss):
    """
    Calculate the logarithm of the likelihood ratio for the signal model compared to a pure
    Gaussian noise model, but analytically marginalised over the unknown model amplitude. This is
    calculated for each timestep in the data, and assumes a constant noise level over the data.

    As this function uses :func:`numpy.correlate`, which works via FFTs, the data should be
    contiguous and evenly spaced.

    .. note:: Use this instead of :func:`log_marg_amp_original` when required.

    Parameters
    ----------
    d : :class:`numpy.array`
        A 1D array containing the light curve time series data.
    m : :class:`numpy.array`
        A 1D array (of the same length as `d`) containing the model function.
    ss : float or double
        The noise variance of the data (assumed constant)

    Returns
    -------
    B : double
        The logarithm of the likelihood ratio.

    See also
    --------
    log_marg_amp_original : A slower version of this function.

    """

    #get the data/model cross term
    dm = np.correlate(d, m, mode='same')

    # get the model autocorrelation term
    cdef double m2 = np.sum(m*m)

    # get the likelihood marginalised over the signal amplitude
    cdef double inside_erf = sqrt(0.5/(ss*m2))

    # use log of complementary error function from GSL
    cdef double logpart = 0.5*log(np.pi/(2.*m2))

    cdef np.ndarray B = np.zeros(d.shape[0], dtype=DTYPE)
    cdef double logerf = 0.
    cdef double logerfc = 0.
    cdef int i = 0
    cdef int loopmax = len(B)
    cdef double k = 0
    for i in range(loopmax):
        k = dm[i]*inside_erf
        log_erf = log(1. + gsl_sf_erf( k ))
        B[i] = ((dm[i]*dm[i])/(2.*m2*ss)) + logpart + log_erf

    return B


cpdef log_marg_amp_full_model(i, shape, np.ndarray[DTYPE_t, ndim=1] sk, nbackground, lastHalfRange,
                              np.ndarray[DTYPE_t, ndim=1] d,
                              np.ndarray ms, np.ndarray[DTYPE_t, ndim=3] bgcross,
                              np.ndarray mdbgcross, np.ndarray mdcross,
                              np.ndarray[DTYPE_t, ndim=2] dbgr):
    """
    Calculate the logarithm of the likelihood ratio for data consisting of a model (e.g. a flare
    model) *and* modelled background variations compared to Gaussian noise. This function will
    calculate this for a given position in the model parameter value space and for each time step in
    the data. The background model amplitude coefficients will be analytically marginalised
    over. The model amplitude will also be analytically marginalised over, but the
    marginalisation can be specified to either be between :math:`-\infty` and :math:`\infty`, or 0 and
    :math:`\infty`.

    Parameters
    ----------
    i : int
        The ravelled index within the model parameter space.
    shape : list
        A list of integers giving the size of each parameter range.
    sk : :class:`numpy.ndarray`
        The data's noise standard deviation.
    nbackground : int
        The number of background models with amplitudes to be marginalised over
    lastHalfRange : bool
        If True then the model amplitude will be analytically integrated between 0 and :math:`\infty`, if
        False it will be integrated between :math:`-\infty` and :math:`\infty`.

    d : :class:`numpy.ndarray`
        A 1D array containing the light curve time series data.

    ms : :class:`numpy.ndarray`
        A matrix containing a time series for the model for each set of the model's parameters. The
        time series is the final dimension of the matrix.

    bgcross : :class:`numpy.ndarray`
        A 2D array containing the cross products of the background polynomial terms.

    mdbgcross : :class:`numpy.ndarray`
        A matrix containing the cross product of the model with each of the background polynomial
        terms as a function of time.

    mdcross : :class:`numpy.ndarray`
        A matrix containing the cross product of the model with the data as a function of time.

    dbgr : :class:`numpy.ndarray`
        A matrix containing the cross product of the data with each background polynomial term as a
        function of time.

    Returns
    -------
    B : :class:`numpy.ndarray`
        A matrix containing the 1D time series of the likelihood ratio.

    See also
    --------
    log_marg_amp_full_2Dmodel : A version of this function specifically for a 2D model.
    :func:`log_marg_amp_full_C` : The kernel of this function, which calculates the
                                  marginalised likelihood ratio for a single model.
    """
    q = np.unravel_index(i, shape) # get tuple of index positions

    cdef np.ndarray modelModel = np.zeros((nbackground+1)*(nbackground+1), dtype=DTYPE)
    cdef np.ndarray dataModel = np.zeros(nbackground+1, dtype=DTYPE)
    cdef unsigned int loopmax = 0, j = 0, k = 0, l = 0, endidx = nbackground, lhr = 0

    # get the model (e.g. flare) crossed with the data
    dm = np.correlate(d, ms[q], 'same')

    loopmax = len(dm)

    cdef np.ndarray B = np.zeros(loopmax, dtype=DTYPE)

    cdef double sigma = 1. # assume data and cross terms have been pre-whitened with variance values and set to one
    cdef double sumlogsigma = 0.
    if len(sk) == 1:
        sigma = sk[0]
    else: # get normalising factor from sum of sigmas
        sumlogsigma = np.sum(np.log(sk[np.isfinite(sk)]))

    if lastHalfRange:
        lhr = 1
    else:
        lhr = 0

    for j in range(loopmax): # loop over data
        # fill in arrays for marginalisation function
        for k in range(endidx):
            for l in range(k, endidx):
                modelModel[k*(nbackground+1)+l] = bgcross[k,l,j]

        # fill in the model model and data model terms
        for k in range(nbackground+1):
            if k < nbackground:
                modelModel[k*(nbackground+1)+endidx] = mdbgcross[q+(k,j)]
                dataModel[k] = dbgr[k,j]
            else:
                modelModel[k*(nbackground+1)+endidx] = mdcross[q+(j,)]
                dataModel[k] = dm[j]

        B[j] = log_marg_amp_full_C(nbackground+1, <double*> modelModel.data, <double*> dataModel.data, sigma, lhr)
        B[j] += sumlogsigma

    return B


cpdef log_marg_amp_full_2Dmodel(i, shape, np.ndarray[DTYPE_t, ndim=1] sk, nbackground, lastHalfRange, np.ndarray[DTYPE_t, ndim=1] d,
                                np.ndarray[DTYPE_t, ndim=3] ms, np.ndarray[DTYPE_t, ndim=3] bgcross,
                                np.ndarray[DTYPE_t, ndim=4] mdbgcross, np.ndarray[DTYPE_t, ndim=3] mdcross,
                                np.ndarray[DTYPE_t, ndim=2] dbgr):
    """
    A version of :func:`log_marg_amp_full_model`, but only for models where it is known that they
    have only two parameters over which the likelihood ratio must be calculated. This allows us to
    specify the shapes of the input data, which can speed up the cython implementation.

    In this case for the input model functions array `ms` the first two dimensions specify the
    values of the model's two parameters, whilst the third is the time series.

    See also
    --------
    log_marg_amp_full_model : A generic version of this function for models of arbitrary
                              dimensionality.
    """
    q = np.unravel_index(i, shape) # get tuple of index positions
    cdef np.ndarray modelModel = np.zeros((nbackground+1)*(nbackground+1), dtype=DTYPE) # make array 1D
    cdef np.ndarray dataModel = np.zeros(nbackground+1, dtype=DTYPE)
    cdef unsigned int loopmax = 0, j = 0, k = 0, l = 0, endidx = nbackground, lhr = 0

    # get the model (e.g. flare) crossed with the data
    dm = np.correlate(d, ms[q[0], q[1]], 'same')

    loopmax = len(dm)

    cdef np.ndarray B = np.zeros(loopmax, dtype=DTYPE)

    cdef double sigma = 1. # assume data and cross terms have been pre-whitened with variance values and set to one
    cdef double sumlogsigma = 0.
    if len(sk) == 1:
        sigma = sk[0]
    else: # get normalising factor from sum of sigmas
        sumlogsigma = np.sum(np.log(sk[np.isfinite(sk)]))

    if lastHalfRange:
        lhr = 1
    else:
        lhr = 0

    for j in range(loopmax): # loop over data
        # fill in arrays for marginalisation function
        for k in range(endidx):
            for l in range(k, endidx):
                modelModel[k*(nbackground+1) + l] = bgcross[k,l,j]

        # fill in the model model and data model terms
        for k in range(nbackground+1):
            if k < nbackground:
                modelModel[k*(nbackground+1) + endidx] = mdbgcross[q[0],q[1],k,j]
                dataModel[k] = dbgr[k,j]
            else:
                if np.isinf(mdcross[q[0],q[1],j]):
                    B = -np.inf*np.ones(loopmax, dtype=DTYPE)
                    return B

                modelModel[k*(nbackground+1) + endidx] = mdcross[q[0],q[1],j]
                dataModel[k] = dm[j]

        B[j] = log_marg_amp_full_C(nbackground+1, <double*> modelModel.data, <double*> dataModel.data, sigma, lhr)
        B[j] += sumlogsigma

    return B


cpdef log_marg_amp_full_background(np.ndarray[DTYPE_t, ndim=1] sk, dlen, nbackground, np.ndarray[DTYPE_t, ndim=3] bgcross, np.ndarray[DTYPE_t, ndim=2] dbgr):
    """
    Run :func:`log_marg_amp_full_C` to get the background (modelled by `nbackground` functions)
    versus Gaussian noise log likelihood ratio.

    Parameters
    ----------
    sk : :class:`numpy.ndarray`
        The noise standard deviation.
    dlen : int
        The length of the data.
    nbackground : int
        The number of background functions with amplitudes to be marginalised.

    bgcross : :class:`numpy.ndarray`
        A matrix of cross-terms of the models from each background term [size:
        (`nbackground`)-by-(`nbackground`)].

    dbgr : :class:`numpy.ndarray`
       A matrix of the data crossed with each of the background terms (as a
       sliding window of `bglen` across the data `dlen`) [size: (`nbackground`)-by-`dlen`].

    Returns
    -------
    B : :class:`numpy.ndarray`
        A 1D array of the log-likelihood ratio at each data time stamp [size: `dlen`]
    """
    cdef int j = 0, k = 0, n = 0

    cdef np.ndarray B = np.zeros(dlen, dtype=DTYPE)
    cdef np.ndarray modelModel = np.zeros(nbackground*nbackground, dtype=DTYPE)
    cdef np.ndarray dataModel = np.zeros(nbackground, dtype=DTYPE)

    cdef double sigma = 1. # assume data and cross terms have been pre-whitened with variance values and set to one
    cdef double sumlogsigma = 0.
    if len(sk) == 1:
        sigma = sk[0]
    else: # get normalising factor from sum of sigmas
        sumlogsigma = np.sum(np.log(sk[np.isfinite(sk)]))

    for j in range(dlen):
        for k in range(nbackground):
            dataModel[k] = dbgr[k,j]

            for n in range(k, nbackground):
                modelModel[k*nbackground+n] = bgcross[k,n,j]

        B[j] = log_marg_amp_full_C(nbackground, <double *> modelModel.data, <double*> dataModel.data, sigma, 0)
        B[j] += sumlogsigma

    return B


cpdef log_marg_amp_full(Nmodels, np.ndarray[DTYPE_t, ndim=2] modelModel,
                        np.ndarray[DTYPE_t, ndim=1] dataModel, sigma, lastHalfRange):
    """
    Calculates the log-likelihood ratio for a signal model (consisting of a number `Nmodels`
    components, each with an amplitude that is marginalised over) compared to a pure Gaussian noise
    model. For all bar the last model component the amplitudes are analytically marginalised
    between :math:`-\infty` and :math:`\infty`, whilst if `lastHalfRange` is true the final amplitude
    will be marginalised between 0 and infinity (i.e. it must be positive).

    .. note:: This function is several times *slower* than the equivalent straight C version
        :func:`log_marg_amp_full_C`, which should generally be used instead. In that function the
        `modelModel` array is flattened.

    Parameters
    ----------
    Nmodels : int
        The number of model components.
    modelModel : :class:`numpy.ndarray`
        The summed cross terms for each model component [size: `Nmodels`-by-`Nmodels`].

    dataModel : :class:`numpy.ndarray`
        The summed data crossed with each model component [size: `Nmodels`]

    sigma : float
        The underlying Gaussian noise standard deviation
    lastHalfRange : bool
        A boolean saying whether the final model component amplitudes should be marginalised over
        the full :math:`-\infty` to :math:`\infty` range (False), or between 0 and inf (True).

    Returns
    -------
    logL : double
        The log-likelihood ratio marginalised over the model component amplitudes.


    """

    cdef np.ndarray squared = np.zeros(Nmodels, dtype=DTYPE)

    # coefficients of model amplitudes
    cdef np.ndarray coeffs = np.zeros((Nmodels, Nmodels), dtype=DTYPE) # initialise all values to zero

    cdef int i = 0, j = 0, k = 0, n = 0, nm = Nmodels-1

    cdef double X = 0., invX = 0., invTwoX = 0., invFourX = 0., Y = 0., Z = 0., logL = 0.

    # set up coeffs matrix
    for i in range(Nmodels):
        if isinf(dataModel[i]): # return -inf if values of infinity are input
            return -np.inf

        squared[i] = modelModel[i,i]
        coeffs[i,i] = -2.*dataModel[i]

        for j in range(i+1, Nmodels):
            if isinf(modelModel[i,j]):
                return -np.inf

            coeffs[i,j] = 2.*modelModel[i,j]

    for i in range(nm):
        X = squared[i]
        invX = 1./X
        twoX = 0.5*invX
        fourX = 0.25*invX

        # get the coefficients from the Y^2 term
        for j in range(i, Nmodels):
            for k in range(i, Nmodels):
                # add on new coefficients of squared terms
                if j == i:
                    if k > j:
                        squared[k] = squared[k] - coeffs[j,k]*coeffs[j,k]*invFourX
                    elif k == j:
                        Z = Z - coeffs[j,k]*coeffs[j,k]*invFourX
                else:
                    if k == i:
                        coeffs[j,j] = coeffs[j,j] - (coeffs[i,j]*coeffs[i,k])*invTwoX
                    elif k > j:
                        coeffs[j,k] = coeffs[j,k] - (coeffs[i,j]*coeffs[i,k])*invTwoX

    X = squared[nm]
    Y = coeffs[nm, nm]

    # calculate analytic integral and get log likelihood
    for i in range(Nmodels):
        logL = logL - 0.5*log(squared[i])

    logL = logL - (Z - 0.25*Y*Y/X) / (2.*sigma*sigma)

    logL = logL + log(sigma)

    # check whether final model is between 0 and infinity or -infinity and infinity
    if lastHalfRange is True:
        logL = logL + 0.5*nm*LOG2PI + 0.5*LOGPI_2 + gsl_sf_log_erfc(Y/(2.*sigma*sqrt(2.*X)))
    else:
        logL = logL + 0.5*Nmodels*LOG2PI

    return logL


cpdef log_marg_amp_original(d, m, ss):
    """
    Calculate the logarithm of the likelihood ratio for the signal model compared to a pure
    Gaussian noise model, but analytically marginalised over the unknown model amplitude. This is
    calculated for each time step in the data, and assumes a constant noise level over the data.

    As this function uses :func:`numpy.correlate`, which works via FFTs, the data should be
    contiguous and evenly spaced.


    .. note::
        This uses the :mod:`scipy.special` `erf` function and is slower than :func:`log_marg_amp`,
        so in general is deprecated.

    Parameters
    ----------
    d : :class:`numpy.array`
        A 1D array containing the light curve time series data.

    m : :class:`numpy.array`
        A 1D array (of the same length as `d`) containing the model function.

    ss : float or double
        The noise variance of the data (assumed constant)

    Returns
    -------
    B : double
        The logarithm of the likelihood ratio.


    See also
    --------
    log_marg_amp : A faster version of this function.
    """
    #get the data/model cross term
    dm = np.correlate(d, m, mode='same')

    # get the model autocorrelation term
    m2 = np.sum(m*m)

    # get the likelihood marginalised over the signal amplitude
    inside_erf = sqrt(0.5 / (ss*m2))

    log_erf = np.log(1+scipy.special.erf(dm*inside_erf))

    B = ((dm*dm)/ (2 * m2 * ss)) + 0.5*np.log(np.pi / (2*m2)) + log_erf
    return B


cpdef logplus(double x, double y):
    """
    Calculate :math:`\log{(e^x + e^y)}` in a way that preserves numerical precision.

    .. note:: This should be deprecated to :func:`numpy.logaddexp`, which can also handle arrays.

    Parameters
    ----------
    x, y : double
        The natural logarithm of two values.

    Returns
    -------
    z : double
        The value of :math:`\log{(e^x + e^y)}`.

    See also
    --------
    logminus : A similar calculation, but for subtracting two values.

    """
    cdef double z = np.inf
    if isinf(x) and isinf(y) and (x < 0) and (y < 0):
        z = -np.inf
    elif (x > y) and not (isinf(x) and isinf(y)):
        z = x + log(1 + exp(y - x))
    elif (x <= y) and not (isinf(x) and isinf(y)):
        z = y + log(1 + exp(x - y))
    return z


cpdef logminus(double x, double y):
    """
    Calculate :math:`\log{(e^x - e^y)}` in a way that preserves numerical precision.

    Parameters
    ----------
    x, y : double
        The natural logarithm of two values.

    Returns
    -------
    z : double
        The value of :math:`\log{(e^x - e^y)}`.

    See also
    --------
    logplus : A similar calculation, but for adding two values.
    """
    cdef double z = np.inf
    if isinf(x) and isinf(y) and (x < 0) and (y < 0):
        z = -np.inf
    elif (x > y) and not (isinf(x) and isinf(y)):
        z = x + log(1 - exp(y - x))
    elif (x <= y) and not (isinf(x) and isinf(y)):
        z = y + log(1 - exp(x - y))
    return z


cpdef double logtrapz(np.ndarray[DTYPE_t, ndim=1] lx, np.ndarray[DTYPE_t, ndim=1] t) except? -2:
    """
    This function calculates the integral via the trapezium rule of the logarithm of a function `lx`
    over a range of values `t`.

    Parameters
    ----------
    lx : :class:`numpy.ndarray`
        The logarithm of a function.
    t : :class:`numpy.ndarray`
        The values at which the function *lx* was evaluated.

    Returns
    -------
    B : double
        The logarithm of the integral (double)
    """
    assert lx.dtype == DTYPE and t.dtype == DTYPE
    dts = np.diff(t)   # Calculate the integral step sizes
    ldts = np.log(dts) # log them
    cdef double B = (-1)*np.inf

    cdef int i = 0
    cdef double z
    cdef int loopmax = len(lx)-1

    for i in range(loopmax):
        z = logplus(lx[i], lx[i+1])
        z = z + ldts[i] - LOG2

        B = logplus(z,B)
    return B


cpdef log_likelihood_marg_background(np.ndarray mmcross, np.ndarray[DTYPE_t, ndim=1] dmcross, nmodels, sk):
   """
   This is a wrapper to :c:func:`log_marg_amp_except_final_C`, which calculates the Gaussian
   likelihood for data containing a model versus just Gaussian noise. In this case the model
   consists of a signal (e.g. a flare) and a background variation described by a polynomial, and
   the function marginalises over the unknown polynomial amplitude coefficients, but not the
   signal amplitude.

   Parameters
   ----------
   mmcross : :class:`numpy.ndarray`
      An array (`nmodels`-by-`nmodels`) consisting of the sums of the model terms (each of the
      polynomial model terms and the signal model) cross with each other. Terms involving the
      signal model must be in the final rows and columns of the array. This is flattened for use
      in :c:func:`log_marg_amp_except_final_C`.

   dmcross : :class:`numpy.ndarray`
      A 1D array consisting of the sums of the model terms crossed with the data. This must
      have the same ordering of model terms as in `mmcross`.

   nmodels: int
      The number of model terms (i.e. the number of background polynomial terms and the signal
      term).

   sk : float or double
      The data's noise standard deviation.

   Returns
   -------
   B : double
      The logarithm of the likelihood ratio.
   """

   cdef np.ndarray modelModel = np.zeros((nmodels)*(nmodels), dtype=DTYPE)
   cdef np.ndarray dataModel = np.zeros(nmodels, dtype=DTYPE)
   cdef unsigned int j = 0, k = 0, l = 0

   cdef double B

   for k in range(nmodels):
       dataModel[k] = dmcross[k]
       for l in range(k, nmodels):
           modelModel[k*(nmodels)+l] = mmcross[k,l]

   B = log_marg_amp_except_final_C(nmodels, <double*> modelModel.data, <double*> dataModel.data, sk)

   return B


cpdef log_likelihood_ratio(np.ndarray[DTYPE_t, ndim=1] model, np.ndarray[DTYPE_t, ndim=1] data, sk):
    """
    Calculate the logarithm of the Gaussian likelihood ratio for data containing a model versus the
    data just consisting of Gaussian noise. This is given by

    .. math::
        B = -\\frac{1}{2\\sigma^2}\\sum_i (m_i^2 - 2 d_i m_i),

    where :math:`\\sigma` is the (assumed constant) noise standard deviation, :math:`m` is the
    signal model, and :math:`d` is the data.

    Parameters
    ----------
    model : :class:`numpy.ndarray`
        A 1D array containing the signal model.

    data : :class:`numpy.ndarray`
        A 1D array (of the same length as `model`) containing the data.

    sk : float or double
        The noise standard deviation.

    Returns
    -------
    B : double
        The logarithm of the likelihood ratio.
    """

    cdef double B

    B = -np.sum(model**2 - 2.*data*model)/(2.*sk**2)

    return B
