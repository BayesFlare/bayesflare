Tutorial
=========

The BayesFlare package has been designed to make working with Kepler data
easier, and searching for flares in the data more straight-forward.

To use the package you first need to import it into your Python script

   >>> import bayesflare as bf

This tutorial assumes that the module has been imported with the name 'bf'.

In order to conduct any meaningful work with this package, we need access to
data from the Kepler mission. At present the package only supports the
public light curves, but this may change in the future. The data should be stored
in the following format:

::

    root-dir
       - Q1_public
       - Q2_public
       - ...
       - Q14_public

To access the data it's then as simple as

   >>> client = bf.Loader('./data')

assuming that the root-dir is ``./data`` in this case.

The loader can then be used to search for lightcurves for any star in the Kepler Input Catalogue;

   >>> client.find(757450)

which will return a list of FITS files in the data folder which correspond to KIC 757450.
To work with this data we load it into a Lightcurve object

   >>> curves = client.find(757450)
   >>> lightcurve = bf.Lightcurve(curves[0])

The :class:`.Lightcurve` object handles a large number of processing for the data,
including DC offset removal and detrending (with e.g. the Savitzky-Golay algorithm implementing
in :func:`.savitzky_golay`). When data is added to the :class:`.Lightcurve` object it is interpolated to
remove ``NAN`` and ``INF`` values which cause disruption to a number of the
detection processes. The various light curves are then assembled into a
combined light curve.

In order to conduct analysis on the data we'll need a model to compare
it to. For example, a flare model:

   >>> M = bf.Flare(lightcurve.cts)

This will produce a flare model class containing a signal parameter range (for the Gaussian rise time
:math:`\tau_g` and exponential decay time :math:`\tau_e`) and default grid of signal parameters.

The odds ratio (or Bayes factor) for this model versus Gaussian noise, as a function of time and
parameter space, is produced with:

   >>> B = bf.Bayes(lightcurve, M) # create the Bayes class
   >>> B.bayes_factors()           # calculate the log likelihood ratio

Within the :class:`.Bayes` object this odds ratio is contained within the :class:`numpy.ndarray`
`lnBmargAmp`. A final odds ratio as a function of time, with the model parameters
marginalised over, is produced via

   >>> O = B.marginalise_full()

where ``O`` will be a new instance of the :class:`.Bayes` class in which `lnBmargAmp` is a 1D array
of the natural logarithm of the odds ratio.

Much of this functionality, including thresholding the odds ratio for detection purposes, can be
found in the :class:`.OddsRatioDetector` class.

More involved examples of using the code can be found in the scripts described in :ref:`scripts-label`
