.. _scripts-label:

Pre-made Scripts
================

The BayesFlare package is supplied with a number of pre-made scripts that can be
used as examples of how to perform an analysis. Currently these scripts use the
analysis parameters used for the search performed in `Pitkin, Williams, Fletcher & Grant, arXiv:1406.1712 <http://arxiv.org/abs/1406.1712>`_.
The scripts in general provide good examples of using the :class:`.OddsRatioDetector` class.

.. _plot-light-curve-label:
``plot_lightcurve.py``
----------------------

This script will plot a light curve and the associated log odds ratio. The underlying light curve
can either be:
 * a simulated curve consisting of pure Gaussian noise,
 * a simulated curve consisting of Gaussian noise *and* a sinusoidal variation,
 * or, a real Kepler light curve.

The light curve (whether simulated or real) can have a simulated flare signal added to it. By default
any simulated light curve will have the same length and time steps as Kepler Quarter 1 long cadence data.
By default the noise standard deviation in the data is calculated using the :func:`.estimate_noise_ps` method
(with `estfrac` = 0.5).

The default odds ratio calculation assumes a sliding analysis window of 55 time bins and:
 * a signal model consisting of a flare (with :math:`0 < \tau_g \leq 1800` seconds, :math:`0 < \tau_g \leq 3600` seconds and :math:`\tau_g \leq \tau_e`)
   *and* a 4th order polynomial variation,
 * a noise model consisting of a 4th order polynomial *or* a polynomial *and* a positive or negative impulse
   (anywhere within the analysis window length) *or* a polynomial *and* an exponential decay (with :math:`0 < \tau_g \leq 900`
   seconds) *or* a polynomial *and* an exponential rise (with :math:`0 < \tau_g \leq 900`).

To see the command line input options for this script use:

   >>> ./plot_lightcurve --help

``plot_spectrum.py``
--------------------

This script will plot the one-sided power spectrum of a real or simulated light curve. As
with :ref:`plot-light-curve-label` the simulated data in Gaussian, but can also contain a sinusoid, and
both real or simulated data can have flare signals added to them.

To see the command line input options for this script use:

   >>> ./plot_spectrum --help

``flare_detection_threshold.py``
--------------------------------

This script will compute a threshold on the log odds ratio for a given false alarm rate of detections.
The threshold can either be calculated using a set of simulated light curves containing Gaussian noise
(and also potentially containing sinusoidal variations randomly produced with amplitude and frequency
given specified ranges), or a set of real Kepler light curves.

By default any simulated light curve will have the same length and time steps as Kepler Quarter 1 long
cadence data. The noise standard deviation in the data will be calculated using the
:func:`.estimate_noise_tv` method (with `sigma` = 1.0). The odds ratio calculation assumes
 * a signal model consisting of a flare (with :math:`0 < \tau_g \leq 1800` seconds, :math:`0 < \tau_g \leq 3600` seconds and :math:`\tau_g \leq \tau_e`)
   *and* a 4th order polynomial variation,
 * a noise model consisting of a 4th order polynomial *or* a polynomial *and* a positive or negative impulse
   (anywhere within the analysis window length) *or* a polynomial *and* an exponential decay (with :math:`0 < \tau_g \leq 900`
   seconds) *or* a polynomial *and* an exponential rise (with :math:`0 < \tau_g \leq 900`).

The required false alarm probability is given as a the percentage probability that a single light curve
(with by default the Kepler Quarter 1 long cadence time scales) will contain a false detection.

To see the command line input options for this script use:

   >>> ./flare_detection_threshold --help

``flare_detection_efficiency.py``
---------------------------------

This script will calculate the efficiency of detecting flare signals for a given log odds ratio threshold.
This is done by adding simulated flare signals (which by default have time scale parameters drawn at random
uniformly within the ranges :math:`0 < \tau_g \leq 1800` seconds, :math:`0 < \tau_g \leq 3600` seconds
where :math:`\tau_g \leq \tau_e`) with a range of signal-to-noise ratios to simulated, or real, data.
Simulated data consists of Gaussian noise to which sinusoidal variations can be added.

By default any simulated light curve will have the same length and time steps as Kepler Quarter 1 long
cadence data. The noise standard deviation in the data will be calculated using the
:func:`.estimate_noise_tv` method (with `sigma` = 1.0). The odds ratio calculation assumes
 * a signal model consisting of a flare (with :math:`0 < \tau_g \leq 1800` seconds, :math:`0 < \tau_g \leq 3600` seconds and :math:`\tau_g \leq \tau_e`)
   *and* a 4th order polynomial variation,
 * a noise model consisting of a 4th order polynomial *or* a polynomial *and* a positive or negative impulse
   (anywhere within the analysis window length) *or* a polynomial *and* an exponential decay (with :math:`0 < \tau_g \leq 900`
   seconds) *or* a polynomial *and* an exponential rise (with :math:`0 < \tau_g \leq 900`).

To see the command line input options for this script use:

   >>> ./flare_detection_efficiency --help

``kepler_analysis_script.py``
-----------------------------

This scripts was used in the analysis of `Pitkin, Williams, Fletcher & Grant <http://arxiv.org/abs/14XX.XXX>`_
to automatically detect flares in Kepler Quarter 1 data. The script will get a list of Kepler
stars from `MAST <http://archive.stsci.edu/kepler/>`_ (this uses functions heavily indebted to those from [1]_)
based on effective temperature and surface gravity criteria (for which the defaults are those used in the
analysis in [2]_ with effective temperature less than 5150 and log(g) greater than 4.2). It will initially
ignore any Kepler stars for which the `condition flag <http://archive.stsci.edu/kepler/condition_flag.html>`_
is not 'None' e.g. it will ignore stars with known exoplanets or planetary candidates.

Other vetos that are used are
 * stars with known periodicities (including secondary periods) of less than two days are vetoed, based on
   values given in Tables 1 and 2 of [3] and the table in [4]_.
 * stars in eclipsing binaries (that are not covered by the condition flag veto) are vetoed, based on stars
   given in [5]_.

The analysis estimates the data noise standard deviation using the :func:`.estimate_noise_tv` method (with `sigma` = 1.0).
The odds ratio calculation assumes
 * a signal model consisting of a flare (with :math:`0 < \tau_g \leq 1800` seconds, :math:`0 < \tau_g \leq 3600` seconds and :math:`\tau_g \leq \tau_e`)
   *and* a 4th order polynomial variation,
 * a noise model consisting of a 4th order polynomial *or* a polynomial *and* a positive or negative impulse
   (anywhere within the analysis window length) *or* a polynomial *and* an exponential decay (with :math:`0 < \tau_g \leq 900`
   seconds) *or* a polynomial *and* an exponential rise (with :math:`0 < \tau_g \leq 900`).

The results (which includes a list of stars containing flare candidates and the times for each of the flares)
are returned in a `JSON <http://json.org/>`_ format text file.

``parameter_estimation_example.py``
-----------------------------------

This script shows an example of how to perform parameter estimation with the code. It sets up some fake
data containing Gaussian noise and adds a simulated flare signal to it. It then sets up a grid in
the flare parameter space upon which to calculate the posterior probability distribution. This is then
marginalised to produce 1D distributions for each parameter.

References
----------

.. [1] **kplr** - *A Python interface to the Kepler data* (http://dan.iel.fm/kplr/)
.. [2] Walkowicz *et al*, *AJ*, **141** (2011), `arXiv:1008.0853 <http://arxiv.org/abs/1008.0853>`_
.. [3] McQuillan *et al*, *ApJS*, **211** (2014), `arXiv:1402.5694 <http://arxiv.org/abs/1402.5694>`_
.. [4] Reinhold *et al*, *A&A*, **560**, (2013), `arXiv:1308.1508 <http://arxiv.org/abs/1308.1508>`_
.. [5] Prsa *et al*, *AJ*, **141** (2011), `arXiv:1006.2815 <http://arxiv.org/abs/1006.2815>`_
