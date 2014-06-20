"""
BayesFlare
==========

Provides:
   1. A number of pythonic means for handling lightcurve data from the Kepler spacecraft
   2. A set of functions for conducting analysis of the lightcurve data to find flaring events

Using the documentation
-----------------------

Documentation for pyFlare is available in the form of docstrings, and this compiled reference
guide. Further information on the methods used are covered in the paper (Pitkin et al, 2014).

The examples in the docstrings assume that `BayesFlare` has been imported as `pf`

   >>> import bayesflare as pf

Code snippets are indicated by three greater-than signs

   >>> x = 2 + 3

in common with standard usage in Python documentation.

A docstring can be read using the python interpretter's built-in function ``help``

   >>> help(pf.plot)

"""


from __future__ import absolute_import

from .data.data import Loader, Lightcurve

from .models.flare import Flare
from .models.transit import Transit
from .models.impulse import Impulse
from .models.expdecay import Expdecay
from .models.gaussian import Gaussian

from .finder.find import SigmaThresholdMethod, OddsRatioDetector

from .noise.noise import estimate_noise_ps, estimate_noise_tv, make_noise_lightcurve, addNoise, highpass_filter_lightcurve, savitzky_golay

from .stats import *
from .stats.bayes import Bayes, ParameterEstimationGrid
#from .stats.thresholding import Thresholder

from .misc.misc import nextpow2, mkdir
from .inject.inject import inject_model
from .simulate.simulate import SimLightcurve, simulate_single


