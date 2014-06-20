#############
Data Handling
#############

The classes defined here are used to find and store Kepler light curves.

The :class:`.Lightcurve` class is in general the required data format for passing to the
Bayesian analysis functions.

.. note::
   At the moment the :class:`.Lightcurve` class is specifically designed for Kepler data (although
   it could hold any time series if necessary by hardwiring some values). In the future versions
   it will be made more generic.

Data Loader
===========
.. automodule:: bayesflare.data.data
   :members:
