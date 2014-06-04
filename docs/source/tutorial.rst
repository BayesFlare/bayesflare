Tutorial
=========

The pyFlare package has been designed to make working with Kepler data
easier, and searching for flares in the data more straight-forward.

To use the package you first need to import it into your Python script 

   >>> import bayesflare as pf

This tutorial assumes that the module has been imported with the name 'pf'.

In order to conduct any meaningful work with this package, we need access to 
data from the Kepler mission. At present the package only supports the
public lightcurves, but this may change in the future. The data should be stored 
in the following format:

::

    root-dir
       - Q1_public
       - Q2_public
       - ...
       - Q14_public

To access the data it's then as simple as

   >>> client = pf.Loader('./data')

assuming that the root-dir is ``./data`` in this case.

The loader can then be used to search for lightcurves for any star in the Kepler Input Catalogue; 

   >>> client.find(757450)

which will return a list of FITS files in the data folder which correspond to KIC 757450.
To work with this data we load it into a Lightcurve object

   >>> curves = client.find(757450)
   >>> lightcurve = pf.Lightcurve(curves)

The Lightcurve object handles a large number of processing for the data, 
including DC offset removal, and primitive stitching of different quarters' data. 
When data is added to the Lightcurve object it is interpolated to
remove ``NAN`` and ``INF`` values which cause disruption to a number of the
detection processes. The various lightcurves are then assembled into a
combined lightcurve.

In order to conduct analysis on the data we'll need a model to compare
it to. For example, a flare model:

   >>> M = pf.Flare(lightcurve.cts)

but transit models are also available:

   >>> M2 = pf.Transit(lightcurve.cts)
