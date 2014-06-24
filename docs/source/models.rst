###################
Models
###################

These classes provide model functions and also set up grid over the parameters spaces of
the models and provide log(prior) values for those parameter spaces.

A generic model class is provided that can set up model parameters, parameter ranges, and
filtering functions. This class is inherited by all the other models, which provide model
functions and priors.

.. autoclass:: bayesflare.models.Model

.. toctree::
   :maxdepth: 2

The Flare Model
===============

This class provides a flare model in which the flare light curve has a Gaussian rise and
an exponential decay as given by

.. math::
   :nowrap:

   m(t, \tau_g, \tau_e, T_0) = A_0
   \begin{cases}
    e^{-(t-T_0)^2/(2\tau_g^2)} & \textrm{if } t \le T_0, \\
    e^{-(t-T_0)/\tau_e} & \textrm{if } t > T_0,
   \end{cases}

where :math:`\tau_g` is the width of the Gaussian rise, :math:`\tau_e` is the time constant
of the exponential decay, :math:`T_0` is the time of the flare peak, and :math:`A_0` is the peak
amplitude.

In this class the parameter space grid and prior is set up such that :math:`\tau_e > \tau_g`.

.. autoclass:: bayesflare.models.Flare
   :members:

The Transit Model
=================

This class provides a generic transit model. It is not the fully physical model of [1]_, but is instead
a simple model with Gaussian wings and a flat trough.

The parameter space grid and prior is set up such that the total length on the transit does not
exceed a given value.

.. autoclass:: bayesflare.models.Transit
   :members:

General models
==============

These classes provide models for a range of generic signal types. These can be used either as signal
or noise models when forming an odds ratio.

Exponential decay/rise
---------------------

A class giving a model with exponential decay or rise.

.. autoclass:: bayesflare.models.Expdecay
   :members:

Impulse
-------

A class giving a model with a delta-function-like (single bin) impulse.

.. autoclass:: bayeflare.models.Impulse
   :members:

Gaussian
--------

A class giving a model with a Gaussian profile.

.. autoclass:: bayeflare.models.Gaussian
   :members:

Step
----

A class giving a `step function <http://en.wikipedia.org/wiki/Heaviside_step_function>`_  profile.

.. autoclass:: bayesflare.models.Step

References
==========
.. [1] Mandel and Agol, *Ap. J. Lett.*, **580** (2002), `arXiv:astro-ph/0210099 <http://arxiv.org/abs/astro-ph/0210099>`_.