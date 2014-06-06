A Brief Introduction to BayesFlare
==================================

BayesFlare was created to provide an automated means of identifying
flaring events in light curves released by the Kepler
mission. The aim was to provide a technique that was able to identify
even weak events by making use of the flare signal shape. This has
led to the modern package containing functions to
perform Bayesian hypothesis testing comparing the probability of
light curves containing flares to that of them containing noise
(or non-flare-like) artefacts.

The statistical methods used in BayesFlare owe much to data analysis
developments from the field of gravitational wave research; the
detection statistic which is used is based on one developed to
identify ring-downs signals from neutron stars in gravitational wave
detector data [1]_.

During the development of the analysis a method was found to
account for underlying sinusoidal variations in light curve data by
including such variations in the signal model, and then analytically
`marginalising <http://en.wikipedia.org/wiki/Marginal_distribution>`_ over them.
The functions to do this have also been included in the `amplitude-marginaliser <https://github.com/mattpitkin/amplitude-marginaliser>`_
suite.

References
----------

.. [1] Clark, Heng, Pitkin and Woan, *PRD*, **043003** (2007), `arXiv:gr-qc/0703138 <http://arxiv.org/abs/gr-qc/0703138>`_

