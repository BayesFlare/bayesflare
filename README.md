# BayesFlare

[![Semver](http://img.shields.io/SemVer/2.0.0.png)](http://semver.org/spec/v2.0.0.html)
[![GitHub version](https://badge.fury.io/gh/BayesFlare%2Fbayesflare.svg)](http://badge.fury.io/gh/BayesFlare%2Fbayesflare)

## Automated flare-finding algorithm for Kepler photometric data


This repository contains a series of Python scripts which are designed
to analyse photometric data produced by the Kepler program to find 
flaring events. The long-term aim for the program is to be capable of
running on data from all quarters collected by the spacecraft, and at
both long and short cadences.


## Requirements


* Python 2.7
* sshfs
* NumPy 1.6.1
* SciPy 0.9.0
* Matplotlib 1.1.1
* PyFITS 2.4.0


## Documentation

The documentation is available online at [readthedocs](bayesflare.readthedocs.org).

The Documentation for BayesFlare is produced by Sphinx with two extension packages,
* numpydoc
* sphinx_rtd_theme
