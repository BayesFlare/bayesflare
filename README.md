##########
BayesFlare
##########

Automated flare-finding algorithm for Kepler photometric data
=============================================================

This repository contains a series of Python scripts which are designed
to analyse photometric data produced by the Kepler program to find 
flaring events, using a Bayesian method. The long-term aim for the program is to be capable of
running on data from all quarters collected by the spacecraft, and at
both long and short cadences.

==================
   Requirements
==================

* Python 2.7
* sshfs
* NumPy 1.7
* SciPy 0.12.0
* Matplotlib 1.3.0
* PyFITS 3.1.2

============
   Usage
============

Before the analysis is run it is necessary to set up the environment in which
the script is to run. A shell script is located at src/bash/keplerenv.sh which
will set up the required SSHFS mounts. This should be run in the following 
fashion:

	>> sh ./src/bash/keplerenv.sh

To run the analysis code:

       	>>  ./src/python/analysis.py -f <file location cotaining list of stars to analyse>

or, to run the analysis on a single star:

       >> ./src/python/analysis.py -i <idnumber>


=============
Documentation
=============
The Documentation for BayesFlare is produced by Sphinx with two extension packages,
* numpydoc
* sphinx_rtd_theme
