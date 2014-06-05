# BayesFlare

[![Semver](http://img.shields.io/SemVer/2.0.0.png)](http://semver.org/spec/v2.0.0.html)

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


## Usage


Before the analysis is run it is necessary to set up the environment in which
the script is to run. A shell script is located at src/bash/keplerenv.sh which
will set up the required SSHFS mounts. This should be run in the following 
fashion:

	>> sh ./src/bash/keplerenv.sh

To run the analysis code:

       	>>  ./src/python/analysis.py -f <file location cotaining list of stars to analyse>

or, to run the analysis on a single star:

       >> ./src/python/analysis.py -i <idnumber>



## Documentation

The documentation is available online at [readthedocs](bayesflare.readthedocs.org).

The Documentation for BayesFlare is produced by Sphinx with two extension packages,
* numpydoc
* sphinx_rtd_theme
