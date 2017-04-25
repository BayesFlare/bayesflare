#! /usr/bin/env python

from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import os, sys, numpy

ext_modules = [ Extension("bayesflare.stats.general",
                          sources =[ "bayesflare/stats/log_marg_amp_full.c", "bayesflare/stats/general.pyx"],
                          include_dirs=[numpy.get_include(), '.', os.popen('gsl-config --cflags').read()[2:-1]],
                          library_dirs=['.', os.popen('gsl-config --libs').read().split()[0][2:]],
                          libraries=['gsl', 'gslcblas'], extra_compile_args=['-O3']) ]

directives = {'embedsignatjobsure': True} # embed cython function signature in docstring

setup(
  name = 'bayesflare',
  version = '1.0.2',
url = 'https://github.com/BayesFlare/bayesflare',
  description = 'Python functions and classes implementing a Bayesian approach to flare finding.',
  author = 'Matthew Pitkin, Daniel Williams',
  author_email = 'matthew.pitkin@gla.ac.uk',
  packages = find_packages(),
  cmdclass = {'build_ext': build_ext},
  ext_modules = cythonize(ext_modules, gdb_debug=False, compiler_directives=directives),
  classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: GNU General Public License (GPL)',
      'Operating System :: POSIX :: Linux',
      'Programming Language :: Python',
      'Programming Language :: C',
      'Natural Language :: English',
      'Topic :: Scientific/Engineering :: Astronomy',
      'Topic :: Scientific/Engineering :: Information Analysis'
      ]
)

