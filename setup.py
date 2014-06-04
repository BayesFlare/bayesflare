from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import os, sys

ext_modules = [ Extension("bayesflare.stats.general", ["bayesflare/stats/general.pyx", "bayesflare/stats/log_marg_amp_full.c"], 
                include_dirs=['.', os.popen('gsl-config --cflags').read()[2:-1]],  
                library_dirs=['.', os.popen('gsl-config --libs').read().split()[0][2:]], 
                libraries=['gsl', 'gslcblas'], extra_compile_args=['-O3']) ]

directives = {'embedsignature': True} # embed cython function signature in docstring

packs = ['bayesflare.data.data', 
         'bayesflare.models.flare', 
         'bayesflare.models.transit', 
         'bayesflare.models.sinusoid',
         'bayesflare.models.impulse',
         'bayesflare.models.gaussian',
         'bayesflare.models.expdecay',
         'bayesflare.noise.noise',
         'bayesflare.misc.misc',
         'bayesflare.simulate.simulate',
         'bayesflare.inject.inject', 
         'bayesflare.finder.find',
         'bayesflare.stats.thresholding',
         'bayesflare.stats.bayes',
         'bayesflare.stats.general']
setup(
  name = 'BayesFlare',
  version = '0.1',
  description = 'Python functions and classes implementing a Bayesian approach to flare finding.',
  author = 'Matthew Pitkin, Daniel Williams',
  author_email = 'matthew.pitkin@gla.ac.uk',
  packages = ['bayesflare'],
  py_modules = packs,
  cmdclass = {'build_ext': build_ext},
  ext_modules = cythonize(ext_modules, gdb_debug=False)
)
