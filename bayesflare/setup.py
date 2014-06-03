from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import os, sys

ext_modules = [ Extension('pyflare.stats.general', ["pyflare/stats/general.pyx", "pyflare/stats/log_marg_amp_full.c"], 
                include_dirs=['.', os.popen('gsl-config --cflags').read()[2:-1]],  
                library_dirs=['.', os.popen('gsl-config --libs').read().split()[0][2:]], 
                libraries=['gsl', 'gslcblas'], extra_compile_args=['-O3']) ]
packs = ['pyflare.data.data', 
         'pyflare.models.flare', 
         'pyflare.models.transit', 
         'pyflare.models.sinusoid',
         'pyflare.models.impulse',
         'pyflare.models.gaussian',
         'pyflare.models.expdecay',
         'pyflare.noise.noise',
         'pyflare.misc.misc',
         'pyflare.simulate.simulate',
         'pyflare.inject.inject', 
         'pyflare.finder.find',
         'pyflare.stats.thresholding',
         'pyflare.stats.bayes']
setup(
  name = 'pyFlare',
  version = '0.2 alpha',
  description = 'Python functions and classes for handling photometry data from the Kepler mission.',
  author = 'Daniel Williams, Matthew Pitkin',
  author_email = '1007382w@student.gla.ac.uk',
  packages = ['pyflare'],
  py_modules = packs,
  cmdclass = {'build_ext': build_ext},
  ext_modules = cythonize(ext_modules, gdb_debug=False)
)
