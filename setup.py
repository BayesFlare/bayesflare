#! /usr/bin/env python

from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.sdist import sdist  as _sdist
import os, sys



# Make sure that when a source distribution gets rolled that the Cython files get rebuilt into C
class sdist(_sdist):
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        from Cython.Build import cythonize
        cythonize(['bayesflare/stats/general.pyx'])
        _sdist.run(self)


# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

cmdclass = { 'build_ext': build_ext, 'sdist': sdist}
# check if cython is available in the system
try:
    import Cython
except ImportError:
    use_cython = False
else:
    use_cython = True

if use_cython:
    from Cython.Build import cythonize
    #import numpy
    directives = {'embedsignatjobsure': True} # embed cython function signature in docstring
    ext_modules = [ Extension("bayesflare.stats.general",
                              sources =[ "bayesflare/stats/log_marg_amp_full.c", "bayesflare/stats/general.pyx"],
                              include_dirs=['.', os.popen('gsl-config --cflags').read()[2:-1]],
                              library_dirs=['.', os.popen('gsl-config --libs').read().split()[0][2:]],
                              libraries=['gsl', 'gslcblas'], extra_compile_args=['-O3']) ]
    ext_modules = cythonize(ext_modules, gdb_debug=False, compiler_directives=directives)
else:
    #import numpy
    ext_modules = [ Extension("bayesflare.stats.general",
                              sources =[ "bayesflare/stats/log_marg_amp_full.c", "bayesflare/stats/general.c"],
                              include_dirs=['.', os.popen('gsl-config --cflags').read()[2:-1]],
                              library_dirs=['.', os.popen('gsl-config --libs').read().split()[0][2:]],
                              libraries=['gsl', 'gslcblas'], extra_compile_args=['-O3']) ]

setup(
    name = 'bayesflare',
    version = '1.0.4',
    url = 'https://github.com/BayesFlare/bayesflare',
    description = 'Python functions and classes implementing a Bayesian approach to flare finding.',
    author = 'Matthew Pitkin, Daniel Williams',
    author_email = 'matthew.pitkin@glasgow.ac.uk',
    setup_requires=['numpy'],
    install_requires = ['numpy', 'scipy', 'matplotlib', 'pyfits'],
    packages = find_packages(),
    cmdclass = cmdclass,
    ext_modules = ext_modules,
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

