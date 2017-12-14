from distutils.sysconfig import get_config_vars
from numpy.distutils.misc_util import Configuration
from Cython.Build import cythonize
import numpy
import os


def strict_prototypes_workaround():
    (opt,) = get_config_vars('OPT')
    os.environ['OPT'] = " ".join(flag for flag in opt.split() if flag != '-Wstrict-prototypes')


def cython_path(filename):
    if "setup.pyc" in __file__:
        return __file__.replace("setup.pyc", filename)
    else:
        return __file__.replace("setup.py", filename)


def configuration(parent_package='', top_path=None):
    strict_prototypes_workaround()

    config = Configuration("svm", parent_package, top_path)
    cythonize(cython_path("optimizer.pyx"), language="c++")
    config.add_extension('optimizer', sources=["optimizer.cpp"], include_dirs=[".", numpy.get_include()],
                         extra_compile_args=["-O3", "-Wno-cpp", "-Wno-unused-function"])
    return config
