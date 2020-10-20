from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["sigKer_fast.pyx","gaKer_fast.pyx"]),
    include_dirs=[numpy.get_include()]
)
