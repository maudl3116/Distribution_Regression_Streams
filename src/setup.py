from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("sigKer_fast.pyx")
)


#from distutils.core import setup
#from Cython.Build import cythonize
#from distutils.extension import Extension
#from Cython.Distutils import build_ext

#ext_modules=[
#    Extension("sigKer_fast",
#              ["sigKer_fast.pyx"],
#              libraries=["m"],
#              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
#              extra_link_args=['-fopenmp']
#              ) 
#]

#setup( 
#  name = "sigKer_fast",
#  cmdclass = {"build_ext": build_ext},
#  ext_modules = ext_modules
#)