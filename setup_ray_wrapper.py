from distutils.core import setup, Extension   # sudo apt install python-distutils-extra, sudo get install python3-dev
import numpy

# define the extension module
ray_wrapper_module = Extension('ray_wrapper', sources=['ray_wrapper.c'],
                               include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[ray_wrapper_module])

#python setup_ray_wrapper.py build_ext --inplace

#Compile the Extension:
#Use a C compiler (e.g., GCC) to compile the code into a shared library.
# Run python setup_ray_wrapper.py build_ext --inplace to build the extension module.
# This will generate a .so file that you can import in Python.