from setuptools import setup, Extension
import numpy
import os

# Define the extension module
# It's inside the reproduce_neutron package
ray_wrapper_module = Extension('reproduce_neutron.ray_wrapper', 
                               sources=['reproduce_neutron/ray_wrapper.c'],
                               include_dirs=[numpy.get_include()])

# Run the setup
setup(name='ray_wrapper_ext', 
      ext_modules=[ray_wrapper_module])

# How to run:
# python setup_ray_wrapper.py build_ext --inplace
