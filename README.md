# NeutronSpinForward
A project that aims to use forward models and machine learning to reconstruct magnetic fields from polarized neutron imaging experiments

for a parallel ray for a range of projection angles we calculate for each ray the precession of the neutron ray passing through a grid og magnetic fields. in 2D.
The calculation of the intersection of each ray with the grid (which voxels and distance within them) is done by using jacobs_rays.
These files are not included as I am not the author, so you will have to obtain them yourselves.
Put jacobs_rays.c and jacobs_rays.h in a folder called jacobs_rays
To compile the wrapper function for creating the module based on the c-code for your architecture, (e.g mine is intel-mac or arm64-linux, both included for some python versions), please use the setup_ray_wrapper.py script.
Call it by running: python setup_ray_wrapper.py build_ext --inplace


calc_paths.py calculates then intersections of neutron rays with a given grid and saves it as numpy array.
plot_paths.py can be used to view the rays and the grid.


USE MODEL STUFF 3 FOR NOW WITH CALC 3