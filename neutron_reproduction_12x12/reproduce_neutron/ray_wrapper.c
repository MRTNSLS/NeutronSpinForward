/* wrapper for using jacobs_rays.c with Python C API.
* Morten Sales okt. 2023
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "jacobs_rays/jacobs_rays.c"


static PyObject* ray_wrapper(PyObject* self, PyObject* args) {


    double source_x, source_z, det_x, det_z;
    int im_size;
    if (!PyArg_ParseTuple(args, "ddddi", &source_x, &source_z, &det_x, &det_z, &im_size)) {
        return NULL; // Error parsing arguments
    }

    // start and end are the starting coordinates of the ray
    double start[3], end[3];

    start[0] = source_x;
    start[1] = 0.0;
    start[2] = source_z;

    end[0] = det_x;
    end[1] = 0.0;
    end[2] = det_z;

    // im_size (int) is size of grid along x and y
    // along y in this case is 1

    int voxel_count;  // how many voxels the ray is intercepting
    int *voxel_index = (int *)malloc( (im_size+1+im_size) * sizeof(int)); // which voxels the ray is intercepting
    double *voxel_data = (double *)malloc( (im_size+1+im_size) * sizeof(double));  // the path length within each voxel traversed

    struct jacobs_options options;

    options.im_size_default = 0;
    options.im_size_x = im_size;
    options.im_size_y = 1;
    options.im_size_z = im_size;

    options.b_default = 0;
    options.b_x = -.5 * im_size;
    options.b_y = -.5 * 1;
    options.b_z = -.5 * im_size;

    options.d_default = 0;
    options.d_x = 1.0;
    options.d_y = 1.0;
    options.d_z = 1.0;

    jacobs_ray_3d(im_size, start, end, voxel_index, voxel_data, &voxel_count, &options);


    npy_intp size = (npy_intp)voxel_count;  // Convert the int size to npy_intp

    // Create a NumPy array from the malloc-allocated memory
    PyObject* numpy_voxel_index = PyArray_SimpleNewFromData(1, &size, NPY_INT, voxel_index);

    // Prevent NumPy from deallocating the memory when the Python object is deleted
    PyArray_ENABLEFLAGS((PyArrayObject*)numpy_voxel_index, NPY_ARRAY_OWNDATA);

    // Create a NumPy array from the malloc-allocated memory
    PyObject* numpy_voxel_data = PyArray_SimpleNewFromData(1, &size, NPY_DOUBLE, voxel_data);

    // Prevent NumPy from deallocating the memory when the Python object is deleted
    PyArray_ENABLEFLAGS((PyArrayObject*)numpy_voxel_data, NPY_ARRAY_OWNDATA);

    // Create a Python tuple containing both NumPy arrays
    PyObject* result_tuple = PyTuple_Pack(2, numpy_voxel_index, numpy_voxel_data);

    // Create a Python tuple containing both NumPy arrays and an integer
    //PyObject* result_tuple = PyTuple_Pack(3, numpy_voxel_index, numpy_voxel_data, PyLong_FromLong(single_integer_value));

    // Return the tuple containing both arrays
    return result_tuple;
}



// Method table for the extension module
static PyMethodDef MyExtensionMethods[] = {
    {"ray_wrapper", ray_wrapper, METH_VARARGS, "My custom function"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef ray_wrapper_module = {
    PyModuleDef_HEAD_INIT,
    "ray_wrapper",
    "Extension module example",
    -1,
    MyExtensionMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_ray_wrapper(void) {
    import_array();  // Initialize the NumPy C API for the entire module
    return PyModule_Create(&ray_wrapper_module);
}

// gcc -shared -o ray_wrapper.so -I/usr/include/python3.x -I/path/to/numpy ray_wrapper.c -lpython3.x
// Replace "3.x" with your Python version and /path/to/numpy with the path to your NumPy installation.