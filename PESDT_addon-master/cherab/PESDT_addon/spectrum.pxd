# cython: language_level=3
cimport cython
from raysect.optical cimport Spectrum, Point3D
from numpy cimport ndarray

from raysect.core.math.cython cimport integrate, interpolate
from numpy cimport PyArray_SimpleNew, PyArray_FILLWBYTE, NPY_FLOAT64, npy_intp, import_array

# required by numpy c-api
import_array()


cdef class OpaqueSpectrum(Spectrum):
    cdef:
        readonly ndarray absorbances
        double[::1] absorbances_mv
        bint prev_init
        Point3D prev_point

cdef OpaqueSpectrum new_spectrum(double min_wavelength, double max_wavelength, int bins)