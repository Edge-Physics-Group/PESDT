# cython: language_level=3
import numpy as np
cimport numpy as np
from numpy cimport import_array, ndarray
from libc.math cimport pow, log, exp
cimport cython
import_array()
cdef class RateFunction():
    #
    # Generalized rate function for calculating emission from 2D AMJUEL rate coefficients
    # returns the cross-section/ratio multiplied by an Einstein coefficient, i.e. all that is need is to multiply the result by the correct
    #

    cdef ndarray _MARc
    cdef double _A_coeff
    cdef double res
    cpdef evaluate(self, double ne, double T)
cdef class NullRateFunction():
    #
    # Generalized rate function for calculating emission from 2D AMJUEL rate coefficients
    # returns the cross-section/ratio multiplied by an Einstein coefficient, i.e. all that is need is to multiply the result by the correct
    #

    cdef object _MARc
    cdef object _A_coeff
    cdef double res
    cpdef evaluate(self, double ne, double T)