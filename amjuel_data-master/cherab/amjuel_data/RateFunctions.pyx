import numpy as np
cimport numpy as np
from libc.math cimport pow, log, exp
from .amread import A_coeff

cimport cython

#ctypedef np.double_t DTYPE_t
DTYPE = np.double

cdef class RateFunction():
    #
    # Generalized rate function for calculating emission from 2D AMJUEL rate coefficients
    # returns the cross-section/ratio multiplied by an Einstein coefficient, i.e. all that is need is to multiply the result by the correct
    #

    cdef np.ndarray _MARc
    cdef double _A_coeff
    cdef double res

    def __init__(self, MARc, transition):
        self._MARc = MARc
        self._A_coeff = A_coeff(transition)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef double _evaluate(self, double ne, double T):
        cdef Py_ssize_t x_max = self._MARc.shape[0]
        
        cdef double nE

        cross_sections = np.zeros((x_max, x_max), dtype=DTYPE)

        #Define memory_views for fast access
        cdef double[:, :] cross_sections_mem_view = cross_sections
        cdef double[:, :] MARc_mem_view = self._MARc

        nE = (ne/1e8)*1e-6
        cdef Py_ssize_t n, m
        for n in range(x_max):
            for m in range(x_max):
                cross_sections_mem_view[n,m] = MARc_mem_view[n,m]*pow( log(T), n)*pow(log(nE), m)

        res = np.sum(cross_sections)
        return self._A_coeff*exp(res)

    def evaluate(self, ne, T):
        return self._evaluate(ne, T)   

cdef class NullRateFunction():
    
    cdef dict __dict__

    def __init__(self, MARc, transition):
        self.MARC = None # Do not accept any coeff
        self.A_coeff = None

    cpdef double evaluate(self, double T, double ne):
        return 0.0