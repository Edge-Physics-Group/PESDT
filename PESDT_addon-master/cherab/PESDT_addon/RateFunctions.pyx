# cython: language_level=3
cimport numpy as np
from libc.math cimport pow, log, exp
cimport cython

#ctypedef np.double_t DTYPE_t
DTYPE = np.double
def  A_coeff(transition):

    '''
    Returns the Einstein coefficient for transition "n"<-"m"

    transition ("str", "str")

    Coefficients from "Elementary Processes in Hydrogen-Helium Plasmas" (1987), by Janev, Appendix A.2.The original source of the table is Wiese et al. (1966)
    '''

    coeff_dict = {
        1: {2: 4.699e8, 3: 5.575e7, 4: 1.278e7, 5: 4.125e6, 6: 1.644e6}, 
        2: {3: 4.410e7, 4: 8.419e6, 5: 2.53e6, 6: 9.732e5, 7: 4.389e5},
        3: {4: 8.989e6, 5: 2.201e6, 6: 7.783e5, 7: 3.358e5, 8: 1.651e5},
        4: {5: 2.699e6, 6: 7.711e5, 7: 3.041e5, 8:1.424e5, 9: 7.459e4}
    }

    return coeff_dict[transition[1]][transition[0]]

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