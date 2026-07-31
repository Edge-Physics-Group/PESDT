# cython: language_level=3

from cherab.core.atomic.elements cimport Element
from cherab.core cimport Line

cdef class PESDTLine(Line):
    cdef object _arb_transition

cdef class PESDTLineMol(Line):

    cdef object _mol_transition

cdef class PESDTLinePower(Line):

    cdef object _pow_transition
