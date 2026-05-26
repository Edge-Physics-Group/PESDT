# cython: language_level=3

from cherab.core.atomic.elements cimport Element
from cherab.core cimport Line

cdef class PESDTLine(Line):
    pass

cdef class PESDTLineMol(Line):

    cdef object _mol_transition
