# cython: language_level=3
from cherab.core.atomic cimport Element, Isotope
from cherab.core.distribution cimport DistributionFunction
from cherab.core cimport Species


# immutable, so the plasma doesn't have to track changes
cdef class PESDTSpecies(Species):
    pass

cdef class PESDTElement(Element):
    cdef Element _base_element
    cpdef Element base_element(self)