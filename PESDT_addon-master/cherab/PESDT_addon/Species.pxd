# cython: language_level=3
from cherab.core.atomic cimport Element, Isotope
from cherab.core.distribution cimport DistributionFunction
from cherab.core cimport Species

cdef class PESDTSpecies(Species):
    pass

cdef class PESDTElement(Element):
    cdef Element _base_element