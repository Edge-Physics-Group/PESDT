# cython: language_level=3
from cherab.core.atomic cimport AtomicData
from cherab.core.plasma cimport PlasmaModel
from cherab.core cimport Line, Species, LineShapeModel

cdef class Continuo(PlasmaModel):
    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         LineShapeModel _lineshape

    cdef double _continuo(self, double wvl, double te, double ne, double zeff=?)
