# cython: language_level=3
from cherab.core.atomic cimport AtomicData
from cherab.core.plasma cimport PlasmaModel
from cherab.core cimport Line, Species, LineShapeModel
from raysect.optical cimport Spectrum, Point3D, Vector3D

cdef class Continuo(PlasmaModel):
    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         LineShapeModel _lineshape
    
    cdef Spectrum _emission(self, Point3D point, Vector3D direction, Spectrum spectrum)
    cdef double _continuo(self, double wvl, double te, double ne, double zeff=?)
