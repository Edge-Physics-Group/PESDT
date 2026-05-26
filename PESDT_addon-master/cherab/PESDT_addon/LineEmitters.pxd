# cython: language_level=3
from cherab.core.atomic cimport AtomicData
from cherab.core.plasma cimport PlasmaModel

from .LineShapes cimport DeltaLine, OpaqueDeltaLine, OpaqueLine
from .spectrum cimport OpaqueSpectrum
from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core cimport Line, Species, Plasma, Beam
from cherab.core.model.lineshape cimport GaussianLine, LineShapeModel
from cherab.core.utility.constants cimport RECIP_4_PI

cdef class DirectEmission(PlasmaModel):

    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         LineShapeModel _lineshape

    cdef int _populate_cache(self) except -1


cdef class DirectEmissionMol(PlasmaModel):

    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         LineShapeModel _lineshape

    cdef int _populate_cache(self) except -1

    cdef double H2_wavelength(self, str band=?)

cdef class OpaqueDirectEmission(PlasmaModel):

    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         OpaqueLine _lineshape

    cdef int _populate_cache(self) except -1
    
    cpdef OpaqueSpectrum _emission(self, Point3D point, Vector3D direction, OpaqueSpectrum spectrum)

cdef class LineExcitation_AM(PlasmaModel):

    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         LineShapeModel _lineshape

    cdef int _populate_cache(self) except -1

cdef class LineRecombination_AM(PlasmaModel):

    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         LineShapeModel _lineshape

    cdef int _populate_cache(self) except -1

cdef class LineH2_AM(PlasmaModel):

    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         LineShapeModel _lineshape

    cdef int _populate_cache(self) except -1

cdef class LineH2_pos_AM(PlasmaModel):

    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         LineShapeModel _lineshape

    cdef int _populate_cache(self) except -1


cdef class LineH_neg_AM(PlasmaModel):

    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         LineShapeModel _lineshape

    cdef int _populate_cache(self) except -1


cdef class LineH3_pos_AM(PlasmaModel):

    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         LineShapeModel _lineshape

    cdef int _populate_cache(self) except -1