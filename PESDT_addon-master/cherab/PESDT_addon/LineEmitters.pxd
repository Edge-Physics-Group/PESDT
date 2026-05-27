# cython: language_level=3
from cherab.core.atomic cimport AtomicData
from cherab.core.plasma cimport PlasmaModel
from .LineShapes cimport DeltaLine, OpaqueDeltaLine, OpaqueLine
from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core cimport Line, Species, Plasma, Beam
from cherab.core.model.lineshape cimport GaussianLine, LineShapeModel
from cherab.core.utility.constants cimport RECIP_4_PI
from numpy cimport ndarray

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

cdef class OpaqueDeltaDirectEmission(PlasmaModel):

    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         LineShapeModel _lineshape
         ndarray absorbances
         double[::1] absorbances_mv
         bint prev_init
         Point3D prev_point

    cdef int _populate_cache(self) except -1
    
cdef class OpaqueGaussianDirectEmission(PlasmaModel):

    cdef:
         Line _line
         object _lineshape_class
         object _lineshape_args
         object _lineshape_kwargs
         Species _target_species
         double _wavelength
         LineShapeModel _lineshape
         ndarray absorbances
         double[::1] absorbances_mv
         bint prev_init
         Point3D prev_point

    cdef int _populate_cache(self) except -1
    
    cpdef Spectrum add_opaque_gaussian_line(self, double radiance, double absorbance, double Td, double ds, double wavelength, double sigma, Spectrum spectrum)

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