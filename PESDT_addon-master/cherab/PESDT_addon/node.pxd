
from raysect.optical cimport Node, Primitive, AffineMatrix3D
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator

from cherab.core.atomic cimport AtomicData, Element
from cherab.core.distribution cimport DistributionFunction
from cherab.core.species cimport Species
from cherab.core.math cimport VectorFunction3D
from cherab.core.plasma.model cimport PlasmaModel
from .PlasmaModel cimport OpaquePlasmaModel

cdef class Composition:

    cdef:
        dict _species
        readonly object notifier

    cpdef object set(self, object species)

    cpdef object add(self, Species species)

    cpdef Species get(self, Element element, int charge)

    cpdef object clear(self)


cdef class OpaqueModelManager:

    cdef:
        list _models
        readonly object notifier

    cpdef object set(self, object models)

    cpdef object add(self, OpaquePlasmaModel model)

    cpdef object clear(self)


cdef class OpaquePlasma(Node):

    cdef:

        readonly object notifier
        VectorFunction3D _b_field
        DistributionFunction _electron_distribution
        Composition _composition
        AtomicData _atomic_data
        Primitive _geometry
        AffineMatrix3D _geometry_transform
        OpaqueModelManager _models
        VolumeIntegrator _integrator

    cdef object __weakref__

    cdef VectorFunction3D get_b_field(self)

    cdef DistributionFunction get_electron_distribution(self)

    cdef Composition get_composition(self)

    cpdef double z_effective(self, double x, double y, double z) except -1

    cpdef double ion_density(self, double x, double y, double z)