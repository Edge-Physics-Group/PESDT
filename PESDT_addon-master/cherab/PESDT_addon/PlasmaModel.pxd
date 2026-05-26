
from raysect.optical cimport Point3D, Vector3D

from cherab.core.plasma.node cimport Plasma
from cherab.core.atomic cimport AtomicData
from .spectrum cimport OpaqueSpectrum

cdef class OpaquePlasmaModel:

    cdef:
        Plasma _plasma
        AtomicData _atomic_data

    cdef object __weakref__

    cpdef OpaqueSpectrum emission(self, Point3D point, Vector3D direction, OpaqueSpectrum spectrum)