
from raysect.optical cimport World, Primitive, Ray, Spectrum, SpectralFunction, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter

from .node cimport OpaquePlasma
from cherab.core.atomic cimport AtomicData


cdef class OpaquePlasmaMaterial(InhomogeneousVolumeEmitter):

    cdef:
        OpaquePlasma _plasma
        AtomicData _atomic_data
        AffineMatrix3D _local_to_plasma
        list _models