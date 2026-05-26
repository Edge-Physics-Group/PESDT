# cython: language_level=3
from libc.math cimport exp, M_PI
from raysect.optical cimport Vector3D
cimport cython

from cherab.core.math cimport autowrap_function3d, autowrap_vectorfunction3d
from cherab.core.utility.constants cimport ELEMENTARY_CHARGE
from cherab.core cimport DistributionFunction
from cherab.core.math cimport Function3D, VectorFunction3D

cdef class PESDTMaxwellian(DistributionFunction):

    cdef :
            Function3D _density, _temperature, _emission
            VectorFunction3D _velocity
            double _atomic_mass
            dict _emission_dict

cdef class PESDTOpaqueMaxwellian(DistributionFunction):

    cdef :
            Function3D _density, _temperature, _emission
            VectorFunction3D _velocity
            double _atomic_mass
            dict _emission_dict