# cython: language_level=3

from raysect.core.math.function.float.function2d.interpolate.common cimport MeshKDTree2D
from cherab.core.math.function cimport Function2D, VectorFunction2D
from raysect.core.math.vector cimport Vector3D, new_vector3d
from raysect.core.math.point cimport new_point2d

import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython


cdef class EIRENEFunction2D(Function2D):

    cdef:
        MeshKDTree2D _kdtree
        object _triangle_data
        double[:] _triangle_data_mv

cdef class EIRENEVectorFunction2D(VectorFunction2D):

    cdef:
        MeshKDTree2D _kdtree
        object _triangle_vectors
        double[:, :] _triangle_vectors_mv