# cython: language_level=3

from raysect.core.math.function.float.function2d.interpolate.common cimport MeshKDTree2D
from cherab.core.math.function cimport Function2D
from raysect.core.math.point cimport new_point2d
from cherab.core.math.function cimport VectorFunction2D
from raysect.core.math.vector cimport Vector3D, new_vector3d

import numpy as np
cimport numpy as np

cimport cython


cdef class EIRENEFunction2D(Function2D):

    def __init__(
        self,
        object vertex_coords not None,
        object triangles not None,
        object triangle_data not None
    ):

        # internal mesh arrays
        vertex_coords = np.asarray(vertex_coords, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int32)

        # IMPORTANT:
        # do not copy triangle_data
        self._triangle_data = triangle_data

        # acceleration structure
        self._kdtree = MeshKDTree2D(vertex_coords, triangles)

        # typed memoryview
        self._triangle_data_mv = self._triangle_data

    def __getstate__(self):
        return self._triangle_data, self._kdtree

    def __setstate__(self, state):
        self._triangle_data, self._kdtree = state
        self._triangle_data_mv = self._triangle_data

    def __reduce__(self):
        return self.__new__, (self.__class__,), self.__getstate__()

    @classmethod
    def instance(
        cls,
        EIRENEFunction2D instance not None,
        object triangle_data=None
    ):

        cdef EIRENEFunction2D m

        m = EIRENEFunction2D.__new__(EIRENEFunction2D)

        # shared geometry
        m._kdtree = instance._kdtree

        # replacement data?
        if triangle_data is None:
            m._triangle_data = instance._triangle_data
        else:
            m._triangle_data = triangle_data

        m._triangle_data_mv = m._triangle_data

        return m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double x, double y) except? -1e999:

        cdef np.int32_t triangle_id

        if self._kdtree.is_contained(new_point2d(x, y)):

            triangle_id = self._kdtree.triangle_id
            return self._triangle_data_mv[triangle_id]

        return 0.0




cdef class EIRENEVectorFunction2D(VectorFunction2D):

    def __init__(
        self,
        object vertex_coords not None,
        object triangles not None,
        object triangle_vectors not None
    ):

        vertex_coords = np.asarray(vertex_coords, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int32)

        # IMPORTANT:
        # do not copy vector data
        self._triangle_vectors = triangle_vectors

        self._kdtree = MeshKDTree2D(vertex_coords, triangles)

        self._triangle_vectors_mv = self._triangle_vectors

    def __getstate__(self):
        return self._triangle_vectors, self._kdtree

    def __setstate__(self, state):
        self._triangle_vectors, self._kdtree = state
        self._triangle_vectors_mv = self._triangle_vectors

    def __reduce__(self):
        return self.__new__, (self.__class__,), self.__getstate__()

    @classmethod
    def instance(
        cls,
        EIRENEVectorFunction2D instance not None,
        object triangle_vectors=None
    ):

        cdef EIRENEVectorFunction2D m

        m = EIRENEVectorFunction2D.__new__(EIRENEVectorFunction2D)

        # shared geometry
        m._kdtree = instance._kdtree

        # replacement vectors?
        if triangle_vectors is None:
            m._triangle_vectors = instance._triangle_vectors
        else:
            m._triangle_vectors = triangle_vectors

        m._triangle_vectors_mv = m._triangle_vectors

        return m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef Vector3D evaluate(self, double x, double y):

        cdef:
            np.int32_t triangle_id
            double vx, vy, vz

        if self._kdtree.is_contained(new_point2d(x, y)):

            triangle_id = self._kdtree.triangle_id

            vx = self._triangle_vectors_mv[0, triangle_id]
            vy = self._triangle_vectors_mv[1, triangle_id]
            vz = self._triangle_vectors_mv[2, triangle_id]

            return new_vector3d(vx, vy, vz)

        return new_vector3d(0.0, 0.0, 0.0)