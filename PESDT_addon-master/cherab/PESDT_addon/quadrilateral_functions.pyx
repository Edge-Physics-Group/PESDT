
# cython: language_level=3

# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

import numpy as np
cimport numpy as np

from raysect.core.math.function.float.function2d.interpolate.common cimport MeshKDTree2D
from raysect.core.math.vector cimport Vector3D, new_vector3d
from raysect.core.math.point cimport new_point2d

from cherab.core.math.function cimport Function2D, VectorFunction2D
cimport cython


cdef class Quadrilateral2DFunction(Function2D):

    def __init__(self, object vertex_coords not None, object triangles not None, object triangle_to_grid_map not None, object grid_data not None):

        # use numpy arrays to store data internally
        vertex_coords = np.array(vertex_coords, dtype=np.float64)
        triangles = np.array(triangles, dtype=np.int32)
        triangle_to_grid_map = np.array(triangle_to_grid_map, dtype=np.int32)

        # Attention!!! Do not copy grid_data! Attribute self._grid_data must point to the original data array,
        # so as not to re-initialize the interpolator if the user changes data values.

        # build kdtree
        self._kdtree = MeshKDTree2D(vertex_coords, triangles)

        # populate internal attributes
        self._grid_data = grid_data
        self._triangle_to_grid_map = triangle_to_grid_map

        self._grid_data_mv = self._grid_data
        self._triangle_to_grid_map_mv = self._triangle_to_grid_map

    def __getstate__(self):
        return self._grid_data, self._triangle_to_grid_map, self._kdtree

    def __setstate__(self, state):
        self._grid_data, self._triangle_to_grid_map, self._kdtree = state
        self._triangle_to_grid_map_mv = self._triangle_to_grid_map
        self._grid_data_mv = self._grid_data

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @classmethod
    def instance(cls, Quadrilateral2DFunction instance not None, object grid_data=None):
        """
        Creates a new interpolator instance from an existing interpolator instance.
        The new interpolator instance will share the same internal acceleration
        data as the original interpolator. The grid_data of the new instance can
        be redefined.
        This method should be used if the user has multiple sets of grid_data
        that lie on the same mesh geometry. Using this methods avoids the
        repeated rebuilding of the mesh acceleration structures by sharing the
        geometry data between multiple interpolator objects.
        :param Quadrilateral2DFunction instance: Quadrilateral2DFunction object.
        :param ndarray grid_data: An array containing data on a quadrilateral grid.
        :return: A Quadrilateral2DFunction object.
        :rtype: Quadrilateral2DFunction
        """

        cdef Quadrilateral2DFunction m

        # copy source data
        m = Quadrilateral2DFunction.__new__(Quadrilateral2DFunction)
        m._kdtree = instance._kdtree
        m._triangle_to_grid_map = instance._triangle_to_grid_map

        # do we have replacement triangle data?
        if grid_data is None:
            m._grid_data = instance._grid_data
        else:
            m._grid_data = grid_data

        m._triangle_to_grid_map_mv = m._triangle_to_grid_map
        m._grid_data_mv = m._grid_data

        return m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double x, double y) except? -1e999:

        cdef:
            np.int32_t triangle_id, i

        if self._kdtree.is_contained(new_point2d(x, y)):

            triangle_id = self._kdtree.triangle_id
            i = self._triangle_to_grid_map_mv[triangle_id]
            return self._grid_data_mv[i]

        return 0.0

cdef class Quadrilateral2DVectorFunction(VectorFunction2D):

    def __init__(self, object vertex_coords not None, object triangles not None, object triangle_to_grid_map not None, object grid_vectors not None):

        # use numpy arrays to store data internally
        vertex_coords = np.array(vertex_coords, dtype=np.float64)
        triangles = np.array(triangles, dtype=np.int32)
        triangle_to_grid_map = np.array(triangle_to_grid_map, dtype=np.int32)

        # Attention!!! Do not copy grid_vectors! Attribute self._grid_vectors must point to the original data array,
        # so as not to re-initialize the interpolator if the user changes data values.

        # build kdtree
        self._kdtree = MeshKDTree2D(vertex_coords, triangles)

        # populate internal attributes
        self._grid_vectors = grid_vectors
        self._triangle_to_grid_map = triangle_to_grid_map
        self._grid_vectors_mv = self._grid_vectors
        self._triangle_to_grid_map_mv = self._triangle_to_grid_map

    def __getstate__(self):
        return self._grid_vectors, self._triangle_to_grid_map, self._kdtree

    def __setstate__(self, state):
        self._grid_vectors, self._triangle_to_grid_map, self._kdtree = state
        self._grid_vectors_mv = self._grid_vectors
        self._triangle_to_grid_map_mv = self._triangle_to_grid_map

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @classmethod
    def instance(cls, Quadrilateral2DVectorFunction instance not None, object grid_vectors=None):
        """
        Creates a new interpolator instance from an existing interpolator instance.
        The new interpolator instance will share the same internal acceleration
        data as the original interpolator. The grid_data of the new instance can
        be redefined.
        This method should be used if the user has multiple sets of grid_data
        that lie on the same mesh geometry. Using this methods avoids the
        repeated rebuilding of the mesh acceleration structures by sharing the
        geometry data between multiple interpolator objects.
        :param Quadrilateral2DVectorFunction instance: Quadrilateral2DVectorFunction object.
        :param ndarray grid_vectors: An array containing vector data on a quadrilateral grid.
        :return: A Quadrilateral2DVectorFunction object.
        :rtype: Quadrilateral2DVectorFunction
        """

        cdef Quadrilateral2DVectorFunction m

        # copy source data
        m = Quadrilateral2DVectorFunction.__new__(Quadrilateral2DVectorFunction)
        m._kdtree = instance._kdtree
        m._triangle_to_grid_map = instance._triangle_to_grid_map

        # do we have replacement triangle data?
        if grid_vectors is None:
            m._grid_vectors = instance._grid_vectors
        else:
            m._grid_vectors = grid_vectors

        m._triangle_to_grid_map_mv = m._triangle_to_grid_map
        m._grid_vectors_mv = m._grid_vectors

        return m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef Vector3D evaluate(self, double x, double y):

        cdef:
            np.int32_t triangle_id, i
            double vx, vy, vz

        if self._kdtree.is_contained(new_point2d(x, y)):

            triangle_id = self._kdtree.triangle_id
            i = self._triangle_to_grid_map_mv[triangle_id]
            vx = self._grid_vectors_mv[0, i]
            vy = self._grid_vectors_mv[1, i]
            vz = self._grid_vectors_mv[2, i]

            return new_vector3d(vx, vy, vz)

        return new_vector3d(0, 0, 0)