import numpy as np

class EIRENEMesh:

    def __init__(self, vertices, triangles, vol=None):

        vertices = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int32)

        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError(
                "vertices must have shape (N_vertices, 2)"
            )

        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(
                "triangles must have shape (N_triangles, 3)"
            )

        if np.any(triangles < 0):
            raise ValueError("triangle indices must be >= 0")

        if np.any(triangles >= len(vertices)):
            raise ValueError(
                "triangle index exceeds number of vertices"
            )

        if vol is not None:
            vol = np.asarray(vol, dtype=np.float64)

            if vol.shape != (triangles.shape[0],):
                raise ValueError(
                    "vol must have shape (N_triangles,)"
                )

        self._vertex_coords = vertices
        self._triangles = triangles
        self._vol = vol

        self._initial_setup()

    def _initial_setup(self):

        vertices = self._vertex_coords
        triangles = self._triangles

        self.vessel = None

        self._num_vertices = vertices.shape[0]
        self._num_tris = triangles.shape[0]

        # mesh extent
        self._mesh_extent = {
            "minr": vertices[:, 0].min(),
            "maxr": vertices[:, 0].max(),
            "minz": vertices[:, 1].min(),
            "maxz": vertices[:, 1].max()
        }

        # triangle vertex coordinates
        tri_vertices = vertices[triangles]

        r1 = tri_vertices[:, 0, 0]
        z1 = tri_vertices[:, 0, 1]

        r2 = tri_vertices[:, 1, 0]
        z2 = tri_vertices[:, 1, 1]

        r3 = tri_vertices[:, 2, 0]
        z3 = tri_vertices[:, 2, 1]

        # triangle centroids
        self._cr = (r1 + r2 + r3) / 3.0
        self._cz = (z1 + z2 + z3) / 3.0

        # RZ triangle area
        self._area = 0.5 * np.abs(
            (r2 - r1) * (z3 - z1)
            -
            (r3 - r1) * (z2 - z1)
        )

        # toroidal volume
        self._vol = 2.0 * np.pi * self._cr * self._area


    @property
    def vertex_coordinates(self):
        return self._vertex_coords

    @property
    def triangles(self):
        return self._triangles

    @property
    def num_vertices(self):
        return self._num_vertices

    @property
    def num_triangles(self):
        return self._num_tris

    @property
    def mesh_extent(self):
        return self._mesh_extent

    @property
    def vol(self):
        return self._vol

    @property
    def cr(self):
        return self._cr

    @property
    def cz(self):
        return self._cz
    
    @property
    def triangle_vertices(self):
        return self._vertex_coords[self._triangles]
    
    def __getstate__(self):

        return {
            "vertices": self._vertex_coords,
            "triangles": self._triangles,
            "vol": self._vol
        }

    def __setstate__(self, state):

        self._vertex_coords = state["vertices"]
        self._triangles = state["triangles"]
        self._vol = state["vol"]

        self._initial_setup()