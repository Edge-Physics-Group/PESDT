import numpy as np
from raysect.optical.library.metal import RoughTungsten, RoughBeryllium
from raysect.primitive import Mesh
from raysect.optical.material import AbsorbingSurface
from cherab.tools.primitives.toroidal_mesh import toroidal_mesh_from_polygon
from cherab.tools.primitives.axisymmetric_mesh import axisymmetric_mesh_from_polygon
import os
def read_D3D_dat():
    def read_structs(lines: list[str], num_structs):
        ptr = 0
        structs = [[]]*num_structs
        for i in range(num_structs):
            ptr +=1 # Skip name of struct
            num_pairs = int(lines[ptr])
            ptr +=1 # Move ptr to start of data
            st = np.zeros((num_pairs, 2))
            for j in range(num_pairs):
                st[j, :] = np.array([np.float64(x) for x in lines[ptr+j].split()])
            structs[i] = st
            ptr += num_pairs
        return structs
    filepath = os.path.join(os.environ.get('PESDT_HOME', os.path.expanduser('~')), "devices/DIIID/structure.dat")
    with open(filepath, "r") as f:
        lines = f.readlines()
        num_structs = int(lines.pop(0)); lines.pop(0)
        structs = read_structs(lines, num_structs)
    return structs

def construct_DIIID_mesh(world, material = AbsorbingSurface(), num_toroidal_segments: int = 64):
    structs = read_D3D_dat()
    for struct in structs:

        mesh = axisymmetric_mesh_from_polygon(
        polygon=struct,
        num_toroidal_segments=num_toroidal_segments
        )
        # Build mesh volume and assign materials
        mesh.parent = world

        # Assign materials per face (optional: same material for all if needed)
        mesh.material = material

def segment_intersection(p1, p2, q1, q2):
    """
    Returns the distance t along p1->p2 where the segments intersect.

    Parameters
    ----------
    p1, p2 : (2,) ndarray
        LOS endpoints.
    q1, q2 : (2,) ndarray
        Wall segment.

    Returns
    -------
    float or None
        Distance along the LOS from p1 to the intersection,
        or None if there is no intersection.
    """

    r = p2 - p1
    s = q2 - q1

    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-12:
        # Parallel
        return None
    qp = q1 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = (qp[0] * r[1] - qp[1] * r[0]) / rxs

    if 0 <= t <= 1 and 0 <= u <= 1:
        return t * np.linalg.norm(r)

    return None

def move_los_origin(p1,p2,structures,clearance=1e-3,epsilon=1e-4,max_iterations=50,
):
    """
    Move p1 along the LOS until the nearest wall intersection
    is at least 'clearance' away.

    Parameters
    ----------
    p1, p2 : ndarray shape (2,)
        LOS endpoints.

    structures : list of (N,2) ndarrays
        Each array contains (R,Z) coordinates of one wall polyline.

    clearance : float
        Minimum allowed distance from p1 to the nearest wall.

    epsilon : float
        Extra distance moved beyond each wall intersection.

    Returns
    -------
    ndarray
        New p1.
    """
    p1 = np.asarray(p1, dtype=float).copy()
    p2 = np.asarray(p2, dtype=float)

    direction = p2 - p1
    direction /= np.linalg.norm(direction)
    for _ in range(max_iterations):
        intersections = []
        for wall in structures:
            for i in range(len(wall) - 1):
                d = segment_intersection(p1,p2,wall[i],wall[i + 1])
                if d is not None:
                    intersections.append(d)
        if not intersections:
            break
        nearest = min(intersections)
        if nearest > clearance:
            break
        p1 += (nearest + epsilon) * direction

    return p1

if __name__ == "__main__":
    import sys
    sys.path.append('/home/vp/PESDT/core/utils')
    from machine_defs import get_DIIIDdefs
    import matplotlib.pyplot as plt
    d3d = get_DIIIDdefs()
    structs = read_D3D_dat()

    fig, ax = plt.subplots()
    for struct in structs:
        ax.plot(struct[:, 0], struct[:, 1])


    diags = list(d3d.diag_dict.keys())
    for diag in diags:
        p1 = d3d.diag_dict[diag]["p1"]
        p2 = d3d.diag_dict[diag]["p2"]
        for i in range(len(p1)):
            p1_ = move_los_origin(p1[i], p2[i], structs, clearance=0.55, epsilon=0.01)
            ax.plot([p1_[0], p2[i][0]], [p1_[1], p2[i][1]], color = "black")
    plt.show()