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
            print(num_pairs)
            ptr +=1 # Move ptr to start of data
            st = np.zeros((num_pairs, 2))
            print(st.shape)
            for j in range(num_pairs):
                st[j, :] = np.array([np.float64(x) for x in lines[ptr+j].split()])
            structs[i] = st
            ptr += num_pairs
        return structs
    filepath = os.path.join(os.environ.get('PESDT_HOME', os.path.expanduser('~')) + "/PESDT/devices/DIIID/structure.dat")
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