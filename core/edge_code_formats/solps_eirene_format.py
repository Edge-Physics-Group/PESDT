import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import numpy as np
import sys, os
from shapely.geometry import Polygon
from .cell import Cell
from .background_plasma import BackgroundPlasma

import logging
logger = logging.getLogger(__name__)

# TODO
# 
class EIRENE(BackgroundPlasma):

    def __init__(self, sim_path: str, unstructured = True, num_species = 11):
        super().__init__()
        self.edge_code = "eirene"
        self.sim_path = sim_path

        self.fort33 = os.path.join(sim_path, "fort.33")
        self.fort34 = os.path.join(sim_path, "fort.34")
        self.fort35 = os.path.join(sim_path, "fort.35")
        #self.fort44 = os.path.join(sim_path, "fort.44") # fort.44 is data on the quadrilateral grid, unneeded
        #self.fort46 = os.path.join(sim_path, "fort.46") # fort.46 is data on the quadrilateral grid, unneeded
        
        self.intal1 = os.path.join(sim_path, "intal_1_1-1")
        self.intal2 = os.path.join(sim_path, f"intal_2_1-{num_species}")
        self.intal3 = os.path.join(sim_path, "intal_3_1-1")
        self.intal4 = os.path.join(sim_path, f"intal_4_1-{num_species}")
        self.outtal3 = os.path.join(sim_path, "outtal_0_3_1-1")
        self.vertices = self.read_ft33(self.fort33) 
        self.triangles, self.ntria = self.read_ft34(self.fort34)
        self.triangles -=1 #convert F to C indexing
        self.tria = self.vertices[self.triangles]
        self.R = self.tria[:, :, 0]
        self.Z = self.tria[:, :, 1]
        try:
            if unstructured:
                self.links = self.read_ft35_us(self.fort35)
                #self.neut, self.wld = self.read_ft44_us(self.fort44, self.ntria)
                #self.tdata = self.read_ft46_us(self.fort46, self.ntria, self.links)
            else:
                self.links = self.read_ft35(self.fort35)
                #self.neut, self.wld = self.read_ft44(self.fort44)
                #self.tdata = self.read_ft46(self.fort46)
        except Exception as e:
            logger.warning(f"Error: {e}. Could not read fort.35 links")
            self.links = None
        te_data = read_intal(self.intal1)
        t_data = read_intal(self.intal2)
        ne_data = read_intal(self.intal3)
        n_data = read_intal(self.intal4)
        n2p_data = read_outtal(self.outtal3)
        # temperatures [eV]
        self.te = te_data["data"][1, :]
        self.ti = t_data["data"][1, :]
        self.t0 = t_data["data"][2, :]
        self.t2 = t_data["data"][5, :]
        self.t2p = t_data["data"][5, :] #Use T_H2+ = T_H2
        # Densities [cm^-3], convert to m^-3
        self.ne = ne_data["data"][1, :]*1e6
        self.ni = n_data["data"][1, :]*1e6
        self.n0 = n_data["data"][2, :]*1e6
        self.n2 = n_data["data"][5, :]*1e6
        self.n2p = n2p_data["data"][1, :]*1e6

        self.n0_ph2 = n_data["data"][-2, :]*1e6 # ph. contribution to N=2
        self.n0_ph3 = n_data["data"][-1, :]*1e6 # ph. contribution to N=3

        self.poly = [Polygon(self.tria[i]) for i in range(self.ntria)]
        self.cells = [
            Cell(self.R[i, :], self.Z[i, :], None, None, self.poly[i], self.te[i], self.ti[i], self.ne[i], self.ni[i], self.n0[i], self.n2[i], self.n2p[i], t0 = self.t0[i])
            for i in range(self.ntria)
        ]

    def read_ft33(self, file, ntrfrm=0):
        """
        Read fort.33 files (triangle nodes).
        Converts coordinates from cm to m.

        Parameters
        ----------
        file : str
            Path to fort.33 file.
        ntrfrm : int, optional
            Format flag:
                0 -> x y
                1 -> id x y
            Default is 0.

        Returns
        -------
        nodes : ndarray of shape (nnodes, 2)
            Node coordinates in meters.
        """

        if ntrfrm is None:
            print("read_ft33: assuming ntrfrm = 0.")
            ntrfrm = 0

        with open(file, "r") as fid:

            # Read all numeric tokens from file
            lines = fid.readlines()
            nnodes = int(lines.pop(0))
        nodes = np.zeros((nnodes, 2))
        for i in range(nnodes):
            nodes[i, :] = np.array([float(x) for  x in lines[i].split()[1:]])


        # Convert from cm to m
        nodes *= 1e-2

        return nodes

    def read_ft34(self, file):
        """
        Read fort.34 files (nodes composing each triangle).

        Parameters
        ----------
        file : str
            Path to fort.34 file.

        Returns
        -------
        cells : ndarray of shape (ntria, 3)
            Triangle node indices.
        """

        with open(file, "r") as fid:

            # Read all numeric tokens
            tokens = fid.read().split()

        ptr = 0

        # Number of triangles
        ntria = int(tokens[ptr])
        ptr += 1

        cells = np.zeros((ntria, 3), dtype=int)

        for i in range(ntria):

            # Read 4 integers: triangle_id node1 node2 node3
            data = [
                int(tokens[ptr]),
                int(tokens[ptr + 1]),
                int(tokens[ptr + 2]),
                int(tokens[ptr + 3]),
            ]
            ptr += 4

            # Store node indices only
            cells[i, :] = data[1:]

        return cells, ntria
    
    def read_ft35(self, file):
        """
        Read fort.35 files (triangle data).

        Parameters
        ----------
        file : str
            Path to fort.35 file.

        Returns
        -------
        links : dict
            Dictionary containing:
                - 'nghbr' : neighboring triangles
                - 'side'  : side information
                - 'cont'  : connectivity information
                - 'ixiy'  : ix/iy indices
        """

        with open(file, "r") as fid:

            # Read all numeric tokens
            tokens = fid.read().split()

        ptr = 0

        # Number of triangles
        ntria = int(tokens[ptr])
        ptr += 1

        links = {
            "nghbr": np.zeros((ntria, 3), dtype=int),
            "side":  np.zeros((ntria, 3), dtype=int),
            "cont":  np.zeros((ntria, 3), dtype=int),
            "ixiy":  np.zeros((ntria, 2), dtype=int),
        }

        for i in range(ntria):

            # Read 12 integers
            data = [int(tokens[ptr + j]) for j in range(12)]
            ptr += 12

            # MATLAB:
            # data(2:3:8)  -> Python indices [1,4,7]
            # data(3:3:9)  -> Python indices [2,5,8]
            # data(4:3:10) -> Python indices [3,6,9]
            # data(11:12)  -> Python indices [10,11]

            links["nghbr"][i, :] = [data[1], data[4], data[7]]
            links["side"][i, :]  = [data[2], data[5], data[8]]
            links["cont"][i, :]  = [data[3], data[6], data[9]]
            links["ixiy"][i, :]  = [data[10], data[11]]

        return links
    
    def read_ft35_us(self, file):
        """
        Read unstructured fort.35 files (triangle data).

        Parameters
        ----------
        file : str
            Path to fort.35 file.

        Returns
        -------
        links : dict
            Dictionary containing:
                - 'nghbr'
                - 'side'
                - 'cont'
                - 'plasma_cell'
                - 'faces'
        """

        with open(file, "r") as fid:

            # Read all numeric tokens
            tokens = fid.read().split()

        ptr = 0

        # Number of triangles
        ntria = int(tokens[ptr])
        ptr += 1

        links = {
            "nghbr": np.zeros((ntria, 3), dtype=int),
            "side": np.zeros((ntria, 3), dtype=int),
            "cont": np.zeros((ntria, 3), dtype=int),
            "plasma_cell": np.zeros((ntria, 1), dtype=int),
            "faces": np.zeros((ntria, 3), dtype=int),
        }

        for i in range(ntria):

            # Read 14 integers
            data = [int(tokens[ptr + j]) for j in range(14)]
            ptr += 14

            # MATLAB:
            # data(2:3:8)  -> Python [1,4,7]
            # data(3:3:9)  -> Python [2,5,8]
            # data(4:3:10) -> Python [3,6,9]
            # data(11)     -> Python [10]
            # data(12:14)  -> Python [11:14]

            links["nghbr"][i, :] = [data[1], data[4], data[7]]
            links["side"][i, :] = [data[2], data[5], data[8]]
            links["cont"][i, :] = [data[3], data[6], data[9]]

            links["plasma_cell"][i, 0] = data[10]

            links["faces"][i, :] = data[11:14]

        return links

    def read_ft44(self, file):
        """
        Read fort.44 file.

        Notes
        -----
        Assumptions preserved from MATLAB version:
            - nlwrmsh = 1
            - nfla = 1

        Supported versions:
            - 20081111
            - 20160829
            - 20170328
        """

        print("read_ft44: assuming nlwrmsh = 1, nfla = 1.")

        nlwrmsh = 1
        nfla = 1

        neut = {}
        wld = {}

        with open(file, "r") as fid:

            #
            # Read dimensions
            #

            dims = []

            while len(dims) < 3:
                dims.extend(fid.readline().split())

            dims = [int(x) for x in dims[:3]]

            nx = dims[0]
            ny = dims[1]
            ver = dims[2]

            if ver not in [20081111, 20160829, 20170328]:
                raise ValueError(
                    "read_ft44: unknown format of fort.44 file"
                )

            # Skip possible git hash remainder of line
            fid.readline()

            #
            # natm, nmol, nion
            #

            dims = [int(x) for x in fid.readline().split()]

            dims = [int(x) for x in dims[:3]]

            natm = dims[0]
            nmol = dims[1]
            nion = dims[2]

            #
            # Ignore species labels
            #

            fid.readline()

            for _ in range(natm):
                fid.readline()

            for _ in range(nmol):
                fid.readline()

            for _ in range(nion):
                fid.readline()

            #
            # Read basic data
            #

            neut["dab2"] = read_ft44_rfield(
                fid, ver, "dab2", [nx, ny, natm]
            )

            neut["tab2"] = read_ft44_rfield(
                fid, ver, "tab2", [nx, ny, natm]
            )

            neut["dmb2"] = read_ft44_rfield(
                fid, ver, "dmb2", [nx, ny, nmol]
            )

            neut["tmb2"] = read_ft44_rfield(
                fid, ver, "tmb2", [nx, ny, nmol]
            )

            neut["dib2"] = read_ft44_rfield(
                fid, ver, "dib2", [nx, ny, nion]
            )

            neut["tib2"] = read_ft44_rfield(
                fid, ver, "tib2", [nx, ny, nion]
            )

            neut["rfluxa"] = read_ft44_rfield(
                fid, ver, "rfluxa", [nx, ny, natm]
            )

            neut["rfluxm"] = read_ft44_rfield(
                fid, ver, "rfluxm", [nx, ny, nmol]
            )

            neut["pfluxa"] = read_ft44_rfield(
                fid, ver, "pfluxa", [nx, ny, natm]
            )

            neut["pfluxm"] = read_ft44_rfield(
                fid, ver, "pfluxm", [nx, ny, nmol]
            )

            neut["refluxa"] = read_ft44_rfield(
                fid, ver, "refluxa", [nx, ny, natm]
            )

            neut["refluxm"] = read_ft44_rfield(
                fid, ver, "refluxm", [nx, ny, nmol]
            )

            neut["pefluxa"] = read_ft44_rfield(
                fid, ver, "pefluxa", [nx, ny, natm]
            )

            neut["pefluxm"] = read_ft44_rfield(
                fid, ver, "pefluxm", [nx, ny, nmol]
            )

        return neut, wld
        
    def read_ft44_us(self, file, nCv):
        """
        Read unstructured fort.44 file.

        Parameters
        ----------
        file : str
            Path to fort.44 file.
        nCv : int
            Number of control volumes / cells.

        Returns
        -------
        neut : dict
            Neutral data fields.
        wld : dict
            Wall data (currently empty).
        """

        print("read_ft44: assuming nlwrmsh = 1, nfla = 1.")

        nlwrmsh = 1
        nfla = 1

        neut = {}
        wld = {}

        with open(file, "r") as fid:

            #
            # Skip possible git-hash line
            #

            fid.readline()

            #
            # natm, nmol, nion
            #

            dims = [int(x) for x in fid.readline().split()[:3]]

            natm = dims[0]
            nmol = dims[1]
            nion = dims[2]

            #
            # Version
            #

            ver = 20170328

            #
            # Read basic data
            #

            neut["dab2"] = read_ft44_rfield(
                fid, ver, "dab2", [nCv, 1, natm]
            )

            neut["tab2"] = read_ft44_rfield(
                fid, ver, "tab2", [nCv, 1, natm]
            )

            neut["dmb2"] = read_ft44_rfield(
                fid, ver, "dmb2", [nCv, 1, nmol]
            )

            neut["tmb2"] = read_ft44_rfield(
                fid, ver, "tmb2", [nCv, 1, nmol]
            )

            neut["dib2"] = read_ft44_rfield(
                fid, ver, "dib2", [nCv, 1, nion]
            )

            neut["tib2"] = read_ft44_rfield(
                fid, ver, "tib2", [nCv, 1, nion]
            )

            neut["rfluxa"] = read_ft44_rfield(
                fid, ver, "rfluxa", [nCv, 1, natm]
            )

            neut["rfluxm"] = read_ft44_rfield(
                fid, ver, "rfluxm", [nCv, 1, nmol]
            )

            neut["pfluxa"] = read_ft44_rfield(
                fid, ver, "pfluxa", [nCv, 1, natm]
            )

            neut["pfluxm"] = read_ft44_rfield(
                fid, ver, "pfluxm", [nCv, 1, nmol]
            )

            neut["refluxa"] = read_ft44_rfield(
                fid, ver, "refluxa", [nCv, 1, natm]
            )

            neut["refluxm"] = read_ft44_rfield(
                fid, ver, "refluxm", [nCv, 1, nmol]
            )

            neut["pefluxa"] = read_ft44_rfield(
                fid, ver, "pefluxa", [nCv, 1, natm]
            )

            neut["pefluxm"] = read_ft44_rfield(
                fid, ver, "pefluxm", [nCv, 1, nmol]
            )

        return neut, wld
    
    def read_ft46(self, file):
        """
        Read fort.46 file.
        Convert data to SI units.

        Supported versions:
            - 20160513
            - 20160829
            - 20170930
        """

        with open(file, "r") as fid:

            #
            # Read dimensions
            #

            ntri = int(fid.readline().split()[0])

            ver = int(fid.readline().split()[0])

            if ver not in [20160513, 20160829, 20170930]:
                raise ValueError(
                    "read_ft46: unknown format of fort.46 file"
                )

            #
            # Skip possible git-hash line
            #

            fid.readline()

            #
            # natm, nmol, nion
            #

            dims = [int(x) for x in fid.readline().split()[:3]]

            natm = dims[0]
            nmol = dims[1]
            nion = dims[2]

            #
            # Ignore species labels
            #

            fid.readline()

            for _ in range(natm):
                fid.readline()

            for _ in range(nmol):
                fid.readline()

            for _ in range(nion):
                fid.readline()

            eV = 1.6022e-19

            tdata = {}

            #
            # Read data
            #

            # Particle densities [m^-3]
            tdata["pdena"] = (
                read_ft44_rfield(fid, ver, "pdena", [ntri, natm])
                * 1e6
            )

            tdata["pdenm"] = (
                read_ft44_rfield(fid, ver, "pdenm", [ntri, nmol])
                * 1e6
            )

            tdata["pdeni"] = (
                read_ft44_rfield(fid, ver, "pdeni", [ntri, nion])
                * 1e6
            )

            # Energy densities [J m^-3]
            tdata["edena"] = (
                read_ft44_rfield(fid, ver, "edena", [ntri, natm])
                * 1e6 * eV
            )

            tdata["edenm"] = (
                read_ft44_rfield(fid, ver, "edenm", [ntri, nmol])
                * 1e6 * eV
            )

            tdata["edeni"] = (
                read_ft44_rfield(fid, ver, "edeni", [ntri, nion])
                * 1e6 * eV
            )

            # Flux densities [kg s^-1 m^-2]
            tdata["vxdena"] = (
                read_ft44_rfield(fid, ver, "vxdena", [ntri, natm])
                * 1e1
            )

            tdata["vxdenm"] = (
                read_ft44_rfield(fid, ver, "vxdenm", [ntri, nmol])
                * 1e1
            )

            tdata["vxdeni"] = (
                read_ft44_rfield(fid, ver, "vxdeni", [ntri, nion])
                * 1e1
            )

            tdata["vydena"] = (
                read_ft44_rfield(fid, ver, "vydena", [ntri, natm])
                * 1e1
            )

            tdata["vydenm"] = (
                read_ft44_rfield(fid, ver, "vydenm", [ntri, nmol])
                * 1e1
            )

            tdata["vydeni"] = (
                read_ft44_rfield(fid, ver, "vydeni", [ntri, nion])
                * 1e1
            )

            tdata["vzdena"] = (
                read_ft44_rfield(fid, ver, "vzdena", [ntri, natm])
                * 1e1
            )

            tdata["vzdenm"] = (
                read_ft44_rfield(fid, ver, "vzdenm", [ntri, nmol])
                * 1e1
            )

            tdata["vzdeni"] = (
                read_ft44_rfield(fid, ver, "vzdeni", [ntri, nion])
                * 1e1
            )

        return tdata
        
    def read_ft46_us(self, file, nCi, ft35):
        """
        Read unstructured fort.46 file.
        Convert data to SI units.

        Parameters
        ----------
        file : str
            Path to fort.46 file.
        nCi : int
            Number of cells / control volumes.
        ft35 : dict
            fort.35 connectivity data.

        Returns
        -------
        tdata : dict
            Triangle-mapped transport data.
        """

        with open(file, "r") as fid:

            #
            # Read dimensions
            #
            line = fid.readline().split()
            ntri = int(line[0])

            ver = int(line[1])

            '''
            if ver not in [
                20160513,
                20160829,
                20170930,
                20231224,
            ]:
                raise ValueError(
                    "read_ft46: unknown format of fort.46 file"
                )
            '''
            #
            # Skip possible git-hash line
            #

            #fid.readline()

            #
            # natm, nmol, nion
            #

            dims = [int(x) for x in fid.readline().split()[:3]]

            natm = dims[0]
            nmol = dims[1]
            nion = dims[2]

            #
            # Ignore species labels
            #

            #fid.readline()

            for _ in range(natm):
                fid.readline()

            for _ in range(nmol):
                fid.readline()

            for _ in range(nion):
                fid.readline()

            eV = 1.6022e-19

            #
            # Allocate arrays
            #

            tdata = {}

            # Particle densities
            tdata["pdena"] = np.zeros((ntri, natm))
            tdata["pdenm"] = np.zeros((ntri, nmol))
            tdata["pdeni"] = np.zeros((ntri, nion))

            # Energy densities
            tdata["edena"] = np.zeros((ntri, natm))
            tdata["edenm"] = np.zeros((ntri, nmol))
            tdata["edeni"] = np.zeros((ntri, nion))

            # x fluxes
            tdata["vxdena"] = np.zeros((ntri, natm))
            tdata["vxdenm"] = np.zeros((ntri, nmol))
            tdata["vxdeni"] = np.zeros((ntri, nion))

            # y fluxes
            tdata["vydena"] = np.zeros((ntri, natm))
            tdata["vydenm"] = np.zeros((ntri, nmol))
            tdata["vydeni"] = np.zeros((ntri, nion))

            # z fluxes
            tdata["vzdena"] = np.zeros((ntri, natm))
            tdata["vzdenm"] = np.zeros((ntri, nmol))
            tdata["vzdeni"] = np.zeros((ntri, nion))

            #
            # Read data
            #

            # Particle densities [m^-3]
            tdata["pdena"][:, :] = (
                read_ft44_rfield(fid, ver, "pdena", [ntri, natm])
                * 1e6
            )

            tdata["pdenm"][:, :] = (
                read_ft44_rfield(fid, ver, "pdenm", [ntri, nmol])
                * 1e6
            )

            tdata["pdeni"][:, :] = (
                read_ft44_rfield(fid, ver, "pdeni", [ntri, nion])
                * 1e6
            )

            # Energy densities [J m^-3]
            tdata["edena"][:, :] = (
                read_ft44_rfield(fid, ver, "edena", [ntri, natm])
                * 1e6 * eV
            )

            tdata["edenm"][:, :] = (
                read_ft44_rfield(fid, ver, "edenm", [ntri, nmol])
                * 1e6 * eV
            )

            tdata["edeni"][:, :] = (
                read_ft44_rfield(fid, ver, "edeni", [ntri, nion])
                * 1e6 * eV
            )

            # x fluxes [kg s^-1 m^-2]
            tdata["vxdena"][:, :] = (
                read_ft44_rfield(fid, ver, "vxdena", [ntri, natm])
                * 1e1
            )

            tdata["vxdenm"][:, :] = (
                read_ft44_rfield(fid, ver, "vxdenm", [ntri, nmol])
                * 1e1
            )

            tdata["vxdeni"][:, :] = (
                read_ft44_rfield(fid, ver, "vxdeni", [ntri, nion])
                * 1e1
            )

            # y fluxes
            tdata["vydena"][:, :] = (
                read_ft44_rfield(fid, ver, "vydena", [ntri, natm])
                * 1e1
            )

            tdata["vydenm"][:, :] = (
                read_ft44_rfield(fid, ver, "vydenm", [ntri, nmol])
                * 1e1
            )

            tdata["vydeni"][:, :] = (
                read_ft44_rfield(fid, ver, "vydeni", [ntri, nion])
                * 1e1
            )

            # z fluxes
            tdata["vzdena"][:, :] = (
                read_ft44_rfield(fid, ver, "vzdena", [ntri, natm])
                * 1e1
            )

            tdata["vzdenm"][:, :] = (
                read_ft44_rfield(fid, ver, "vzdenm", [ntri, nmol])
                * 1e1
            )

            tdata["vzdeni"][:, :] = (
                read_ft44_rfield(fid, ver, "vzdeni", [ntri, nion])
                * 1e1
            )

        #
        # Translate data to triangle mesh
        #

        tdata["pdena"] = ft46_to_triangle_data(
            tdata["pdena"], nCi, ft35
        )

        tdata["pdenm"] = ft46_to_triangle_data(
            tdata["pdenm"], nCi, ft35
        )

        tdata["pdeni"] = ft46_to_triangle_data(
            tdata["pdeni"], nCi, ft35
        )

        tdata["edena"] = ft46_to_triangle_data(
            tdata["edena"], nCi, ft35
        )

        tdata["edenm"] = ft46_to_triangle_data(
            tdata["edenm"], nCi, ft35
        )

        tdata["edeni"] = ft46_to_triangle_data(
            tdata["edeni"], nCi, ft35
        )

        tdata["vxdena"] = ft46_to_triangle_data(
            tdata["vxdena"], nCi, ft35
        )

        tdata["vxdenm"] = ft46_to_triangle_data(
            tdata["vxdenm"], nCi, ft35
        )

        tdata["vxdeni"] = ft46_to_triangle_data(
            tdata["vxdeni"], nCi, ft35
        )

        tdata["vydena"] = ft46_to_triangle_data(
            tdata["vydena"], nCi, ft35
        )

        tdata["vydenm"] = ft46_to_triangle_data(
            tdata["vydenm"], nCi, ft35
        )

        tdata["vydeni"] = ft46_to_triangle_data(
            tdata["vydeni"], nCi, ft35
        )

        tdata["vzdena"] = ft46_to_triangle_data(
            tdata["vzdena"], nCi, ft35
        )

        tdata["vzdenm"] = ft46_to_triangle_data(
            tdata["vzdenm"], nCi, ft35
        )

        tdata["vzdeni"] = ft46_to_triangle_data(
            tdata["vzdeni"], nCi, ft35
        )

        return tdata

def read_ft44_rfield(fid, ver, fieldname, dims):
    """
    Auxiliary routine to read real fields from fort.44 file.

    Parameters
    ----------
    fid : file object
        Open file handle.
    ver : int
        fort.44 version.
    fieldname : str
        Name of field to locate/read.
    dims : list or tuple
        Desired dimensions.

    Returns
    -------
    field : ndarray
        Field data reshaped using MATLAB/Fortran ordering.
    """

    #
    # Version 20160829 and newer:
    # field labels and sizes are specified in file
    #

    if ver >= 20160829:

        # Search until fieldname is found
        while True:

            line = fid.readline()

            if line == "":
                raise EOFError(
                    f"EOF reached without finding {fieldname}."
                )

            if fieldname in line:
                break

        #
        # Consistency check:
        # number of elements in file should equal prod(dims)
        #

        parts = line.split()

        # MATLAB:
        # strread(... '%*s %*s %*s %*s %*s %*s %d')
        #
        # -> seventh token is integer count

        numin = int(parts[6])

        if numin != np.prod(dims):
            raise ValueError(
                "read_ft44_rfield: inconsistent number "
                "of input elements."
            )

    #
    # Read numerical data
    #

    nelements = int(np.prod(dims))

    values = []

    while len(values) < nelements:

        line = fid.readline()

        if line == "":
            raise EOFError(
                f"EOF reached while reading field {fieldname}."
            )

        values.extend(line.split())

    values = np.array(
        [float(v) for v in values[:nelements]],
        dtype=float
    )

    #
    # MATLAB reshape uses column-major ordering
    #

    if len(dims) > 1:
        field = values.reshape(dims, order="F")
    else:
        field = values

    return field

def ft46_to_triangle_data(field, nCi, ft35):
    """
    Translate unstructured fort.46 data back onto the triangle grid.

    Parameters
    ----------
    field : ndarray
        Input field from unstructured fort.46 output.
        Contains:
            - first nCi internal B2.5 plasma cells
            - followed by void-region triangles (if any)
    nCi : int
        Number of internal plasma cells.
    ft35 : dict
        fort.35 triangle-grid data.

    Returns
    -------
    out : ndarray
        Field mapped onto the triangle grid.
    """

    list_ = ft35["plasma_cell"]

    # Flatten in case stored as shape (ntri,1)
    list_ = np.asarray(list_).flatten()

    # Number of triangles
    ntri = len(list_)

    out = np.zeros((ntri, field.shape[1]))

    #
    # Case 1:
    # fort.35 starts with void triangles,
    # then plasma-region triangles
    #

    if list_[0] == -1:

        doing_cells = False

        for i in range(ntri):

            if list_[i] == -1:

                # MATLAB:
                # field(i+nCi,:)
                #
                # MATLAB indexing:
                #   i starts at 1
                #
                # Python indexing:
                #   i starts at 0
                #
                # Therefore:
                #   field[i + nCi, :]
                #
                out[i, :] = field[i + nCi, :]

                if doing_cells:

                    raise ValueError(
                        "this function expects an ordered list "
                        "where the triangles in the void regions "
                        "all come before those that are in B2.5 cells"
                    )

            else:

                #
                # MATLAB:
                # field(list(i),:)
                #
                # Convert 1-based -> 0-based indexing
                #

                out[i, :] = field[list_[i] - 1, :]

                doing_cells = True

    #
    # Case 2:
    # fort.35 starts with plasma-region triangles,
    # then void triangles
    #

    else:

        doing_cells = True

        for i in range(ntri):

            if list_[i] > 0:

                out[i, :] = field[list_[i] - 1, :]

                if not doing_cells:

                    raise ValueError(
                        "this function expects an ordered list "
                        "where the triangles in the void regions "
                        "all come after those that are in B2.5 cells"
                    )

            else:

                doing_cells = False

                #
                # MATLAB:
                # field(i-nCi,:)
                #
                # MATLAB index:
                #   i starts at 1
                #
                # Python index:
                #   i starts at 0
                #
                # Correct translation:
                #   field[i - nCi, :]
                #

                out[i, :] = field[i - nCi, :]

    return out

def read_intal(file):
    with open(file, "r") as f:
        lines = f.readlines()
        title = lines.pop(0)
        ncells = int(lines.pop(0).split()[1])
        nspecies = int(lines.pop(0).split()[1])
        species = lines.pop(0).split("\t")
        if len(species) != nspecies:
            "Could not get labels, labeling with indices"
            species = list(range(nspecies))
        line = lines[0]
        while ("===========================================" not in line):
            lines.pop(0)
            line = lines[0]
        lines.pop(0)
        # Begin of data
        data = np.zeros((nspecies+1, ncells))
        for i in range(ncells):
            data[:, i] = np.array([np.float64(x) for x in lines[i].split()])

    ret = {"data": data}
    ret["indices"] = ret["data"][0, :]
    for i, s in enumerate(species):
        ret[s] = ret["data"][i+1, :]

    return ret

def read_outtal(file):
    with open(file, "r") as f:
        lines = f.readlines()
        line = lines[0]
        while ("NCELLS" not in line):
            lines.pop(0)
            line = lines[0]
        ncells = int(lines.pop(0).split()[1])
        nspecies = int(lines.pop(0).split()[1])
        
        
        line = lines[0]
        while ("=================================================" not in line):
            lines.pop(0)
            line = lines[0]
        lines.pop(0)
        # Begin of data
        data = np.zeros((nspecies+1, ncells))

        
        i = 0
        while ("=================================================" not in lines[i]):
            row_vals = np.array([np.float64(x) for x in lines[i].split()])
            idx = int(row_vals[0])
            data[:, idx] = row_vals
            i+=1

    return {"data":data}
if __name__ == "__main__":
    eir = EIRENE("read_test/eirene_wg/")