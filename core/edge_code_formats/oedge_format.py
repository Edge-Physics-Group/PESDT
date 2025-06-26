import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import numpy as np
import sys, os
from shapely.geometry import Polygon
from .cell import Cell
from .background_plasma import BackgroundPlasma

import logging
#import netCDF4 as nc
from omfit_classes.omfit_nc import OMFITnc as nc
logger = logging.getLogger(__name__)

class OEDGE(BackgroundPlasma):

    def __init__(self, sim_path, JET_grid = True, print_keys = False):
        super().__init__()
        self.sim_path = sim_path
        self.edge_code = "OEDGE"
        self.nc = nc(self.sim_path)
        if print_keys:
            self.print_keys_to_txt()
        self.JET_grid = JET_grid

        self.read_data()
        self.get_sep()
        self.create_cells()
        self.create_wall_poly()

    def read_data(self):
        #######################################################
        ### LOAD GEOMETRY, PLASMA, AND NEUTRAL DATA FROM NC ###
        #######################################################
        # Geometry
        self.nrs = int(self.nc["NRS"]["data"])  # Number of rings on the grid
        self.nks = self.nc["NKS"]["data"]  # Number of knots on each ring
        self.nds = int(self.nc["NDS"]["data"])  # Number of target elements
        self.irds = self.nc["IRDS"]["data"] # Ring index of target data
        self.area = self.nc["KAREAS"]["data"]  # cell area
        self.korpg = self.nc["KORPG"]["data"]  # IK,IR mapping to polygon index
        self.rvertp = self.nc["RVERTP"]["data"]  # R corner coordinates of grid polygons
        self.zvertp = self.nc["ZVERTP"]["data"]  # Z corner coordinates of grid polygons
        valid_mask = self.area > 0.0 # Find valid cells (area > 0)
        valid_irs, valid_iks = np.where(valid_mask)
        valid_indices = self.korpg[valid_irs, valid_iks] - 1  # Convert 1-based to 0-based

        # Get corner coordinates (first 4) and center coordinates (5th)
        self.rv = self.rvertp[valid_indices, :4]
        self.zv = self.zvertp[valid_indices, :4]
        self.R = self.rvertp[valid_indices, 4]
        self.Z = self.zvertp[valid_indices, 4]

        # Build list of 4-tuples for mesh (corners of each polygon)
        self.mesh = [list(zip(r, z)) for r, z in zip(self.rv, self.zv)]

        # Store the (ir, ik) indices of each valid cell
        self.mesh_idxs = list(zip(valid_irs, valid_iks))

        # Number of valid cells
        self.num_cells = len(self.mesh)

        self.nvesm = int(self.nc["NVESM"]["data"]) # Number of vessel wall points
        self.rvesm = self.nc["RVESM"]["data"]  # R coordinates of vessel wall segment end points
        self.zvesm = self.nc["ZVESM"]["data"]  # V coordinates of vessel wall segment end points

        # indexes for rings defining the SOL rings
        self.irsep = int(self.nc["IRSEP"]["data"])  # Index of first ring in main SOL
        self.irwall = int(self.nc["IRWALL"]["data"])  # Index of outermost ring in main SOL
        self.irwall2 = int(self.nc["IRWALL2"]["data"])  # Second wall ring in double null grid
        self.irtrap = int(self.nc["IRTRAP"]["data"])  # Index of outermost ring in PFZ
        self.irtrap2 = int(self.nc["IRTRAP2"]["data"])  # Index of outermost ring in second PFZ

        # other
        self.qtim = float(self.nc["QTIM"]["data"])  # Time step for ions
        self.kss = self.read_data_2d("KSS")  # S coordinate of cell centers along the field lines
        self.ksmaxs = self.nc["KSMAXS"]["data"]  # S max value for each ring (connection length)

        self.crmb = float(self.nc["CRMB"]["data"])  # Mass of plasma species in amu
        
        # JET grid
        #if self.JET_grid:
        #    self.Z = -self.Z
        #    self.zvertp = -self.zvertp
        #    self.zvesm = -self.zvesm
        # Plasma parameters
        self.ne = self.read_data_2d("KNBS")
        self.ni = self.ne # only one plasma species, no impurities
        self.te = self.read_data_2d("KTEBS")
        self.ti = self.read_data_2d("KTIBS")
        self.vpara = self.read_data_2d("KVHS")/self.qtim

        # Plasma boundary
        self.ne_t = self.read_boundary_data("KNDS")
        self.ni_t = self.ne_t # only one plasma species, no impurities
        self.te_t = self.read_boundary_data("KTEDS")
        self.ti_t = self.read_boundary_data("KTIDS")
        # Neutral species
        self.n0 = self.read_data_2d("PINATO")
        self.n2 = self.read_data_2d("PINMOL")
        self.t0 = self.read_data_2d("PINENA")
        self.t2 = self.read_data_2d("PINENM")

        # Ionization/recombination rates

        self.ioz = self.read_data_2d("PINION")
        self.rec = self.read_data_2d("PINREC")

    def print_keys_to_txt(self):
        with open('output.txt', 'w') as file:
            for key in self.nc.keys():
                try:
                    description = self.nc[key]["long_name"]
                    file.write(f'{key}: {description}\n')
                except:
                    pass

    
    def read_data_2d(self, dataname):
        """
        Reads in 2D data into a 1D array, in a form that is then passed easily
        to PolyCollection for plotting.

        Parameters
        ----------
        dataname : str
            The 2D data as named in the netCDF file.
        Returns
        -------
        data : The data in a 2D format compatible with plot_2d.
        """

        raw_data = self.nc[dataname]["data"]

        # Extract valid data using the mask and flattening as needed
        data = raw_data[self.area > 0.0]

        # Mask out very large values (e.g. fill values)
        data = np.where(data > 1e30, np.nan, data)

        return data

    def read_boundary_data(self, dataname):
        """
        Reads and separates boundary data into two continuous boundary loops:
        main + PFR for each boundary.

        Parameters
        ----------
        dataname : str
            Name of the boundary variable in the NetCDF file.

        Returns
        -------
        bnd1 : np.ndarray
            Data for the first boundary (main + PFR).
        bnd2 : np.ndarray
            Data for the second boundary (main + PFR).
        """
        bdata = np.array(self.nc[dataname]["data"])

        num_main = self.irwall - self.irsep
        num_pfr = self.nrs - self.irtrap

        # Indices of each segment based on assumed order
        i0 = 1                    # first exclude the guard ring
        i1 = i0 + num_main        # end of main1
        i2 = i1 + num_pfr         # end of pfr1
        i3 = i2 + num_pfr         # end of pfr2
        i4 = i3 + num_main        # end of main2

        # Slice each segment
        main1 = bdata[i0:i1]
        pfr1 = bdata[i1:i2]
        pfr2 = bdata[i2:i3]
        main2 = bdata[i3:i4]

        # Combine each boundary loop
        bnd1 = np.concatenate([np.flip(pfr1), np.flip(main1)])
        bnd2 = np.concatenate([pfr2, main2])

        return np.array([bnd1, bnd2])
    
    def get_sep(self):
        """
        Return collection of lines to be plotted with LineCollection method
        of matplotlib for the separatrix.

        Returns
        -------
        lines : List of coordinates to draw the separatrix in a format friendly
                 for LineCollection.
        """

        # Get (R, Z) coordinates of separatrix.
        rsep = self.rvertp[self.korpg[self.irsep - 1, : self.nks[self.irsep - 1]]][:, 0]
        zsep = self.zvertp[self.korpg[self.irsep - 1, : self.nks[self.irsep - 1]]][:, 0]
        nsep = len(rsep)
        lines = []
        

        # Construct separatrix as a series of pairs of coordinates (i.e. a line
        # between the coordinates), to be plotted. Don't connect final point to first.
        for i in range(nsep - 2):
            lines.append([(rsep[i], zsep[i]), (rsep[i + 1], zsep[i + 1])])

        self.sep = lines
        sep_points = list(zip(rsep, zsep))
        self.sep_poly = patches.Polygon(sep_points, closed=False, ec='pink', linestyle='dashed', lw=2.0, fc='None', zorder=10)
        self.shply_sep_poly = Polygon(self.sep_poly.get_xy())
    
    def create_cells(self):
        self.cells = []
        self.patches = []

        for i in range(self.num_cells):
            rv = list(self.rv[i])
            rv.append(self.rv[i][0])
            zv = list(self.zv[i])
            zv.append(self.zv[i][0])
            poly = patches.Polygon(list(zip(rv, zv)))
            self.patches.append(poly)
            shply_poly = Polygon(poly.get_xy())
            self.cells.append(Cell(self.R[i], self.Z[i], row = self.mesh_idxs[i][1], ring = self.mesh_idxs[i][0], poly =shply_poly, 
                                   te = self.te[i], ti = self.ti[i], ne = self.ne[i], ni = self.ni[i], n0 = self.n0[i], n2 = self.n2[i], 
                                   n2p= 0.0 ,Srec= self.rec[i], Sion= self.ioz[i]))

    def create_wall_poly(self):
        Rpts = self.rvesm[0][:self.nvesm]
        Zpts = self.zvesm[0][:self.nvesm]
        wall_poly_pts = list(zip(Rpts, Zpts))
        self.wall_poly = patches.Polygon(wall_poly_pts, closed=False, ec='k', lw=2.0, fc='None', zorder=10)
        self.shply_wall_poly = Polygon(self.wall_poly.get_xy())

