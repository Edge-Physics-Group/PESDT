import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import numpy as np
import sys, os
from shapely.geometry import Polygon
from .cell import Cell
from .background_plasma import BackgroundPlasma

import logging
import netCDF4 as nc
logger = logging.getLogger(__name__)

class OEDGE(BackgroundPlasma):

    def __init__(self, sim_path, JET_grid = True):
        super().__init__()
        self.sim_path = sim_path
        self.edge_code = "OEDGE"
        self.nc = nc.Dataset(self.sim_path)
        self.JET_grid = JET_grid
        self.load_data_from_nc()

    def load_data_from_nc(self):
        #######################################################
        ### LOAD GEOMETRY, PLASMA, AND NEUTRAL DATA FROM NC ###
        #######################################################
        # Geometry
        self.rs = self.nc["RS"]["data"]  # R coordinante of cell centers
        self.zs = self.nc["ZS"]["data"]  # Z coordinate of cell centers
        self.nrs = int(self.nc["NRS"]["data"])  # Number of rings on the grid
        self.nks = self.nc["NKS"]["data"]  # Number of knots on each ring
        self.nds = self.nc["MAXNDS"]["data"]  # Maximum Number of target elements
        self.area = self.nc["KAREAS"]["data"]  # cell area
        self.korpg = self.nc["KORPG"]["data"]  # IK,IR mapping to polygon index
        self.rvertp = self.nc["RVERTP"]["data"]  # R corner coordinates of grid polygons
        self.zvertp = self.nc["ZVERTP"]["data"]  # Z corner coordinates of grid polygons
        self.rvesm = self.nc["RVESM"]["data"]  # R coordinates of vessel wall segment end points
        self.zvesm = self.nc["ZVESM"]["data"]  # V coordinates of vessel wall segment end points

        # indexes for rings defining the SOL rings
        self.irsep = int(self.nc["IRSEP"]["data"])  # Index of first ring in main SOL
        self.irwall = int(self.nc["IRWALL"]["data"])  # Index of outermost ring in main SOL
        self.irwall2 = int(self.nc["IRWALL2"]["data"])  # Second wall ring in double null grid
        self.irtrap = int(self.nc["IRTRAP"]["data"])  # Index of outermost ring in PFZ
        self.irtrap2 = int(self.nc["IRTRAP2"]["data"])  # Index of outermost ring in second PFZ

        # other
        self.qtim = self.nc["QTIM"]["data"]  # Time step for ions
        self.kss = self.nc["KSS"]["data"]  # S coordinate of cell centers along the field lines
        self.kfizs = self.nc["KFIZS"]["data"]  # Impurity ionization rate
        self.ksmaxs = self.nc["KSMAXS"]["data"]  # S max value for each ring (connection length)

        self.crmb = self.nc["CRMB"]["data"]  # Mass of plasma species in amu
        
        # JET grid
        if self.JET_grid:
            self.zs = -self.zs
            self.zvertp = -self.zvertp
            self.zvesm = -self.zvesm
        # Plasma parameters
        self.ne = self.nc["KNBS"]["data"]
        self.ni = [self.ne] # only one plasma species, no impurities
        self.te = self.nc["KTEBS"]["data"]
        self.ti = self.nc["KTIBS"]["data"]
        self.vpara = self.nc["KHVS"]["data"]/self.nc["QTIM"]["data"]

        # Plasma boundary
        self.ne_t = self.nc["KNDS"]["data"]
        self.ni_t = [self.ne_t] # only one plasma species, no impurities
        self.te_t = self.nc["KTEDS"]["data"]
        self.ti_t = self.nc["KTIDS"]["data"]
        # Neutral species
        self.n0 = self.nc["PINATO"]["data"]
        self.n2 = self.nc["PINMOL"]["data"]
        self.t0 = self.nc["PINENA"]["data"]
        self.t2 = self.nc["PINENM"]["data"]

        # Ionization/recombination rates

        self.ioz = self.nc["PINION"]["data"]
        self.rec = self.nc["PINREC"]["data"]
    
    def format_data():

        return