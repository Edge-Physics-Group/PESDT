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

    def __init__(self, sim_path):
        super().__init__()
        self.sim_path = sim_path

        self.nc = nc.Dataset(self.sim_path)

        self.load_data_from_nc()

    def load_data_from_nc(self):

        return