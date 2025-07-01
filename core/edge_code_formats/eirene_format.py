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

    def __init__(self, sim_path):
        super().__init__()