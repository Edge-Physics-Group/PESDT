import numpy as np
from collections import OrderedDict

import struct
import logging
logger = logging.getLogger(__name__)


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def floatToBits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>l', s)[0]


def gaussian(cwl, wv, area, fwhm):
    sigma = fwhm / (2. * np.sqrt(2 * np.log(2.)) )
    g = area * (1./(sigma*np.sqrt(2.*np.pi))) * np.exp(-1.*((wv - cwl)**2) / (2*(sigma**2)) )
    return g

def interp_nearest_neighb(point, neighbs, neighbs_param_pervol):
    """
        Nearest neigbours weighted average of given parameter with per unit volume units
        point = [r,z]
        neighbs = [[r1,z1,[rn,zn]]
        neighbs_param_pervol = [val1, ...valn]

        returns point_param_pervol, the weighted nearest neighbours average
    """
    distances = []
    for neighb in neighbs:
        distances.append(np.sqrt((point[0]-neighb[0])**2 + (point[1]-neighb[1])**2))

    numerator = 0
    denominator = 0
    for i in range(len(neighbs_param_pervol)):
        numerator += distances[i]*neighbs_param_pervol[i]
        denominator += distances[i]

    point_param_pervol = numerator / denominator

    return point_param_pervol

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)