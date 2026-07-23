# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d
import os, ctypes

CONV = 1/(4.0*np.pi) * (1.0e-06) * 10.0

class ContinuumRadiation(ctypes.Structure):
    _fields_ = [
        ("free_free", ctypes.c_double),
        ("free_bound", ctypes.c_double),
    ]

cr_dtype = np.dtype([
    ("free_free", np.float64),
    ("free_bound", np.float64),
], align=True)

lib_root = os.environ.get("CONTINUO_LIB")
if lib_root is None:
    lib_root = os.path.join(os.environ.get('PESDT_HOME', os.path.expanduser('~')), "PESDT/core/utils")

lib = ctypes.CDLL(os.path.join(lib_root, "libcontinuo_.so"))

lib.continuo_.argtypes = [
    ctypes.c_double,  # wavelength_A
    ctypes.c_double,  # Te_eV
    ctypes.c_int,     # atomic_number
    ctypes.c_int,     # ion_charge
]

lib.continuov_.argtypes = [
    ctypes.POINTER(ctypes.c_double),   # wavelength_A
    ctypes.POINTER(ctypes.c_double),   # Te_eV
    ctypes.c_int,                      # atomic_number
    ctypes.c_int,                      # ion_charge
    ctypes.c_size_t,                   # num_wl
    ctypes.c_size_t,                   # num_te
    ctypes.POINTER(ContinuumRadiation) # output
]
lib.continuov_.restype = None

def continuo_(wavelength_A, Te_eV, atomic_number, ion_charge):
    result = lib.continuo_(
        float(wavelength_A),
        float(Te_eV),
        int(atomic_number),
        int(ion_charge),
    )

    return result.free_free*CONV, (result.free_bound+result.free_free)*CONV # Imitate adaslib behaviour

def continuov_(wavelength_A: np.ndarray, Te_eV: np.ndarray, atomic_number: int, ion_charge: int):
    wavelength_A = np.ascontiguousarray(np.atleast_1d(wavelength_A), dtype=np.float64)
    Te_eV = np.ascontiguousarray(np.atleast_1d(Te_eV), dtype=np.float64)
    num_wl =len(wavelength_A); num_te = len(Te_eV)
    output = np.empty(num_wl * num_te, dtype=cr_dtype)

    lib.continuov_(
        wavelength_A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Te_eV.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(atomic_number),
        ctypes.c_int(ion_charge),
        ctypes.c_size_t(num_wl),
        ctypes.c_size_t(num_te),
        output.ctypes.data_as(ctypes.POINTER(ContinuumRadiation))
    )
    output = output.reshape(num_te, num_wl)
    if num_te == 1: output = output.flatten()
    return output["free_free"]*CONV, (output["free_bound"]+output["free_free"])*CONV

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def get_fffb_intensity_ratio_fn_T(wv_lo_nm, wv_hi_nm, Zeff, **kwargs):
    

    Te_rnge = kwargs.get("Te_range", [0.2, 30]) 
    
    print('Processing continuum ratio...')
    wave_nm = np.linspace(wv_lo_nm - 10, wv_hi_nm + 10, 100)
    ilo, vlo = find_nearest(wave_nm, wv_lo_nm)
    ihi, vhi = find_nearest(wave_nm, wv_hi_nm)
    Te_arr = np.logspace(np.log10(Te_rnge[0]), np.log10(Te_rnge[1]), 50)

    intensity_ratio = np.zeros(((np.size(Te_arr)), 2))
    for iTe, vTe in enumerate(Te_arr):
        ff_only, ff_fb_tot = continuov_(wave_nm, vTe, 1, 1)
        intensity_ratio[iTe, 0] = vTe
        intensity_ratio[iTe, 1] = ff_fb_tot[ilo] / ff_fb_tot[ihi]

    Te_fine = np.logspace(np.log10(Te_rnge[0]), np.log10(Te_rnge[1]), 1000)
    f = interp1d(Te_arr, intensity_ratio[:, 1], kind='linear')
    intensity_ratio_interp = np.zeros(((np.size(Te_fine)), 2))
    intensity_ratio_interp[:, 0] = Te_fine
    intensity_ratio_interp[:, 1] = f(Te_fine)
    
    print ('Done')

    return intensity_ratio_interp


if __name__ == '__main__':

    print('continuo_read')
