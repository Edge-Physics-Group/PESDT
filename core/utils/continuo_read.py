# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d
from subprocess import Popen, PIPE
import os, ctypes
import scipy.io as io

class ContinuumRadiation(ctypes.Structure):
    _fields_ = [
        ("free_free", ctypes.c_double),
        ("free_bound", ctypes.c_double),
    ]

cr_dtype = np.dtype([
    ("free_free", np.float64),
    ("free_bound", np.float64),
])


lib = ctypes.CDLL("./libcontinuo_.so")

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

    return result.free_free, result.free_bound+result.free_free # Imitate adaslib behaviour

def continuov_(wavelength_A: np.ndarray, Te_eV: np.ndarray, atomic_number: int, ion_charge: int):
    num_wl =len(wavelength_A); num_te = len(Te_eV)
    output = np.empty(num_wl * num_te, dtype=cr_dtype)
    lib.continuov_(
        float(wavelength_A),
        float(Te_eV),
        int(atomic_number),
        int(ion_charge),
        num_wl,
        num_te,
        output
    )
    output = output.reshape(num_te, num_wl)
    return output["free_free"], output["free_bound"]+output["free_free"]

h = 6.626E-34 # 'J.s'
c = 299792458.0 # 'm/s'

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def get_fffb_intensity_ratio_fn_T(wv_lo_nm, wv_hi_nm, Zeff, save_output=False, restore=False, build32 = False):
    

    Te_rnge = [0.2, 30]
    
    print('Processing continuum ratio...')
    wave_nm = np.linspace(wv_lo_nm - 10, wv_hi_nm + 10, 100)
    ilo, vlo = find_nearest(wave_nm, wv_lo_nm)
    ihi, vhi = find_nearest(wave_nm, wv_hi_nm)
    Te_arr = np.logspace(np.log10(Te_rnge[0]), np.log10(Te_rnge[1]), 50)

    intensity_ratio = np.zeros(((np.size(Te_arr)), 2))
    for iTe, vTe in enumerate(Te_arr):
        ff_only, ff_fb_tot = adas_continuo_py(wave_nm, vTe, 1, 1, build32=build32)
        intensity_ratio[iTe, 0] = vTe
        intensity_ratio[iTe, 1] = ff_fb_tot[ilo] / ff_fb_tot[ihi]

    Te_fine = np.logspace(np.log10(Te_rnge[0]), np.log10(Te_rnge[1]), 1000)
    f = interp1d(Te_arr, intensity_ratio[:, 1], kind='linear')
    intensity_ratio_interp = np.zeros(((np.size(Te_fine)), 2))
    intensity_ratio_interp[:, 0] = Te_fine
    intensity_ratio_interp[:, 1] = f(Te_fine)
    # plt.plot(intensity_ratio_interp[:, 0], intensity_ratio_interp[:, 1])
    # plt.show()
    print ('Done')

    return intensity_ratio_interp

def adas_continuo_py(wave_nm, Te_eV, iz0, iz1, output_in_ph_s=True, build32 = False):
    # ;               NAME      I/O    TYPE    DETAILS
    # ; REQUIRED   :  wave()     I     real    wavelength required (A)
    # ;               tev()      I     real    electron temperature (eV)
    # ;               iz0        I     long    atomic number
    # ;               iz1        I     long    ion stage + 1
    # ;               contff(,)  O     real    free-free emissivity (ph cm3 s-1 A-1)
    # ;               contin(,)  O     real    total continuum emissivity
    # ;                                        (free-free + free-bound) (ph cm3 s-1 A-1)
    # ;                                        dimensions: wave, te (dropped if just 1).

    n_te = np.size(Te_eV)
    n_wv = np.size(wave_nm)

    
    contff, contin = continuo_(wave_nm*10., Te_eV, iz0, iz1)

    contff_ph = (1. / (4 * np.pi)) * contff * (1.0e-06) * 10.0  # ph s-1 m3 sr-1 nm-1
    contin_ph = (1. / (4 * np.pi)) * contin * (1.0e-06) * 10.0  # ph s-1 m3 sr-1 nm-1

    if output_in_ph_s == False:

        wave_m = wave_nm * 1.0e-09
        if n_te > 1:
            contff_W = contff_ph
            contin_W = contin_ph
            for it in range(0, n_te):
                contff_W[:, it] = contff_ph[:, it] * h * c / wave_m  # W m3 sr-1 nm-1
                contin_W[:, it] = contin_ph[:, it] * h * c / wave_m  # W m3 sr-1 nm-1
        else:
            contff_W = contff_ph * h * c / wave_m  # W m3 sr-1 nm-1
            contin_W = contin_ph * h * c / wave_m  # W m3 sr-1 nm-1

        return contff_W, contin_W

    if n_te == 1:
        contff_ph = contff_ph.flatten()
        contin_ph = contin_ph.flatten()

    return contff_ph, contin_ph

if __name__ == '__main__':

    print('continuo_read')
