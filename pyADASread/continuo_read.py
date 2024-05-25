# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d
from subprocess import Popen, PIPE
import os
import scipy.io as io
from adaslib import *
from adaslib.atomic import continuo

h = 6.626E-34 # 'J.s'
c = 299792458.0 # 'm/s'

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def get_fffb_intensity_ratio_fn_T(wv_lo_nm, wv_hi_nm, Zeff, save_output=False, restore=False, build32 = False):
    # TODO: figure out a way to make the path generic and so that it works from both /pyproc and /pyproc/cherab_bridge
    outfile = os.path.expanduser('~') + '/PESDT/pyADASread/' + (str(wv_lo_nm)+'_'+str(wv_hi_nm)+'_adas_continuo_ratio.npy')

    Te_rnge = [0.2, 30]
    if restore == True:
        return np.load(outfile)

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
    if save_output == True:
        np.save(outfile,intensity_ratio_interp)
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

    if build32 ==True:
        # 32bit executable - depracated
        iz0_in = int(iz0)
        iz1_in = int(iz1)

        contff = np.zeros((n_wv, n_te))
        contin = np.zeros((n_wv, n_te))

        for it in range(0, n_te):
            for iw in range(0, n_wv):

                val_ff = 0.0
                val_in = 0.0
                if n_te > 1:
                    tev_in = Te_eV[it]
                else:
                    tev_in = Te_eV
                if n_wv > 1:
                    wave_in = wave_nm[iw] * 10
                else:
                    wave_in = wave_nm * 10

                p = Popen('/home/bloman/python_tools/pyADASread/adas_continuo.exe', stdin=PIPE, stdout=PIPE)
                # encoding/decoding Python 3 style
                input_str = os.linesep.join([str(wave_in), str(tev_in), str(iz0_in), str(iz1_in)])
                out, err = p.communicate(input=input_str.encode())
                out = out.decode()
                out = out.strip('\n')
                outsplit = out.split(' ')

                contff[iw, it] = float(outsplit[3])
                contin[iw, it] = float(outsplit[5])
    else:
        # ADAS reader
        contff, contin = continuo(wave_nm*10., Te_eV, iz0, iz1)

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
