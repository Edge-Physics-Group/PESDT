import os
import pwd
import math
import numpy as np
import json, pprint, pickle
import operator
from functools import reduce
import matplotlib
# matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib import patches, ticker, colors
from collections import OrderedDict
from core import process
from core import get_ADAS_dict
# from pyproc.process import ProcessEdgeSim

from scipy.constants import Planck, speed_of_light

import sys
sys.path[:0]=['/jet/share/lib/python']


font = {'family': 'normal',
        'weight': 'normal',
        'size': 14}
import matplotlib
matplotlib.rc('font', **font)
import matplotlib.font_manager as font_manager
path = '/usr/share/fonts/gnu-free/FreeSans.ttf'
prop = font_manager.FontProperties(fname=path)
matplotlib.rcParams['font.family'] = prop.get_name()
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = prop.get_name()
matplotlib.rc('lines', linewidth=1.2)
matplotlib.rc('axes', linewidth=1.2)
matplotlib.rc('xtick.major', width=1.2)
matplotlib.rc('ytick.major', width=1.2)
matplotlib.rc('xtick.minor', width=1.2)
matplotlib.rc('ytick.minor', width=1.2)

class PhotonToJ:
    """Converts from photon to Jules (copied from /cherab/core/utility/conversion.py)
    """
    conversion_factor = Planck * speed_of_light * 1e9

    @classmethod
    def to(cls, x, wavelength):
        """Direct conversion; wavelength in nm"""
        return x / wavelength * cls.conversion_factor

    @classmethod
    def inv(cls, x, wavelength):
        """Inverse conversion; wavelength in nm"""
        return x * wavelength / cls.conversion_factor


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def getIdlColorTable(tablenum=None, cxs=False, create=False):
    """
    Copied from jetspec.idl_library. Thanks Andy. I did not want to
    introduce a jetspec dependency within pyproc at this point.
    Wrapper to tek_color_wb (the color table of show)

    :param cxs: use cxs space, otherwise ameigs
    :param create: use stored array or query idl via idlbridge
    :return:
    """
    rgb = np.asarray([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 254, 253, 252, 251, 249, 248, 247, 246,
                        244, 243, 241, 240, 238, 237, 235, 233, 231, 230, 228, 226, 224,
                        222, 220, 218, 216, 213, 211, 209, 207, 204, 202, 199, 197, 194,
                        192, 189, 187, 184, 181, 178, 175, 172, 170, 168, 167, 166, 165,
                        164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152,
                        151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139,
                        138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126,
                        125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113,
                        112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100,
                        99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87,
                        86, 85, 85, 84, 84, 88, 91, 94, 97, 100, 103, 107, 110,
                        114, 117, 121, 124, 128, 131, 135, 139, 143, 146, 150, 154, 158,
                        162, 166, 170, 174, 179, 183, 187, 191, 196, 200, 205, 209, 214,
                        218, 223, 228, 233, 237, 242, 247, 252, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255],
                        [253, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242,
                        241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229,
                        228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216,
                        215, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205, 204, 203,
                        202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190,
                        189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177,
                        176, 175, 174, 173, 172, 171, 170, 169, 169, 169, 170, 171, 172,
                        173, 174, 176, 177, 178, 180, 181, 183, 184, 186, 188, 189, 191,
                        193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 214, 216, 218,
                        221, 223, 226, 228, 231, 233, 236, 239, 241, 244, 247, 250, 253,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 252, 247, 242, 237, 232,
                        226, 221, 216, 211, 205, 200, 194, 189, 183, 178, 172, 166, 160,
                        155, 149, 143, 137, 131, 125, 119, 113, 107, 100, 94, 88, 81,
                        75, 69, 62, 56, 49, 42, 36, 29, 29],
                        [253, 253, 252, 251, 250, 249, 248, 248, 247, 247, 246, 246, 245,
                        245, 244, 244, 244, 244, 243, 243, 243, 243, 243, 243, 243, 243,
                        244, 244, 244, 244, 245, 245, 246, 246, 247, 247, 248, 249, 250,
                        250, 251, 252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                        253, 250, 247, 244, 241, 237, 234, 231, 227, 224, 220, 217, 213,
                        210, 206, 203, 199, 195, 191, 187, 183, 179, 175, 171, 167, 163,
                        159, 155, 150, 146, 142, 137, 133, 128, 124, 119, 114, 110, 105,
                        100, 95, 90, 86, 83, 82, 81, 80, 79, 78, 77, 76, 75,
                        74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62,
                        61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
                        48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36,
                        35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
                        22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 13,
                        9, 8, 7, 6, 5, 4, 3, 2, 2]], dtype=np.uint8)

    return rgb



def make_colourmap(ind, red, green, blue, name):
    """

    from: https://github.com/astrolitterbox/DataUtils/blob/master/califa_cmap.py

    :param ind:
    :param red:
    :param green:
    :param blue:
    :param name:
    :return:
    """
    ncolors = 2**8
    newInd = np.arange(0, ncolors)
    r = np.interp(newInd, ind, red, left=None, right=None)
    g = np.interp(newInd, ind, green, left=None, right=None)
    b = np.interp(newInd, ind, blue, left=None, right=None)
    colours = np.transpose(np.asarray((r, g, b)))
    fctab= colours/(ncolors-1)
    cmap = colors.ListedColormap(fctab, name=name, N=None)
    return cmap

def get_idlshow_color_cmap():

    tablenum = 22
    rgb = getIdlColorTable(tablenum)
    ind = np.arange(0,rgb.shape[1])
    return make_colourmap(ind,*rgb, 'idlShow')



class Plot():
    """
        Class for retrieving, reducing and plotting pyproc saved data
    """
    def __init__(self, work_dir, case, plot_dict=None, icase=1):
        self.work_dir = work_dir
        self.case = case
        self.plot_dict = plot_dict
        self.icase = icase

        # Cherab flags
        self.cherab_bridge = False
        self.cherab_reflection = False
        self.cherab_abs_fac_dict = None
        # Read pickled pyproc object
        try:
            with open(self.work_dir + self.case + '/pyproc.2ddata.pkl', 'rb') as f:
                self.__data2d = pickle.load(f)
        except IOError as e:
            raise

        # Read processed synth diag saved data
        try:
            with open(self.work_dir + self.case +  '/pyproc.proc_synth_diag.json', 'r') as f:
            # with open(self.work_dir + self.case +  '/pyproc.synth_diag.json', 'r') as f:
                self.__res_dict = json.load(f)
        except IOError as e:
            raise

        if plot_dict:

            # Read cherab_bridge synth diag save data
            if 'cherab_bridge_results' in plot_dict:
                self.cherab_bridge = plot_dict['cherab_bridge_results']
                if self.cherab_bridge:
                    self.cherab_abs_fac_dict = plot_dict['cherab_abs_factor']
                    cherab_res_file = self.work_dir + self.case + '/cherab.synth_diag.json'
                    if 'cherab_reflections' in plot_dict:
                        self.cherab_reflections = plot_dict['cherab_reflections']
                        if self.cherab_reflections:
                            cherab_res_file = self.work_dir + self.case + '/cherab_refl.synth_diag.json'
                    try:
                        with open(cherab_res_file, 'r') as f:
                            self.__cherab_res_dict = json.load(f)
                    except IOError as e:
                        raise

            # First restore (or re-read) the ADAS_dict
            self.ADAS_dict = get_ADAS_dict(self.work_dir, plot_dict['spec_line_dict'], restore=True)
            for key, val in plot_dict.items():
                if key == 'spec_line_dict_lytrap':
                    self.ADAS_dict_lytrap = get_ADAS_dict(self.work_dir + self.case + '/',
                                                          plot_dict['spec_line_dict_lytrap'],
                                                          restore=True, lytrap=True)
                if key == 'prof_param_defs':
                    self.plot_param_profiles(lineweight=1.5, alpha=0.2500, legend=False)
                if key == 'prof_Hemiss_defs':
                    self.plot_Hemiss_profiles(lineweight=2.0, alpha=0.2500, legend=True,
                                              Watts=False)
                if key == 'prof_impemiss_defs':
                    self.plot_impemiss_profiles(lineweight=3.0, alpha=0.250, legend=True)
                if key == 'prof_Prad_defs':
                    write_ppf = False
                    if 'write_ppf' in val:
                        write_ppf = val['write_ppf']
                    self.plot_Prad_profiles(lineweight=3.0, alpha=0.250, legend=True,
                                            write_ppf=write_ppf)
                if key == 'param_along_LOS':
                    self.plot_params_along_LOS(lineweight=2.0, alpha=0.250, legend=False)
                if key == '2d_param':
                    diagLOS = val['diagLOS']
                    param = val['param']
                    savefig = val['save']
                    Rrng = val['Rrng']
                    Zrng = val['Zrng']
                    if 'max_emiss' in val:
                        max_emiss = val['max_emiss']
                    else:
                        max_emiss = None

                    self.plot_2d_param(param=param, diagLOS=diagLOS, Rrng=Rrng, Zrng=Zrng, max_abs = max_emiss, savefig=savefig)
                if key == '2d_defs':
                    diagLOS = val['diagLOS']
                    savefig = val['save']
                    Rrng = val['Rrng']
                    Zrng = val['Zrng']
                    writecsv=val['writecsv']
                    if 'max_emiss' in val:
                        max_emiss = val['max_emiss']
                    else:
                        max_emiss = None

                    self.plot_2d_ff_fb(diagLOS, Rrng=Rrng, Zrng=Zrng, savefig=savefig, writecsv=writecsv)

                    for at_num in val['lines']:
                        for stage in val['lines'][at_num]:
                            for line in val['lines'][at_num][stage]:
                                self.plot_2d_spec_line(at_num, stage, line, diagLOS, max_abs = max_emiss,
                                                       Rrng=Rrng, Zrng=Zrng, savefig=savefig, writecsv=writecsv)
                if key == 'imp_rad_coeff':
                    self.plot_imp_rad_coeff(val['region'], val['atnum'], val['ion_stages'])
                if key == 'imp_rad_dist':
                    self.plot_imp_rad_dist(val['region'], val['atnum'], val['te_nbins'])
                if key == 'ring_params':
                    self.plot_along_ring(val['params'], val['linestyles'], 
                                         val['colors'],ring=val['ring'])
                if key == '2d_prad':
                    diagLOS = val['diagLOS']
                    savefig = val['save']
                    Rrng = val['Rrng']
                    Zrng = val['Zrng']
                    self.plot_2d_prad(diagLOS, Rrng=Rrng, Zrng=Zrng, savefig=savefig)
                if key == 'nii_adas_afg':
                    self.plot_nii_adas_afg()

    @property
    def data2d(self):
        return self.__data2d

    @property
    def res_dict(self):
        return self.__res_dict

    @property
    def cherab_res_dict(self):
        return self.__cherab_res_dict

    @staticmethod
    def pprint_json(resdict, indent=0):
        for key, value in resdict.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                Plot.pprint_json(value, indent + 1)
            else:
                if isinstance(value, list):
                    print('\t' * (indent+1) + '[list]')
                else:
                    if isinstance(value, str):
                        print('\t' * (indent + 1) + value)
                    else:
                        print('\t' * (indent + 1) + '[float]')

    # get item from nested dict
    @staticmethod
    def get_from_dict(dataDict, mapList):
        try:
            return reduce(operator.getitem, mapList, dataDict)

        except KeyError:
            print('Key in ', mapList, ' doesnt exist.')
            return 0

    def plot_param_profiles(self, linestyle='-', lineweight=2.0,
                            fontsize=16, alpha=0.50, ne_scal=1.e-20, include_Siz_Srec=True, legend=False):
                            # fontsize=20, alpha=0.250, ne_scal=1.0, legend=True):

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PARAMS
        axs = self.plot_dict['prof_param_defs']['axs']
        diag = self.plot_dict['prof_param_defs']['diag']
        color = self.plot_dict['prof_param_defs']['color']
        color2 = self.plot_dict['prof_param_defs']['color']
        zorder = self.plot_dict['prof_param_defs']['zorder']
        coord = self.plot_dict['prof_param_defs']['coord']
        Sion_H_transition = self.plot_dict['prof_param_defs']['Sion_H_transition']
        Srec_H_transition = self.plot_dict['prof_param_defs']['Srec_H_transition']
        ylim, xlim = None, None
        if 'ylim' in self.plot_dict['prof_param_defs']:
            ylim = self.plot_dict['prof_param_defs']['ylim']
        if 'xlim' in self.plot_dict['prof_param_defs']:
            xlim = self.plot_dict['prof_param_defs']['xlim']
        if 'linestyle' in self.plot_dict['prof_param_defs']:
            linestyle = self.plot_dict['prof_param_defs']['linestyle']
        if 'lineweight' in self.plot_dict['prof_param_defs']:
            lineweight = self.plot_dict['prof_param_defs']['lineweight']

        write_csv = False
        if 'write_csv' in self.plot_dict['prof_param_defs']:
            write_csv = self.plot_dict['prof_param_defs']['write_csv']

        if coord == 'R':
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:,0]
        elif coord == 'Z':
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:,1]
        elif coord == 'angle':
            x = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'los_angle'])
        else:
            # default R coord
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:,0]

        true_val_col = color

        # Ne
        ne = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'stark', 'fit', 'ne'])

        if write_csv:
            filedir = self.work_dir + self.case + '/'
            filename = filedir + self.case + '.' + diag + '.' + 'ne' + '.txt'
            header = 'cols: ' + coord + ', ne (m^-3)'
            np.savetxt(filename, np.transpose((x, ne)), header=header, delimiter=',')

        if not isinstance(axs, np.ndarray):
            axs.plot(x, ne_scal*ne, c=color, ls=linestyle, lw=lineweight, zorder=zorder)
        else:
            axs[0].plot(x, ne_scal*ne, c=color, ls=linestyle, lw=lineweight, zorder=zorder)

            # Ne from N II line ratios
            # file = self.work_dir + self.case + '/niiafg_ne.txt'
            # nii_fit_ne = np.genfromtxt(file, skip_header=1)
            # file = self.work_dir + self.case + '/niiafg_tau01_ne.txt'
            # nii_fit_ne_tau01 = np.genfromtxt(file, skip_header=1)
            # file = self.work_dir + self.case + '/niiafg_R.txt'
            # nii_fit_R = np.genfromtxt(file, skip_header=1)
            # axs[0].plot(nii_fit_R, nii_fit_ne, c=color2, lw=lineweight+1, zorder=zorder)
            # axs[0].plot(nii_fit_R, nii_fit_ne_tau01, c=color2, lw=lineweight+1, zorder=zorder)
            # axs[0].fill_between(x, nii_fit_ne, nii_fit_ne_tau01, facecolor=color2,
            #                 edgecolor=color, alpha=alpha, linewidth=0, zorder=zorder)

        if isinstance(axs, np.ndarray) and len(axs) > 1:
        # Te
            # Te_hi = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_360_400'])
            Te_hi = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_400_500'])
            Te_lo = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_300_360'])

            if write_csv:
                filedir = self.work_dir + self.case + '/'
                filename = filedir + self.case + '.' + diag + '.' + 'Te' + '.txt'
                header = 'cols: ' + coord + ', Te_low, Te_high (eV)'
                np.savetxt(filename, np.transpose((x, Te_lo, Te_hi)), header=header, delimiter=',')

            axs[1].semilogy(x, Te_hi, c=color, ls=linestyle, lw=lineweight, zorder=zorder)
            axs[1].plot(x, Te_lo, c=color, ls=linestyle, lw=lineweight, zorder=zorder)
            axs[1].fill_between(x, Te_hi, Te_lo, facecolor=color,
                                edgecolor=color, alpha=alpha, linewidth=0, zorder=zorder)

            # Te from NII line ratios
            # file = self.work_dir + self.case + '/niiafg_te.txt'
            # nii_fit_te = np.genfromtxt(file, skip_header=1)
            # file = self.work_dir + self.case + '/niiafg_tau01_te.txt'
            # nii_fit_te_tau01 = np.genfromtxt(file, skip_header=1)
            # file = self.work_dir + self.case + '/niiafg_R.txt'
            # nii_fit_R = np.genfromtxt(file, skip_header=1)
            # axs[1].plot(nii_fit_R, nii_fit_te, c=color2, lw=lineweight+1, zorder=zorder)
            # axs[1].plot(nii_fit_R, nii_fit_te_tau01, c=color2, lw=lineweight+1, zorder=zorder)
            # axs[1].fill_between(x, nii_fit_te, nii_fit_te_tau01, facecolor=color2,
            #                 edgecolor=color, alpha=alpha, linewidth=0, zorder=zorder)

        if isinstance(axs, np.ndarray) and len(axs) > 2 and include_Siz_Srec:
            # Ionization
            ls = ['-', '--', '.', '-.']
            for itran, tran in enumerate(Sion_H_transition):
                tran_str = 'H' + str(Sion_H_transition[itran][0]) + str(Sion_H_transition[itran][1])
                Sion_adf11 = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'adf11_fit', tran_str, 'Sion'])

                # Scaling factors for CHERAB derived Sion, due to abs. emission discrepancy (TODO: investigate further)
                # Hardwired values as temporary workaround
                if self.cherab_bridge:
                    if tran_str == 'H21':
                        Sion_scal = 115.
                    elif tran_str == 'H32':
                        Sion_scal = 615
                    else:
                        Sion_scal = 1.
                else:
                    Sion_scal = 1.

                print('Sion_adf11, tran_str (s^-1): ', tran_str, np.sum(Sion_adf11) )
                # Total recombination/ionisation (derived from emission with adf11)
                axs[2].semilogy(x, Sion_scal * Sion_adf11, c=color, ls=ls[itran], lw=lineweight, zorder=zorder, label='Sion_'+tran_str)

            for itran, tran in enumerate(Srec_H_transition):
                tran_str = 'H' + str(Srec_H_transition[itran][0]) + str(Srec_H_transition[itran][1])
                Srec_adf11 = self.get_line_int_sorted_data_by_chord_id(diag,
                                                                       ['los_int', 'adf11_fit', tran_str, 'Srec'])

                # Scaling factors for CHERAB derived Srec due to abs. emisSrec discrepancy (TODO: investigate further)
                # Hardwired values as temporary workaround
                if self.cherab_bridge:
                    if tran_str == 'H62':
                        Srec_scal = 349.3
                    elif tran_str == 'H72':
                        Srec_scal = 370.
                    else:
                        Srec_scal = 1.
                else:
                    Srec_scal = 1.

                print('Srec_adf11, tran_str (s^-1): ', np.sum(Srec_adf11))
                # Total recombination/ionisation (derived from emisSrec with adf11)
                # axs[2].semilogy(x, Srec_scal * Srec_adf11, c=color, ls=ls[itran], lw=lineweight, zorder=zorder,
                #                 label='Srec_' + tran_str)


            # axs[2].plot(0, 0, c='k', linewidth=2, label='Ionization')
            # axs[2].plot(0, 0, '--', c='k', linewidth=2, label='Recombination')
            if legend:
                leg = axs[2].legend(loc='upper right', prop={'size':14}, frameon=False, labelspacing=0.05)

            # Plot Sion_adf11 with LyB/Da ratio escape factor method vs without opacity
            # Sion_lytrap_lineratio = np.asarray([4.7257897353678081e+20, 5.3802025298496186e+20, 8.5578014676312929e+20, 1.958580935988641e+21,
            #                                     2.032739614057646e+21, 2.4279939563161445e+21, 2.883631693369264e+21, 4.5697306094182004e+21,
            #                                     8.7224707334947356e+21, 1.7111927552036078e+22, 3.3676408317293405e+22, 9.4791692751852911e+22,
            #                                     2.3727363283138071e+22, 6.8152481982620107e+21, 2.3222702039845718e+21, 1.0588119111867407e+21,
            #                                     5.7096223176716727e+20, 3.0019403872240153e+20, 3.3478481200987885e+20, 2.3340881317069586e+21,
            #                                     1.8457659632979007e+21, 1.3642617546692103e+21])
            #
            # Sion_wo_lytrap = np.asarray([4.7257897353678081e+20, 5.3802025298496186e+20, 8.5578014676312929e+20, 1.958580935988641e+21,
            #                              2.032739614057646e+21, 2.4279939563161445e+21, 2.883631693369264e+21, 4.5697306094182004e+21,
            #                              8.7224707334947356e+21, 1.7111927552036078e+22, 2.4555755186520179e+22, 3.0123796572893939e+22,
            #                              3.1174839253707868e+21, 1.0758010921885586e+21, 5.7227872854819452e+20, 2.9157528656924892e+20,
            #                              1.4755577257289064e+20, 8.3616829422194147e+19, 1.1787849752383744e+20, 2.3340881317069586e+21,
            #                              1.8457659632979007e+21, 1.3642617546692103e+21])
            # axs[2].semilogy(x, Sion_lytrap_lineratio, c=color, lw=lineweight, zorder=zorder)
            # axs[2].semilogy(x, Sion_wo_lytrap, c='m', lw=lineweight, zorder=zorder)


        # if len(axs) > 2:
        #     # Ne from N II line ratios
        #     file = self.work_dir + self.case + '/niiafg_Nconc.txt'
        #     nii_fit_Nconc = np.genfromtxt(file, skip_header=1)
        #     file = self.work_dir + self.case + '/niiafg_tau01_Nconc.txt'
        #     nii_fit_Nconc_tau01 = np.genfromtxt(file, skip_header=1)
        #     file = self.work_dir + self.case + '/niiafg_R.txt'
        #     nii_fit_R = np.genfromtxt(file, skip_header=1)
        #     axs[2].plot(nii_fit_R, nii_fit_Nconc, c=color2, lw=lineweight+1, zorder=zorder)
        #     axs[2].plot(nii_fit_R, nii_fit_Nconc_tau01, c=color2, lw=lineweight+1, zorder=zorder)
        #     axs[2].fill_between(x, nii_fit_Nconc, nii_fit_Nconc_tau01, facecolor=color2,
        #                     edgecolor=color, alpha=alpha, linewidth=0, zorder=zorder)

        if isinstance(axs, np.ndarray) and len(axs) > 3:
            # N0
            abs_corr_fac = 1.0
            if self.cherab_abs_fac_dict:
                for linekey, fac in self.cherab_abs_fac_dict.items():
                    if linekey == '1215.2':
                        abs_corr_fac = fac
            n0delL = abs_corr_fac*self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'n0delL_fit', 'H21', 'n0delL'])

            axs[3].semilogy(x, n0delL, c=color, lw=lineweight, zorder=zorder)

        # plot ne, Te profiles at max ne along LOS
        if self.plot_dict['prof_param_defs']['include_pars_at_max_ne_along_LOS']:
            ne_max, te_max = self.get_param_at_max_ne_along_los(diag, 'te')
            ne_max, n0_max = self.get_param_at_max_ne_along_los(diag, 'n0')
            if len(axs) > 0:
                axs[0].plot(x, ne_scal*ne_max, '-', c=true_val_col, lw=2.0, zorder=5)
            if len(axs) > 1:
                axs[1].plot(x, te_max, '-', c=true_val_col, lw=2.0, zorder=5)
            if len(axs) > 3:
                axs[3].plot(x, n0_max, '-', c=true_val_col, lw=2.0, zorder=5)

        if self.plot_dict['prof_param_defs']['include_target_vals']:
            if len(axs) > 0:
                axs[0].plot(self.data2d.mesh_data.denel_OT['xdata'][:self.data2d.mesh_data.denel_OT['npts']]+self.data2d.mesh_data.osp[0],
                            ne_scal *self.data2d.mesh_data.denel_OT['ydata'][:self.data2d.mesh_data.denel_OT['npts']], 'o', mfc='None',
#                            mec=true_val_col, mew=1.0, ms=5, zorder=1)
                            mec='k', mew=1.0, ms=5, zorder=1)
                # axs[0].plot(self.data2d.mesh_data.denel_IT['xdata'][:self.data2d.mesh_data.denel_IT['npts']]+self.data2d.mesh_data.isp[0],
                #             self.data2d.mesh_data.denel_IT['ydata'][:self.data2d.mesh_data.denel_IT['npts']], 'o', mfc='None',
                #             mec=color, mew=2.0, ms=8)
            if len(axs) > 1:
                axs[1].plot(self.data2d.mesh_data.teve_OT['xdata'][:self.data2d.mesh_data.teve_OT['npts']]+self.data2d.mesh_data.osp[0],
                            self.data2d.mesh_data.teve_OT['ydata'][:self.data2d.mesh_data.teve_OT['npts']], 'o', mfc='None',
#                            mec=true_val_col, mew=1.0, ms=5, zorder=1)
                            mec='k', mew=1.0, ms=5, zorder=1)
                # axs[1].plot(self.data2d.mesh_data.teve_IT['xdata'][:self.data2d.mesh_data.teve_IT['npts']]+self.data2d.mesh_data.isp[0],
                #             self.data2d.mesh_data.teve_IT['ydata'][:self.data2d.mesh_data.teve_IT['npts']], 'o', mfc='None',
                #             mec=color, mew=2.0, ms=8)
            # Ion flux to outer target
            # if len(axs) > 2:
                # axs[2].plot(self.data2d.mesh_data.pflxd_OT['xdata'][:self.data2d.mesh_data.pflxd_OT['npts']]+self.data2d.mesh_data.osp[0],
                #             -1.0*self.data2d.mesh_data.pflxd_OT['ydata'][:self.data2d.mesh_data.pflxd_OT['npts']], 'o', mfc='None',
                #             mec=color, mew=2.0, ms=8)
                # axs[2].plot(self.data2d.mesh_data.pflxd_IT['xdata'][:self.data2d.mesh_data.pflxd_IT['npts']] + self.data2d.mesh_data.osp[0],
                #             -1.0 * self.data2d.mesh_data.pflxd_IT['ydata'][:self.data2d.mesh_data.pflxd_IT['npts']], 'o', mfc='None',
                #             mec=color, mew=2.0, ms=8)
            
            # neutral density
            if len(axs) > 3:
                axs[3].plot(self.data2d.mesh_data.da_OT['xdata'][:self.data2d.mesh_data.da_OT['npts']] + self.data2d.mesh_data.osp[0],
                            self.data2d.mesh_data.da_OT['ydata'][:self.data2d.mesh_data.da_OT['npts']], 'o', mfc='None',
                            mec=true_val_col, mew=2.0, ms=8)

        if self.plot_dict['prof_param_defs']['include_sum_Sion_Srec']:
            Sion = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Sion', 'val'])
            Srec = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Srec', 'val'])
            if len(axs) > 2:
#               axs[2].semilogy(x, Sion, '-', c=true_val_col, lw=2.0, zorder=5)
                axs[2].semilogy(x, Sion, '-', c='k', lw=2.0, zorder=1)
#                axs[2].semilogy(x, -1.0*Srec, '--', c='k', lw=2.0, zorder=1)


            # Output Sion/Srec direct and spectroscopically inferred summations
            R = p2[:, 0]
            inner_idx, = np.where(R < self.__data2d.mesh_data.geom['rpx'])
            outer_idx, = np.where(R >= self.__data2d.mesh_data.geom['rpx'])
            print('')
            print(diag, ' particle balance')
            print('Direct sum of Sion, Srec along LOS [s^-1]')
            print('Total (R < R_xpt) :', np.sum(Sion[inner_idx]), np.sum(Srec[inner_idx]))
            print('Total (R >= R_xpt) :', np.sum(Sion[outer_idx]), np.sum(Srec[outer_idx]))
            # print('adf11 Sion, Srec estimates[s^-1]')
            # print('Total (R < R_xpt) :', np.sum(Sion_adf11[inner_idx]), np.sum(Srec_adf11[inner_idx]))
            # print('Total (R >= R_xpt) :', np.sum(Sion_adf11[outer_idx]), np.sum(Srec_adf11[outer_idx]))
            print('')

        # xpt, isp, osp locations
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].axvline(self.__data2d.mesh_data.geom['rpx'], ls=':', c='k', linewidth=1.0, zorder=1)
                axs[i].axvline(self.__data2d.mesh_data.osp[0], ls=':', c='k', linewidth=1.0, zorder=1)
                axs[i].axvline(self.__data2d.mesh_data.isp[0], ls=':', c='k', linewidth=1.0, zorder=1)

                if ylim:
                    axs[i].set_ylim(ylim[i][0], ylim[i][1])
                if xlim:
                    axs[i].set_xlim(xlim[0], xlim[1])
                else:
                    axs[i].set_xlim(x[0], x[-1])

            axs[len(axs)-1].set_xlabel('Major radius on tile 5 (m)', fontsize=fontsize)
            # axs[len(axs)-1].set_xlabel('R (m)', fontsize=fontsize)
            # axs[0].set_xlabel('R (m)', fontsize=fontsize)
            axs[0].xaxis.set_tick_params(labelsize=fontsize)
            # axs[len(axs)-1].set_xlabel('R (m)', fontsize=fontsize)
            if len(axs) > 0:
                axs[0].set_ylabel(r'$\mathrm{n_{e}\/(10^{20}\/m^{-3})}$', fontsize=fontsize, labelpad=10)
                axs[0].xaxis.set_tick_params(labelsize=fontsize)
                axs[0].yaxis.set_tick_params(labelsize=fontsize)
            if len(axs) > 1:
                axs[1].set_ylabel(r'$\mathrm{T_{e}\/(eV)}}$', fontsize=fontsize, labelpad=10)
                axs[1].xaxis.set_tick_params(labelsize=fontsize)
                axs[1].yaxis.set_tick_params(labelsize=fontsize)
            if len(axs) > 2:
                axs[2].set_ylabel(r'$\mathrm{(s^{-1})}$', fontsize=fontsize, labelpad=10)
                axs[2].xaxis.set_tick_params(labelsize=fontsize)
                axs[2].yaxis.set_tick_params(labelsize=fontsize)
            if len(axs) > 3:
                axs[3].set_ylabel(r'$\mathrm{n_{H}\Delta L\/(m^{-2})}$', fontsize=fontsize, labelpad=10)
                axs[3].xaxis.set_tick_params(labelsize=fontsize)
                axs[3].yaxis.set_tick_params(labelsize=fontsize)
        else:
            axs.axvline(self.__data2d.mesh_data.geom['rpx'], ls=':', c='k', linewidth=1., zorder=1)
            axs.axvline(self.__data2d.mesh_data.osp[0], ls=':', c='k', linewidth=1., zorder=1)
            axs.axvline(self.__data2d.mesh_data.isp[0], ls=':', c='k', linewidth=1., zorder=1)
            axs.set_xlabel('Major radius on tile 5 (m)', fontsize=fontsize)
            # axs.set_xlabel('R (m)', fontsize=fontsize)
            axs.set_ylabel(r'$\mathrm{n_{e}\/(10^{20}\/m^{-3})}$', fontsize=fontsize, labelpad=10)
            axs.xaxis.set_tick_params(labelsize=fontsize)
            axs.yaxis.set_tick_params(labelsize=fontsize)

        title = self.case
        axs[0].set_title(title, rotation=270, x=1.05, y=0.9)
        
        # axes_dict['main'][3].set_ylabel(r'$\mathrm{n_{H}\/(m^{-3})}$')

    def plot_Hemiss_profiles(self, lineweight=2.0, alpha=0.250, legend=False,
                             linestyle='--', scal=1.0,
                             fontsize=14,
                             Watts=False):

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PARAMS
        lines = self.plot_dict['prof_Hemiss_defs']['lines']
        lines_lytrap = None
        if 'lines_lytrap' in self.plot_dict['prof_Hemiss_defs']:
            lines_lytrap = self.plot_dict['prof_Hemiss_defs']['lines_lytrap']
        axs = self.plot_dict['prof_Hemiss_defs']['axs']
        diag = self.plot_dict['prof_Hemiss_defs']['diag']
        color = self.plot_dict['prof_Hemiss_defs']['color']
        zorder = self.plot_dict['prof_Hemiss_defs']['zorder']
        excrec = self.plot_dict['prof_Hemiss_defs']['excrec']
        coord = self.plot_dict['prof_Hemiss_defs']['coord']
        write_csv = False
        if 'write_csv' in self.plot_dict['prof_Hemiss_defs']:
            write_csv = self.plot_dict['prof_Hemiss_defs']['write_csv']
        ylim, xlim = None, None
        if 'ylim' in self.plot_dict['prof_Hemiss_defs']:
            ylim = self.plot_dict['prof_Hemiss_defs']['ylim']
        if 'xlim' in self.plot_dict['prof_Hemiss_defs']:
            xlim = self.plot_dict['prof_Hemiss_defs']['xlim']
        if 'linestyle' in self.plot_dict['prof_Hemiss_defs']:
            linestyle = self.plot_dict['prof_Hemiss_defs']['linestyle']

        if coord == 'R':
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:, 0]
        elif coord == 'Z':
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:, 1]
        elif coord == 'angle':
            x = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'los_angle'])
        else:
            # default R coord
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:, 0]

        for i, line in enumerate(lines.keys()):
            abs_corr_fac = 1.0
            if self.cherab_abs_fac_dict:
                for linekey, fac in self.cherab_abs_fac_dict.items():
                    if linekey == line:
                        abs_corr_fac = fac

            excit = abs_corr_fac*self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'excit'])
            recom = abs_corr_fac*self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'recom'])
            label = '{:5.1f}'.format(float(line)/10.) + ' nm'

            if Watts:
                centre_wav_nm = float(line)/10.
                excit = PhotonToJ.to(excit, centre_wav_nm)
                recom = PhotonToJ.to(recom, centre_wav_nm)
                ylabel = '$\mathrm{W\/m^{-2}\/sr^{-1}}$'
            else:
                # if i==2:
                #     ylabel = '$\mathrm{10^{18}\/ph\/s^{-1}\/m^{-2}\/sr^{-1}}$'
                #     scal = 1e-18
                # else:
                #     ylabel = '$\mathrm{10^{21}\/ph\/s^{-1}\/m^{-2}\/sr^{-1}}$'

                ylabel = '$\mathrm{ph\/s^{-1}\/m^{-2}\/sr^{-1}}$'

            axs[i].plot(x, scal*(excit+recom), 'o', ls=linestyle, lw=lineweight, c=color, zorder=zorder, label=label)
            # axs[i].plot(x, scal * (excit), ls=linestyle, lw=lineweight, c='k', zorder=zorder, label='excit')
            # axs[i].plot(x, scal * (recom), ls=linestyle, lw=lineweight, c='gray', zorder=zorder, label='recom')

            # Write to ppf:
            if False:
                # if line == '6561.9':
                #     import my_ppf
                #     wppf = my_ppf.WritePPF(int(self.case.split('_')[6]), 'QT1', program='writeppf', comment='-')
                #     wppf.writedata('DALH', np.zeros(1), x-259, scal*(excit+recom), 's', 'deg', 'p/m2ssr', 'pyproc Dalpha emission')
                #     wppf.closeppf()

                if line == '6561.9':
                    import my_ppf
                    wppf = my_ppf.WritePPF(int(self.case.split('_')[6]), 'KT1C', program='writeppf', comment='-')
                    wppf.writedata('KL11', np.zeros(1), x-260.65, scal*(excit+recom), 's', 'deg', 'p/m2ssr', 'pyproc KL11 Da emission')
                    wppf.closeppf()

            if excrec:
                # axs[i].plot(x, scal*excit, '--', lw=1, c=color, zorder=zorder, label=label+' excit')
                axs[i].plot(x, scal*recom, ls=(0,(1,5)), lw=lineweight, c=color, zorder=zorder, label=label+' recom')
            if write_csv:
                filedir = self.work_dir + self.case + '/'
                filename = filedir + self.case + '.' + diag + '.' + 'Hemiss' + '.' + '{:5.1f}'.format(float(line)/10.) + 'nm.txt'
                header = 'line: ' + '{:5.1f}'.format(float(line)/10.) + ' nm, ' + 'cols: ' + coord + ', excit (ph/s/m^2/sr), recom (ph/s/m^2/sr)'
                np.savetxt(filename, np.transpose((x,excit,recom)), header=header, delimiter=',')

            if legend:
                leg = axs[i].legend(loc='upper left')
                # leg.get_frame().set_alpha(0.2)
            if i == len(lines.keys())-1:
                axs[i].set_xlabel('Major radius on tile 5 (m)', fontsize=fontsize)
                # axs[i].set_xlabel(coord, fontsize=fontsize)
                # axs[i].set_xlabel(coord)

            axs[i].set_ylabel(ylabel, fontsize=fontsize, labelpad=10)
            # axs[0].set_xlabel('R (m)', fontsize=fontsize)
            # axs[0].xaxis.set_tick_params(labelsize=fontsize)
            axs[0].yaxis.set_tick_params(labelsize=fontsize)
            # axs[1].set_xlabel('R (m)', fontsize=fontsize)
            
        # also plot Ly-series with photon trapping
        if lines_lytrap:
            if '1215.67'in lines_lytrap:
                excit = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1215.67', 'excit'])
                recom = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1215.67', 'recom'])
                Lya = excit + recom
                label = '{:5.1f}'.format(float('1215.67') / 10.) + ' nm w/ ' + 'ad hoc opacity'
                axs[0].plot(x, Lya, ':', lw=2, c=color, zorder=zorder, label=label)
                if excrec:
                    axs[0].plot(x, excit, '--', lw=1, c=color, zorder=zorder, label=label + ' excit')
                    axs[0].plot(x, recom, ':', lw=1, c=color, zorder=zorder, label=label + ' recom')
                leg = axs[0].legend(loc='upper left')
                leg.get_frame().set_alpha(0.2)
            if '6564.57'in lines_lytrap:
                excit = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '6564.57', 'excit'])
                recom = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '6564.57', 'recom'])
                Da = excit + recom
                label = '{:5.1f}'.format(float('6564.57') / 10.) + ' nm w/ ' + 'ad hoc opacity'
                axs[1].plot(x, Da, ':', lw=2, c=color, zorder=zorder, label=label)

                # EXPERIMENTAL ######################################
                if '1025.72' in lines_lytrap:
                    axs_twinx = axs[1].twinx()
                    # Plot Ly-beta/D-alpha ratio
                    f_1_2 = 0.41641
                    f_1_3 = 0.079142
                    Arate_1_3 = 0.5575e08  # s^-1
                    Arate_2_3 = 0.4410e08  # s^-1

                    excit_Lyb = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1025.72', 'excit'])
                    recom_Lyb = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1025.72', 'recom'])
                    Lyb = excit_Lyb + recom_Lyb
                    label = '{:5.1f}'.format(float('1025.72') / 10.) + ' nm w/ ' + 'ad hoc opacity'
                    axs[1].plot(x, Lyb, '-.', lw=2, c=color, zorder=zorder, label=label)
                    axs[2].plot(x, Lyb, '-.', lw=2, c=color, zorder=zorder, label=label)

                    g_tau_Ly_beta = (Arate_2_3/Arate_1_3)*(Lyb/Da)

                    # label = '{:5.1f}'.format(float('6564.57') / 10.) + ' nm; ' + 'ad hoc opacity'
                    axs_twinx.plot(x, g_tau_Ly_beta, '--', lw=2, c=color, zorder=zorder, label=label)
                    axs_twinx.set_ylim(0,2)

                    print(g_tau_Ly_beta)

                    # from python_tools.pyADASutils import opacity_utils
                    # from pyADASread import adas_adf11_read, adas_adf15_read
                    # # KT3 g_tau_Ly_beta from oct1317/seq#1
                    # g_tau_Ly_beta = [0.46726903,  0.49397789,  0.47733701,  0.6263499,   0.66123852,  0.70666998,
                    #                  0.77926284,  0.84191311,  0.91273625,  0.94184994,  0.81984646,  0.5897365,
                    #                  0.36794049,  0.42311035,  0.46638785,  0.45649988,  0.44488203,  0.4756023,
                    #                  0.54201328,  0.90349965,  0.99123816, 0.98872541]
                    # tau_Ly_beta = np.zeros((len(g_tau_Ly_beta)))
                    # g_tau_Ly_alpha = np.zeros((len(g_tau_Ly_beta)))
                    # for i in range(len(g_tau_Ly_beta)):
                    #     g_dum, tau_Ly_beta[i] = opacity_utils.get_opacity_from_escape_factor(g_tau_Ly_beta[i])
                    # tau_Ly_alpha = tau_Ly_beta * f_1_2 / f_1_3
                    # for i in range(len(tau_Ly_alpha)):
                    #     g_tau_Ly_alpha[i] = opacity_utils.calc_escape_factor(tau_Ly_alpha[i])
                    # # axs_twinx.plot(x, g_tau_Ly_beta, '--', lw=2, c=color, zorder=zorder, label=label)
                    # # axs_twinx.plot(x, g_tau_Ly_alpha, '--', lw=2, c=color, zorder=zorder, label=label)
                    #
                    # # Calculate Siz
                    # Siz = []
                    # Te_arr = np.logspace(np.log10(0.2), np.log10(20), 50)
                    # ne_arr = np.logspace(np.log10(1.0e14), np.log10(1.0e15), 10)
                    # ne = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'stark', 'fit', 'ne'])
                    # Te_hi = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_360_400'])
                    # for icoord, coord in enumerate(x):
                    #     print('Escape factor: R =', coord, ', ', g_tau_Ly_beta[icoord])
                    #     # opacity estimate only valid for R = 2.7-2.8 due to uncertainties in reflections
                    #     if coord >= 2.72 and coord <= 2.955:
                    #         adf11_opa = adas_adf11_read.get_adas_H_adf11_suppressed(Te_arr, ne_arr,
                    #                                                                 g_tau_Ly_beta[icoord])
                    #         PEC_dict_opa = adas_adf15_read.get_adas_H_PECs_n5(Te_arr, ne_arr,
                    #                                                           g_tau_Ly_beta[icoord])
                    #         # adf11_opa = adas_adf11_read.get_adas_H_adf11_suppressed(Te_arr, ne_arr,
                    #         #                                                         1.0)
                    #         # PEC_dict_opa = adas_adf15_read.get_adas_H_PECs_n5(Te_arr, ne_arr,
                    #         #                                                   1.0)
                    #     elif coord < 2.72:
                    #         adf11_opa = adas_adf11_read.get_adas_H_adf11_suppressed(Te_arr, ne_arr, 1.0)
                    #         PEC_dict_opa = adas_adf15_read.get_adas_H_PECs_n5(Te_arr, ne_arr, 1.0)
                    #
                    #     else:
                    #         adf11_opa = adas_adf11_read.get_adas_H_adf11_suppressed(Te_arr, ne_arr, 1.0)
                    #         PEC_dict_opa = adas_adf15_read.get_adas_H_PECs_n5(Te_arr, ne_arr, 1.0)
                    #
                    #     dR = x[1]-x[0]
                    #     area = 2. * np.pi * coord * dR
                    #     idxne, ne_val = find_nearest(adf11_opa.ne_arr, 1.0e-06*ne[icoord])
                    #     idxTe, Te_val = find_nearest(adf11_opa.Te_arr, Te_hi[icoord])
                    #     Siz.append(
                    #         4. * np.pi * area * Lya[icoord] * adf11_opa.scd[idxTe, idxne] /
                    #         PEC_dict_opa['1215.67excit'].pec[idxTe, idxne])
                    # print('Siz: ', np.sum(np.asarray(Siz)))
                    # print(Siz)
                    # plt.semilogy(x, Siz, '-k')
                    # Sion_adf11 = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'adf11_fit', 'Sion'])
                    # Sion_sum = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Sion', 'val'])
                    # plt.semilogy(x, Sion_adf11, '-r')
                    # plt.semilogy(x, Sion_sum, '-', c='gray')
                    # plt.show()
                # EXPERIMENTAL ######################################



                if excrec:
                    axs[1].plot(x, excit, '--', lw=1, c=color, zorder=zorder, label=label + ' excit')
                    axs[1].plot(x, recom, ':', lw=1, c=color, zorder=zorder, label=label + ' recom')
                if legend:
                    leg = axs[1].legend(loc='upper left')
                    leg.get_frame().set_alpha(0.2)

        # xpt, isp, osp locations
        for i in range(len(axs)):
            axs[i].axvline(self.__data2d.mesh_data.geom['rpx'], ls=':', c='k', linewidth=1.0, zorder=1)
            axs[i].axvline(self.__data2d.mesh_data.osp[0], ls=':', c='k', linewidth=1.0, zorder=1)
            axs[i].axvline(self.__data2d.mesh_data.isp[0], ls=':', c='k', linewidth=1.0, zorder=1)
            axs[i].xaxis.set_tick_params(labelsize=fontsize)
            axs[i].yaxis.set_tick_params(labelsize=fontsize)
            if ylim:
                axs[i].set_ylim(ylim[i][0], ylim[i][1])
            if xlim:
                axs[i].set_xlim(xlim[0], xlim[1])
            else:
                axs[i].set_xlim(x[0], x[-1])

            if i==0:
                title = self.case
                axs[i].set_title(title, rotation=270, x=1.05, y=0.9)
            if i==len(axs)-1:
                axs[i].set_xlabel('Major radius on tile 5 (m)', fontsize=fontsize)
                axs[i].xaxis.set_tick_params(labelsize=fontsize)
                axs[i].yaxis.set_tick_params(labelsize=fontsize)

    def plot_Prad_profiles(self, lineweight=2.0, alpha=0.250, legend=True,
                           linestyle='-', scal=1,#.e-21,
                           fontsize=14,#20,
                           write_ppf=False,
                           ppf_pulse=90425):

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED Prad
        axs = self.plot_dict['prof_Prad_defs']['axs']
        diag_list = self.plot_dict['prof_Prad_defs']['diag']
        color = self.plot_dict['prof_Prad_defs']['color']
        zorder = self.plot_dict['prof_Prad_defs']['zorder']
        coord = self.plot_dict['prof_Prad_defs']['coord']
        write_csv = False
        if 'write_csv' in self.plot_dict['prof_Prad_defs']:
            write_csv = self.plot_dict['prof_Prad_defs']['write_csv']
        ylim, xlim = None, None
        if 'ylim' in self.plot_dict['prof_Prad_defs']:
            ylim = self.plot_dict['prof_Prad_defs']['ylim']
        if 'xlim' in self.plot_dict['prof_Prad_defs']:
            xlim = self.plot_dict['prof_Prad_defs']['xlim']

        ppf_data_dict = {}
        for diag in diag_list:

            if coord == 'R':
                p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
                x = p2[:, 0]
            elif coord == 'Z':
                p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
                x = p2[:, 1]
            elif coord == 'angle':
                x = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'los_angle'])
            else:
                # default R coord
                p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
                x = p2[:, 0]

            # TODO: add logic for testing whether imp1 and/or imp2 exist
            Prad_ff_perm2 = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Prad_perm2', 'ff'])
            Prad_H_perm2 = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Prad_perm2', 'H'])
            Prad_imp1_perm2 = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Prad_perm2', 'imp1'])
            Prad_imp2_perm2 = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Prad_perm2', 'imp2'])

            ylabel = '$\mathrm{W\/m^{-2}}$'

            if np.ndim(Prad_imp1_perm2) == 2:
                Prad_imp1_perm2_sum = np.sum(Prad_imp1_perm2, axis=1)
                axs.plot(x, Prad_imp1_perm2_sum, ls='-.',
                         lw=lineweight, c=color, zorder=zorder, label='Prad,imp1')
            else:
                Prad_imp1_perm2_sum = Prad_imp1_perm2
            if np.ndim(Prad_imp2_perm2) == 2:
                Prad_imp2_perm2_sum = np.sum(Prad_imp2_perm2, axis=1)
                axs.plot(x, Prad_imp2_perm2_sum, ls=':',
                         lw=lineweight, c=color, zorder=zorder, label='Prad,imp2')
            else:
                Prad_imp2_perm2_sum = Prad_imp2_perm2

            Prad_sum = Prad_ff_perm2 + Prad_H_perm2 + Prad_imp1_perm2_sum + Prad_imp2_perm2_sum

            axs.plot(x, Prad_sum, ls=linestyle,
                     lw=lineweight, c=color, zorder=zorder, label='Prad,total')
            axs.plot(x, Prad_H_perm2, ls='--',
                     lw=lineweight, c=color, zorder=zorder, label='Prad,H')
            axs.plot(x, Prad_ff_perm2, ls='--', lw=0.5,
                     c=color, zorder=zorder, label='Prad,ff')

            ppf_data_dict[diag] = {'x':x, 'data':Prad_sum}

        if write_ppf: # Restricted to KB5V and KB5H
            if diag =='KB5V' or diag=='KB5H':
                self.write_bolo_ppf(ppf_pulse, ppf_data_dict, tstart=52.0)

        if write_csv:
            filedir = self.work_dir + self.case + '/'
            filename = filedir + self.case + '.' + diag + '.' + 'Prad.txt'
            header =  coord + ', Prad_free_free, Prad_H, Prad_imp1, Prad_imp2 (W/m^2)'
            np.savetxt(filename, np.transpose((x,Prad_ff_perm2, Prad_H_perm2, Prad_imp1_perm2_sum, Prad_imp2_perm2_sum)), header=header, delimiter=',')

        if legend:
            leg = axs.legend(loc='upper left')
            # leg.get_frame().set_alpha(0.2)

        # axs.set_xlabel('Major radius on tile 5 (m)', fontsize=fontsize)
        axs.set_xlabel(coord, fontsize=fontsize)

        axs.set_ylabel(ylabel, fontsize=fontsize, labelpad=10)

        # xpt, isp, osp locations
        axs.axvline(self.__data2d.mesh_data.geom['rpx'], ls=':', c='k', linewidth=1.)
        axs.axvline(self.__data2d.mesh_data.osp[0], ls=':', c='k', linewidth=1.)
        axs.axvline(self.__data2d.mesh_data.isp[0], ls=':', c='k', linewidth=1.)
        axs.xaxis.set_tick_params(labelsize=fontsize)
        axs.yaxis.set_tick_params(labelsize=fontsize)
        if ylim:
            axs.set_ylim(ylim[0], ylim[1])
        if xlim:
            axs.set_xlim(xlim[0], xlim[1])
        else:
            axs.set_xlim(x[0], x[-1])

        title = self.case
        axs.set_title(title, rotation=270, x=1.05, y=0.9)

    def plot_nii_adas_afg(self):

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PARAMS
        axs = self.plot_dict['nii_adas_afg']['axs']
        color = self.plot_dict['nii_adas_afg']['color']
        zorder = self.plot_dict['nii_adas_afg']['zorder']
        writecsv = self.plot_dict['nii_adas_afg']['writecsv']

        for diagname, diag in self.__res_dict.items():
            if diagname == 'KT3':
                p2 = self.get_line_int_sorted_data_by_chord_id(diagname, ['chord', 'p2'])
                R = p2[:, 0]

                wav = np.asarray(diag['1']['los_int']['afg_adasn1_kt3b1200']['wave'])
                nii_adas_afg_intensity = self.get_line_int_sorted_data_by_chord_id(
                    diagname, ['los_int', 'afg_adasn1_kt3b1200', 'intensity'])
                for ichord, chord in enumerate(R):
                    axs.semilogy(wav, 0.01+nii_adas_afg_intensity[ichord], '-', c=color, lw=2.)

                if writecsv:
                    # Also write the results to file for Stuart to process
                    filename = self.work_dir + self.case + '/kt3_nii_adas_afg' '.wav'
                    np.savetxt(filename, wav.T, newline='\n')
                    filename = self.work_dir + self.case + '/kt3_nii_adas_afg' + '.coord'
                    np.savetxt(filename, R, newline='\n')
                    filename = self.work_dir + self.case + '/kt3_nii_adas_afg' + '.data'
                    header = 'units: ph s^-1 m^-2 sr^-1 nm^-1'
                    np.savetxt(filename, nii_adas_afg_intensity.T, header=header, delimiter=',', newline='\n')

            leg = axs.legend(loc='upper right')
            # leg.get_frame().set_alpha(0.2)

    def plot_impemiss_profiles(self, lineweight=2.0, alpha=0.250, legend=True):

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PARAMS
        lines = self.plot_dict['prof_impemiss_defs']['lines']
        axs = self.plot_dict['prof_impemiss_defs']['axs']
        diag = self.plot_dict['prof_impemiss_defs']['diag']
        color = self.plot_dict['prof_impemiss_defs']['color']
        zorder = self.plot_dict['prof_impemiss_defs']['zorder']
        excrec = self.plot_dict['prof_impemiss_defs']['excrec']
        coord = self.plot_dict['prof_impemiss_defs']['coord']
        write_csv = False
        if 'write_csv' in self.plot_dict['prof_impemiss_defs']:
            write_csv = self.plot_dict['prof_impemiss_defs']['write_csv']
        ylim, xlim = None, None
        if 'ylim' in self.plot_dict['prof_impemiss_defs']:
            ylim = self.plot_dict['prof_impemiss_defs']['ylim']
        if 'xlim' in self.plot_dict['prof_impemiss_defs']:
            xlim = self.plot_dict['prof_impemiss_defs']['xlim']

        if coord == 'R':
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:, 0]
        elif coord == 'Z':
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:, 1]
        elif coord == 'angle':
            x = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'los_angle'])
        else:
            # default R coord
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:, 0]

        icol = 0
        for at_num in lines.keys():
            if int(at_num) > 1 : # skip hydrogen
                for i, ion_stage in enumerate(lines[at_num].keys()):
                    for line in lines[at_num][ion_stage]:
                        line_wv = float(line) / 10.

                        label = process.at_sym[int(at_num) - 1] + ' ' + \
                                process.roman[int(ion_stage) - 1] + ' ' + '{:5.1f}'.format(
                            line_wv) + ' nm'

                        excit = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'imp_emiss', at_num, ion_stage, line, 'excit'])
                        recom = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'imp_emiss', at_num, ion_stage, line, 'recom'])

                        axs[i].plot(x, excit+recom, '-', lw=lineweight, c=color[icol], zorder=zorder, label=label)
                        if excrec:
                            axs[i].plot(x, excit, '--', lw=1, c=color[icol], zorder=zorder, label=label +' excit')
                            axs[i].plot(x, recom, ':', lw=1, c=color[icol], zorder=zorder, label=label +' recom')
                        if legend:
                            leg = axs[i].legend(loc='upper left')
                            leg.get_frame().set_alpha(0.2)

                        if write_csv:
                            filedir = self.work_dir + self.case + '/'
                            filename = filedir + self.case + '.' + diag + '.' + 'impemiss' + '.' + \
                                       process.at_sym[int(at_num) - 1] + \
                                       process.roman[int(ion_stage) - 1] + '{:5.1f}'.format(line_wv) + 'nm.txt'
                            header = 'line: ' + label +', ' + 'cols: ' + coord + ', excit (ph/s/m^2/sr), recom (ph/s/m^2/sr)'
                            np.savetxt(filename, np.transpose((x, excit, recom)), header=header, delimiter=',')

                icol += 1

        title = self.case
        axs[0].set_title(title, rotation=270, x=1.05, y=0.9)

        # xpt, isp, osp locations
        for i in range(len(axs)):
            axs[i].axvline(self.__data2d.mesh_data.geom['rpx'], ls=':', c='k', linewidth=1.)
            axs[i].axvline(self.__data2d.mesh_data.osp[0], ls=':', c='k', linewidth=1.)
            axs[i].axvline(self.__data2d.mesh_data.isp[0], ls=':', c='k', linewidth=1.)
            if ylim:
                axs[i].set_ylim(ylim[i][0], ylim[i][1])
            if xlim:
                axs[i].set_xlim(xlim[0], xlim[1])
            else:
                axs[i].set_xlim(x[0], x[-1])
            label = '$\mathrm{ph\/s^{-1}\/m^{-2}\/sr^{-1}}$'
            axs[i].set_ylabel(label)
            if i == len(axs)-1:
                # axs[i].set_xlabel('Major radius on tile 5 (m)')
                # axs[i].set_xlabel(coord)
                axs[i].set_xlabel('R (m)')



    def plot_params_along_LOS(self, lineweight=2.0, alpha=0.250, legend=True):

        dummy = 0

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PARAMS
        axs = self.plot_dict['param_along_LOS']['axs']
        lines = self.plot_dict['param_along_LOS']['lines']
        diag = self.plot_dict['param_along_LOS']['diag']
        chord = self.plot_dict['param_along_LOS']['chord']
        color = self.plot_dict['param_along_LOS']['color']
        zorder = self.plot_dict['param_along_LOS']['zorder']
        imp1_den = None
        imp2_den = None

        ylim, xlim = None, None
        if 'ylim' in self.plot_dict['param_along_LOS']:
            ylim = self.plot_dict['param_along_LOS']['ylim']
        if 'xlim' in self.plot_dict['param_along_LOS']:
            xlim = self.plot_dict['param_along_LOS']['xlim']

        if diag in self.__res_dict.keys():
            if chord in self.__res_dict[diag].keys():
                l = self.__res_dict[diag][chord]['los_1d']['l']
                Te = self.__res_dict[diag][chord]['los_1d']['te']
                ne = self.__res_dict[diag][chord]['los_1d']['ne']
                if 'imp2_den' in self.__res_dict[diag][chord]['los_1d']:
                    imp2_den = np.asarray(self.__res_dict[diag][chord]['los_1d']['imp2_den'])

        if isinstance(axs, np.ndarray) and len(axs) > 0:
            # n_e
            axs[0].semilogy(l, ne, c=color, lw=lineweight, zorder=zorder, label=r'$\mathrm{n_{e}}$')

            if imp2_den.any():
                axs[0].semilogy(l, np.sum(imp2_den, axis=1), '--', c=color, lw=lineweight, zorder=zorder, label=r'$\mathrm{n_{N}}$')
                # axs[0].semilogy(l, imp2_den[:,0], ':', c='darkgray', lw=lineweight, zorder=zorder, label=r'$\mathrm{n_{N0+}}$')
                axs[0].semilogy(l, imp2_den[:,1], ':', c=color, lw=lineweight, zorder=zorder, label=r'$\mathrm{n_{N^{1+}}}$')
                # axs[0].semilogy(l, imp2_den[:,2], ':', c='darkgray', lw=lineweight, zorder=zorder, label=r'$\mathrm{n_{N2+}}$')
                # axs[0].semilogy(l, imp2_den[:,3], ':', c='darkgray', lw=lineweight, zorder=zorder, label=r'$\mathrm{n_{N3+}}$')
                # axs[0].semilogy(l, imp2_den[:,4], ':', c='darkgray', lw=lineweight, zorder=zorder, label=r'$\mathrm{n_{N4+}}$')
                # axs[0].semilogy(l, imp2_den[:,5], ':', c='darkgray', lw=lineweight, zorder=zorder, label=r'$\mathrm{n_{N5+}}$')
                # axs[0].semilogy(l, imp2_den[:,6], ':', c='darkgray', lw=lineweight, zorder=zorder, label=r'$\mathrm{n_{N6+}}$')
                # axs[0].semilogy(l, imp2_den[:,7], ':', c='darkgray', lw=lineweight, zorder=zorder, label=r'$\mathrm{n_{N7+}}$')

                axs_twinx = axs[0].twinx()
                axs_twinx.plot(l, np.sum(imp2_den, axis=1)/ne, '-', c='r', lw=lineweight, zorder=zorder, label=r'$\mathrm{c_{N}}$')
                axs_twinx.set_ylabel(r'$\mathrm{c_{N}}$')
                axs_twinx.spines['left'].set_color('r')
                axs_twinx.yaxis.label.set_color('r')
                axs_twinx.tick_params(axis='y', colors='r')
                if dummy==0:
                    axs[0].legend(loc='upper left', prop={'size': 18}, handletextpad=0.1, borderaxespad=0.1,
                                  framealpha=1.0, frameon=True, labelspacing=0.05)
                    leg = axs[0].get_legend()
                    # for i in range(0,len(leg.legendHandles)):
                    #     leg.legendHandles[i].set_color('k')

        # if isinstance(axs, np.ndarray) and len(axs) > 1:
        #     # Nitrogen concentration
        #     if imp2_den.any():
        #         axs[1].plot(l, np.sum(imp2_den, axis=1)/ne, '-', c=color, lw=lineweight, zorder=zorder, label=r'$\mathrm{c_{N}}$')
        #         if dummy==0:
        #             axs[1].legend(loc='upper right', prop={'size': 18}, frameon=True, labelspacing=0.05)
        #             leg = axs[1].get_legend()
        #             for i in range(0,len(leg.legendHandles)):
        #                 leg.legendHandles[i].set_color('k')

        if isinstance(axs, np.ndarray) and len(axs) > 1:
            # N II 399.6 nm emission
            for at_num in lines.keys():
                if int(at_num) == 7:
                    for i, ion_stage in enumerate(lines[at_num].keys()):
                        for line in lines[at_num][ion_stage]:
                            if line == '3996.13':
                                line_wv = float(line) / 10.

                                label = process.at_sym[int(at_num) - 1] + ' ' + \
                                        process.roman[int(ion_stage) - 1] + ' ' + '{:5.1f}'.format(
                                    line_wv) + ' nm'

                                excit = self.__res_dict[diag][chord]['los_1d']['imp_emiss'][at_num][ion_stage][line]['excit']
                                recom = self.__res_dict[diag][chord]['los_1d']['imp_emiss'][at_num][ion_stage][line]['recom']

                                axs[1].plot(l, np.asarray(excit) + np.asarray(recom), '-', lw=lineweight, c=color, zorder=zorder, label=label)
            # if dummy==0:
                # axs[1].legend(loc='upper right', prop={'size': 18}, frameon=True, labelspacing=0.05)
                # leg = axs[1].get_legend()
                # for i in range(0, len(leg.legendHandles)):
                #     leg.legendHandles[i].set_color('k')

        if isinstance(axs, np.ndarray) and len(axs) > 0:
            axs[len(axs)-1].set_xlabel('Distance along LOS (m)')
        if isinstance(axs, np.ndarray) and len(axs) > 0:
            axs[0].set_ylabel(r'$\mathrm{(m^{-3})}$')
        if isinstance(axs, np.ndarray) and len(axs) > 1:
            label = 'N II 399.6 nm $\mathrm{(ph\/s^{-1}\/m^{-2}\/sr^{-1})}$'
            axs[1].set_ylabel(label)

        # DRAW TILE 5 RECTANGLE
        if dummy==0:
            width = np.abs(axs[0].get_xlim()[0] - self.__res_dict[diag][chord]['los_1d']['l'][-1])
            height = np.abs(axs[0].get_ylim()[1] - axs[0].get_ylim()[0])
            rect = patches.Rectangle((self.__res_dict[diag][chord]['los_1d']['l'][-1], axs[0].get_ylim()[0]), width,
                                     height, hatch='\\', edgecolor='none', facecolor='k', alpha=0.35)
            axs[0].add_patch(rect)
            # DRAW TILE 5 RECTANGLE
            width = np.abs(axs[1].get_xlim()[0] - self.__res_dict[diag][chord]['los_1d']['l'][-1])
            height = np.abs(axs[1].get_ylim()[1] - axs[1].get_ylim()[0])
            rect = patches.Rectangle((self.__res_dict[diag][chord]['los_1d']['l'][-1], axs[1].get_ylim()[0]), width,
                                     height, hatch='\\', edgecolor='none', facecolor='k', alpha=0.35)
            axs[1].add_patch(rect)
            # # DRAW TILE 5 RECTANGLE
            # width = np.abs(axs[2].get_xlim()[0] - self.__res_dict[diag][chord]['los_1d']['l'][-1])
            # height = np.abs(axs[2].get_ylim()[1] - axs[2].get_ylim()[0])
            # rect = patches.Rectangle((self.__res_dict[diag][chord]['los_1d']['l'][-1], axs[2].get_ylim()[0]), width,
            #                          height, hatch='\\', edgecolor='none', facecolor='k', alpha=0.35)
            # axs[2].add_patch(rect)

        title = diag + ' R = ' + '{:3.2f}'.format(self.__res_dict[diag][chord]['chord']['p2'][0]) + ' m'

        # axs[0].set_xlim(np.min(l), np.max(l)+0.1)
        axs[0].set_xlim(5, 5.25)
        axs[0].set_title(title)



    def plot_imp_rad_coeff(self, region, atnum, ion_stages):

        axs = self.plot_dict['imp_rad_coeff']['axs']
        color = self.plot_dict['imp_rad_coeff']['color']
        zorder = self.plot_dict['imp_rad_coeff']['zorder']

        if self.data2d.mesh_data.imp1_atom_num or self.data2d.mesh_data.imp2_atom_num:
            if atnum == self.data2d.mesh_data.imp1_atom_num or atnum == self.data2d.mesh_data.imp2_atom_num:
                atnumstr = str(atnum)
                # rad loss coeff not very sensitive to elec. density so choose a sensible value
                ine, vne = find_nearest(self.ADAS_dict['adf11'][atnumstr].ne_arr, 1.0e14)

                # plot ionisation balance radiative loss coeff
                axs[0].loglog(self.ADAS_dict['adf11'][atnumstr].Te_arr,
                              1.0e-06 * self.ADAS_dict['adf11'][atnumstr].ion_bal_pwr['total'][ine, :], '-k',
                              lw=3.0)
                for i, stage in enumerate(ion_stages):
                    axs[0].loglog(self.ADAS_dict['adf11'][atnumstr].Te_arr,
                                  1.0e-06 * self.ADAS_dict['adf11'][atnumstr].ion_bal_pwr['ion'][ine, :, stage-1],
                                  ':', c='k', lw=1.0)
                    axs[i + 1].loglog(self.ADAS_dict['adf11'][atnumstr].Te_arr,
                                      1.0e-06 * self.ADAS_dict['adf11'][atnumstr].ion_bal_pwr['ion'][ine, :, stage-1],
                                      '-', c='k', lw=2.0)

                # plot sim rad loss coeff/pwr for each stage
                imp_radpwr_coeff_collate = []
                imp_radpwr_collate = []
                te_collate = []
                for cell in self.data2d.regions[region].cells:
                    if atnum == self.data2d.mesh_data.imp1_atom_num:
                        imp_radpwr_coeff_collate.append(cell.imp1_radpwr_coeff)
                        imp_radpwr_collate.append(cell.imp1_radpwr)
                    elif atnum == self.data2d.mesh_data.imp2_atom_num:
                        imp_radpwr_coeff_collate.append(cell.imp2_radpwr_coeff)
                        imp_radpwr_collate.append(cell.imp2_radpwr)

                    te_collate.append(cell.te)

                imp_radpwr_coeff_collate_arr = np.asarray(imp_radpwr_coeff_collate)
                imp_radpwr_collate_arr = np.sum(np.asarray(imp_radpwr_collate), axis=1)
                imp_radpwr_collate_arr_max = np.max(imp_radpwr_collate_arr)
                imp_radpwr_collate_arr/= imp_radpwr_collate_arr_max
                te_collate_arr = np.asarray(te_collate)

                axs[0].scatter(te_collate_arr, np.sum(imp_radpwr_coeff_collate_arr, axis=1),
                               s=500*imp_radpwr_collate_arr, c=color, edgecolors='none')
                # axs[0].scatter(te_collate_arr, np.sum(imp_radpwr_coeff_collate_arr, axis=1),
                #                s=10, c=color, edgecolors='none')
                axs[0].set_ylabel(r'$\mathrm{L_{z}}$' + r'$\mathrm{\/(W m^{3})}$')

                for i, stage in enumerate(ion_stages):
                    scale = np.asarray(imp_radpwr_collate)[:, i]
                    scale/=imp_radpwr_collate_arr_max
                    axs[i + 1].scatter(te_collate_arr, imp_radpwr_coeff_collate_arr[:, i],
                                       s=500*scale, c=color,  edgecolors='none')
                    # axs[i + 1].scatter(te_collate_arr, imp_radpwr_coeff_collate_arr[:, i],
                    #                    s=10, c=color,  edgecolors='none')
                    axs[i + 1].set_ylabel(r'$\mathrm{L_{z}\/+}$' + str(stage-1) + r'$\mathrm{\/(W m^{3})}$')

                    if i == len(axs)-2:
                        axs[i + 1].set_xlabel(r'$\mathrm{T_{e}\/(eV)}$')

                axs[0].set_title(self.case + ' ' + process.at_sym[atnum - 1] + ' in region: ' + region, fontsize=10)

    def plot_imp_rad_dist(self, region, atnum, te_nbins):

        axs = self.plot_dict['imp_rad_dist']['axs']
        color = self.plot_dict['imp_rad_dist']['color']
        zorder = self.plot_dict['imp_rad_dist']['zorder']
        if 'norm' in self.plot_dict['imp_rad_dist']:
            norm = self.plot_dict['imp_rad_dist']['norm']
        else:
            norm = None
        if 'ion_stage' in self.plot_dict['imp_rad_dist']:
            ion_stage = self.plot_dict['imp_rad_dist']['ion_stage']
        else:
            ion_stage = None

        if self.data2d.mesh_data.imp1_atom_num or self.data2d.mesh_data.imp2_atom_num:
            if atnum == self.data2d.mesh_data.imp1_atom_num or atnum == self.data2d.mesh_data.imp2_atom_num:
                atnumstr = str(atnum)
                # rad loss coeff not very sensitive to elec. density so choose a sensible value
                ine, vne = find_nearest(self.ADAS_dict['adf11'][atnumstr].ne_arr, 1.0e14)

                # Get max and min Te in region for Te bin range
                min_Te = 100000.
                max_Te = 0.0
                for cell in self.data2d.regions[region].cells:
                    if cell.te > max_Te: max_Te = cell.te
                    if cell.te < min_Te: min_Te = cell.te

                # Set up elec. temp bins and labels
                te_bins = np.logspace(np.log10(min_Te), np.log10(max_Te), te_nbins)
                te_bin_labels = []

                for ite, vte in enumerate(te_bins):
                    if (ite + 1) != len(te_bins):
                        label = '{:6.1f}'.format(te_bins[ite]) + '-' + '{:6.1f}'.format(te_bins[ite + 1])
                        te_bin_labels.append(label)

                        # BIN RADIATED POWER BY TE
                        te_bin_imp_radpwr = np.zeros((te_nbins-1, atnum))
                        te_bin_H_radpwr = np.zeros((te_nbins-1))
                        for cell in self.data2d.regions[region].cells:
                            for ite, vte in enumerate(te_bins):
                                if (ite + 1) != len(te_bins):
                                    if cell.te > te_bins[ite] and cell.te <= te_bins[ite + 1]:
                                        te_bin_H_radpwr[ite] += cell.H_radpwr
                                        if atnum == self.data2d.mesh_data.imp1_atom_num:
                                            te_bin_imp_radpwr[ite] += cell.imp1_radpwr
                                        elif atnum == self.data2d.mesh_data.imp2_atom_num:
                                            te_bin_imp_radpwr[ite] += cell.imp2_radpwr
                        # convert to MW
                        te_bin_imp_radpwr *= 1.0e-06
                        te_bin_H_radpwr *= 1.0e-06

                # IMP CHARGE STATE DIST
                axs[0].plot(np.where(np.sum(te_bin_imp_radpwr, axis=0))[0],
                   np.sum(te_bin_imp_radpwr, axis=0), '-o', c=color, mfc=color, mec=color, ms=4, mew=2.0)
                axs[0].set_ylabel(r'$\mathrm{P_{RAD}\/(MW)}$')
                axs[0].set_xlabel('Ionisation stage')

                # BAR PLOT BY TE BINS

                if ion_stage:
                    if int(ion_stage)>=0 and int(ion_stage) < atnum:
                        te_bin_imp_radpwr_plot = te_bin_imp_radpwr[:, int(ion_stage)]
                        label=process.at_sym[atnum - 1] +  str(int(ion_stage)) + '+'
                else:
                    te_bin_imp_radpwr_plot = np.sum(te_bin_imp_radpwr, axis=1)
                    label = process.at_sym[atnum - 1]

                if norm:
                    norm_scal_H = te_bin_H_radpwr / np.sum(te_bin_H_radpwr)
                    norm_scal_imp = te_bin_imp_radpwr_plot / np.sum(te_bin_imp_radpwr_plot)#
                else:
                    norm_scal_H = 1.0
                    norm_scal_imp = 1.0


                x_pos = np.arange(len(te_bin_labels))
                width = 0.2
                barspace = width * self.icase
                axs[1].bar(x_pos+barspace, norm_scal_H*te_bin_H_radpwr, width, align='center', color='darkgrey', edgecolor=color, alpha=0.3)
                axs[1].bar(x_pos+barspace, norm_scal_imp*np.sum(te_bin_imp_radpwr, axis=1) , width, align='center', color=color, alpha=0.3, label=label)
                # axs[1].bar(x_pos+barspace, te_bin_imp_radpwr[:,2] , width, align='center', color=color, alpha=0.3)
                axs[1].set_xticks(x_pos+width*(self.icase-1))
                axs[1].set_xticklabels(te_bin_labels, rotation=90)
                axs[1].set_xlabel(r'$\mathrm{T_{e}\/(eV)}$')
                axs[1].set_ylabel(r'$\mathrm{P_{RAD}\/(MW)}$')
                axs[1].set_title(label, loc = 'right')

                axs[0].set_title(self.case + ' ' + process.at_sym[atnum - 1] + ' in region: ' + region)

    def get_line_int_sorted_data_by_chord_id(self, diag, mapList):
        """
            input:
                mapList: list of dict keys below the 'chord' level (e.g., ['los_int', 'stark', 'fit', 'ne']
                diag: synth diag name string
        """

        if self.cherab_bridge:
            loc_res_dict = self.__cherab_res_dict[diag]
        else:
            loc_res_dict = self.__res_dict[diag]

        tmp = []
        chordidx = []
        for chord in loc_res_dict:
            parval= Plot.get_from_dict(loc_res_dict[chord], mapList)
            tmp.append(parval)
            chordidx.append(int(chord)-1)

        chords = np.asarray(chordidx)
        sort_idx = np.argsort(chords, axis=0)
        sorted_parvals = np.asarray(tmp)[sort_idx]

        return sorted_parvals

    def plot_2d_spec_line(self, at_num, ion_stage, line_key, diagLOS, Rrng=None, Zrng=None,
                          min_clip=0.0, max_clip = 1.0, max_abs = None, scal_log10=False, savefig=False,
                          writecsv=False):

        idlcmap = get_idlshow_color_cmap()
        bg = patches.Rectangle((Rrng[0], Zrng[0]),
                               Rrng[1]-Rrng[0], Zrng[1]-Zrng[0], fill=True, color='darkgrey', zorder=1)

        fig, ax = plt.subplots(ncols=1, figsize=(8, 8.5))
        fig.patch.set_facecolor('white')
        ax.add_patch(bg)
        ax.set_ylabel('Z (m)')
        ax.set_xlabel('R (m)')

        if Rrng and Zrng:
            ax.set_xlim(Rrng[0], Rrng[1])
            ax.set_ylim(Zrng[0], Zrng[1])
        else:
            ax.set_xlim(1.8, 4.0)
            ax.set_ylim(-2.0, 2.0)

        cell_patches = []
        spec_line = []
        polygon_coords = [] # For outputting points to file
        for cell in self.__data2d.cells:
            patch = patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=10)
            cell_patches.append(patch)
            polygon_coords.append(patch.get_path().vertices)
            if int(at_num) > 1:
                spec_line.append(cell.imp_emiss[at_num][ion_stage][line_key]['excit'] +
                                cell.imp_emiss[at_num][ion_stage][line_key]['recom'])
            else:
                spec_line.append(cell.H_emiss[line_key]['excit'] +
                                cell.H_emiss[line_key]['recom'])

            # imp_line.append((cell.imp_emiss[at_num][ion_stage][line_key]['fPEC_excit']+cell.imp_emiss[at_num][ion_stage][line_key]['fPEC_recom'])*cell.ne)


        if writecsv:
            label = at_num +'_' + ion_stage + '_' + line_key
            filename = self.work_dir + self.case + '/' + label +'.polys'
            np.save(filename+'.npy', polygon_coords)
            filename = self.work_dir + self.case + '/' + label +'.intensity'
            np.save(filename+'.npy', spec_line)
            np.savetxt(filename+'.csv', spec_line, newline='\n')


        # Clip color scale with min and max threshold
        if scal_log10:
            cscale_min = min_clip * np.max(np.log10(spec_line))
            cscale_max = max_clip * np.max(np.log10(spec_line))
        else:
            cscale_min = min_clip * np.max(spec_line)
            cscale_max = max_clip * np.max(spec_line)
            if max_abs:
                cscale_max = max_abs

        spec_line_clipped = []
        for i, cell_emiss in enumerate(spec_line):
            if scal_log10:
                cell_emiss=np.log10(cell_emiss)
            if cell_emiss >= cscale_min and cell_emiss <= cscale_max:
                spec_line_clipped.append(cell_emiss)
            elif cell_emiss >= cscale_max:
                spec_line_clipped.append(cscale_max)
            else:
                spec_line_clipped.append(0.0)

        # coll1 = PatchCollection(cell_patches, cmap=matplotlib.cm.jet, norm=matplotlib.colors.LogNorm(), zorder=1, lw=0)
        # coll1 = PatchCollection(cell_patches, cmap=matplotlib.cm.jet, zorder=1, lw=0)
        coll1 = PatchCollection(cell_patches, zorder=10)
        # coll1.set_array(np.asarray(spec_line_clipped))

        colors = idlcmap(spec_line_clipped / np.max(spec_line_clipped))

        coll1.set_color(colors)
        collplt = ax.add_collection(coll1)
        # ax.set_yscale
        line_wv = float(line_key) / 10.
        title = self.case + ' ' + process.at_sym[int(at_num) - 1] + ' ' + process.roman[int(ion_stage) - 1] + ' ' + '{:5.1f}'.format(
            line_wv) + ' nm'
        # title = process.at_sym[int(at_num) - 1] + ' ' + process.roman[int(ion_stage) - 1] + ' ' + '{:5.1f}'.format(
        #     line_wv) + ' nm'
        ax.set_title(title, y=1.08, fontsize='16')
        plt.gca().set_aspect('equal', adjustable='box')

        # ADD COLORBAR
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', size='7%', pad=0.1)

        # Very ugly workaround to scale the colorbar without clobbering the patch collection plot
        # (https://medium.com/data-science-canvas/way-to-show-colorbar-without-calling-imshow-or-scatter)
        sm = plt.cm.ScalarMappable(cmap=idlcmap,
                                   norm=plt.Normalize(vmin=cscale_min,
                                                      vmax=cscale_max))
        sm._A = []
        cbar = fig.colorbar(sm, cax=cbar_ax)
        label = '$\epsilon\mathrm{\/(ph\/s^{-1}\/m^{-3}\/sr^{-1})}$'
        cbar.set_label(label)

        # ADD DIAG LOS
        if diagLOS:
            for diag in diagLOS:
                self.__data2d.synth_diag[diag].plot_LOS(ax, color='red', lw=1.0)

        # PLOT SEPARATRIX AND WALL
        # Work around for this error message:
        # RuntimeError: Can not put single artist in more than one figure
        from copy import copy
        wallpoly = copy(self.__data2d.mesh_data.wall_poly)
        seppoly = copy(self.__data2d.mesh_data.sep_poly)
        wallpoly2 = patches.Polygon(wallpoly.get_xy(), closed=True, edgecolor=None, facecolor='w', zorder=2)
        seppoly2 = patches.Polygon(seppoly.get_xy(), closed=False, facecolor='None', edgecolor='k', zorder=10, lw=1)

        ax.add_patch(wallpoly2)
        ax.add_patch(seppoly2)

        ax.set_zorder(10)

        if savefig:
            plt.savefig(self.work_dir + self.case + '/' + title + '.pdf', dpi=plt.gcf().dpi)

    def plot_2d_prad(self, diagLOS, Rrng=None, Zrng=None,
                          min_clip=0.0, max_clip = 0.4, savefig=False):

        idlcmap = get_idlshow_color_cmap()
        bg = patches.Rectangle((Rrng[0], Zrng[0]),
                               Rrng[1]-Rrng[0], Zrng[1]-Zrng[0], fill=True, color='darkgrey', zorder=1)

        fig, ax = plt.subplots(ncols=1, figsize=(8, 5.5))
        fig.patch.set_facecolor('white')
        ax.add_patch(bg)
        ax.set_ylabel('Z (m)')
        ax.set_xlabel('R (m)')

        if Rrng and Zrng:
            ax.set_xlim(Rrng[0], Rrng[1])
            ax.set_ylim(Zrng[0], Zrng[1])
        else:
            ax.set_xlim(1.8, 4.0)
            ax.set_ylim(-2.0, 2.0)

        cell_patches = []
        prad = []
        for cell in self.__data2d.cells:
            cell_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=10))
            prad.append(cell.ff_radpwr_perm3+
                        np.sum(cell.imp2_radpwr_perm3)+
                        np.sum(cell.imp1_radpwr_perm3)+
                        cell.H_radpwr_perm3)
#            prad.append(np.sum(cell.imp2_radpwr_perm3))
            # prad.append(cell.H_radpwr_perm3)


            # imp_line.append((cell.imp_emiss[at_num][ion_stage][line_key]['fPEC_excit']+cell.imp_emiss[at_num][ion_stage][line_key]['fPEC_recom'])*cell.ne)

        prad=np.asarray(prad)*1.0e-06

        # coll1 = PatchCollection(cell_patches, cmap=matplotlib.cm.hot, norm=matplotlib.colors.LogNorm(), zorder=1, lw=0)
        # coll1 = PatchCollection(cell_patches, cmap=matplotlib.cm.hot, zorder=1, lw=0)
        coll1 = PatchCollection(cell_patches, zorder=10)
        # coll1.set_array(np.asarray(imp_line))

        # Clip color scale with min and max threshold
        cscale_min = min_clip * np.max(prad)
        cscale_max = max_clip * np.max(prad)

        prad_clipped = []
        for i, cell_emiss in enumerate(prad):
            if cell_emiss >= cscale_min and cell_emiss <= cscale_max:
                prad_clipped.append(cell_emiss)
            elif cell_emiss >= cscale_max:
                prad_clipped.append(cscale_max)
            else:
                prad_clipped.append(0.0)
        colors = idlcmap(prad_clipped / np.max(prad_clipped))

        coll1.set_color(colors)
        collplt = ax.add_collection(coll1)
        ax.set_yscale
        title = self.case + ' Prad_tot'
#        title = self.case + ' Prad_imp2'

        ax.set_title(title)
        plt.gca().set_aspect('equal', adjustable='box')

        # ADD COLORBAR
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', size='5%', pad=0.3)

        # Very ugly workaround to scale the colorbar without clobbering the patch collection plot
        # (https://medium.com/data-science-canvas/way-to-show-colorbar-without-calling-imshow-or-scatter)
        sm = plt.cm.ScalarMappable(cmap=idlcmap,
                                   norm=plt.Normalize(vmin=cscale_min,
                                                      vmax=cscale_max))

        sm._A = []
        cbar = fig.colorbar(sm, cax=cbar_ax)
        label = r'$\epsilon\ [\mathrm{MW/m^3}]$'

        cbar.set_label(label)
        cbar.locator = ticker.MaxNLocator(nbins=5)
        cbar.update_ticks()
        cbar.ax.tick_params(axis='y', direction='out')

        # ADD DIAG LOS
        if diagLOS:
            for diag in diagLOS:
                self.__data2d.synth_diag[diag].plot_LOS(ax, color='darkgrey', lw=1.0)

        # PLOT SEPARATRIX AND WALL
        # Work around for this error message:
        # RuntimeError: Can not put single artist in more than one figure
        from copy import copy
        wallpoly = copy(self.__data2d.mesh_data.wall_poly)
        seppoly = copy(self.__data2d.mesh_data.sep_poly)
        wallpoly2 = patches.Polygon(wallpoly.get_xy(), closed=True, edgecolor=None, facecolor='w', zorder=2)
        seppoly2 = patches.Polygon(seppoly.get_xy(), closed=False, facecolor='None', edgecolor='k', zorder=10, lw=1)

        ax.add_patch(wallpoly2)
        ax.add_patch(seppoly2)

        ax.set_zorder(10)

        ax.tick_params(axis='both', labelcolor='k', top='off', bottom='off', left='off', right='off')


        if savefig:
            plt.savefig(self.work_dir + self.case + '/' + title + '.png', dpi=plt.gcf().dpi)


    def plot_2d_param(self, param='ne', diagLOS=None, Rrng=None, Zrng=None, min_clip=0.0, max_clip =1.0,
                      max_abs=None, savefig=False, writecsv=False):
        # Allowd params: ne, te, ni, ti, Sion, Srec

        idlcmap = get_idlshow_color_cmap()
        bg = patches.Rectangle((Rrng[0], Zrng[0]),
                               Rrng[1]-Rrng[0], Zrng[1]-Zrng[0], fill=True, color='darkgrey', zorder=1)

        fig, ax = plt.subplots(ncols=1, figsize=(6, 4.5))
        fig.patch.set_facecolor('white')
        ax.add_patch(bg)
        ax.set_ylabel('Z (m)')
        ax.set_xlabel('R (m)')

        if Rrng and Zrng:
            ax.set_xlim(Rrng[0], Rrng[1])
            ax.set_ylim(Zrng[0], Zrng[1])
        else:
            ax.set_xlim(1.8, 4.0)
            ax.set_ylim(-2.0, 2.0)

        cell_patches = []
        _param = []
        polygon_coords = [] # For outputting points to file
        for cell in self.__data2d.cells:
            patch = patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=10)
            cell_patches.append(patch)
            if param == 'ne':
                _param.append(cell.ne)
                label = '$\mathrm{m^{-3}}$'
            elif param == 'n0':
                _param.append(cell.n0)
                label = '$\mathrm{m^{-3}}$'
            elif param == 'te':
                _param.append(cell.te)
                label = '$\mathrm{eV}$'
            elif param == 'Sion':
                _param.append(cell.Sion)
                label = '$\mathrm{s^{-1}}$'
            elif param == 'Srec':
                _param.append(cell.Srec)
                label = '$\mathrm{s^{-1}}$'
            elif param == 'imp2_radpwr_1+':
                _param.append(cell.imp2_radpwr_perm3[1])
                label = '$\mathrm{W\/m^{-3}})$'
            elif param == 'imp2_radpwr_2+':
                _param.append(cell.imp2_radpwr_perm3[2])
                label = '$\mathrm{W\/m^{-3}})$'
            elif param == 'imp2_radpwr_3+':
                _param.append(cell.imp2_radpwr_perm3[3])
                label = '$\mathrm{W\/m^{-3}})$'
            elif param == 'imp2_radpwr_4+':
                _param.append(cell.imp2_radpwr_perm3[4])
                label = '$\mathrm{W\/m^{-3}})$'
            elif param == 'imp2_radpwr_5+':
                _param.append(cell.imp2_radpwr_perm3[5])
                label = '$\mathrm{W\/m^{-3}})$'
            elif param == 'imp2_radpwr_5+':
                _param.append(cell.imp2_radpwr_perm3[6])
                label = '$\mathrm{W\/m^{-3}})$'
            elif param == 'cNtot':
#                _param.append(np.sum(cell.imp2_den[0:])/cell.ne)
                _param.append((cell.imp2_den[4]/cell.ne) / (np.sum(cell.imp2_den[0:])/cell.ne))
#                _param.append((np.sum(cell.imp2_den[0:])/cell.ne) / (np.sum(cell.imp2_den[0:])/cell.ne))
#                _param.append(cell.imp2_den[1])
                label = '$\mathrm{units}$'

            polygon_coords.append(patch.get_path().vertices)

        # Clip color scale with min and max threshold
        cscale_min = min_clip * np.max(_param)
        cscale_max = max_clip * np.max(_param)
        if max_abs:
            cscale_max = max_abs

        spec_line_clipped = []
        for i, cell_emiss in enumerate(_param):
            if cell_emiss >= cscale_min and cell_emiss <= cscale_max:
                spec_line_clipped.append(cell_emiss)
            elif cell_emiss >= cscale_max:
                spec_line_clipped.append(cscale_max)
            else:
                spec_line_clipped.append(0.0)
        # colors = plt.cm.hot(spec_line_clipped / np.max(spec_line_clipped))
        colors = idlcmap(spec_line_clipped / np.max(spec_line_clipped))

        coll1 = PatchCollection(cell_patches, zorder=9)

        coll1.set_color(colors)
        collplt = ax.add_collection(coll1)
        ax.set_yscale
        title = param
        ax.set_title(title)
        plt.gca().set_aspect('equal', adjustable='box')

        # ADD COLORBAR
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', size='7%', pad=0.1)

        # Very ugly workaround to scale the colorbar without clobbering the patch collection plot
        # (https://medium.com/data-science-canvas/way-to-show-colorbar-without-calling-imshow-or-scatter)
        sm = plt.cm.ScalarMappable(cmap=idlcmap,
                                   norm=plt.Normalize(vmin=cscale_min,
                                                      vmax=cscale_max))
        sm._A = []

        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(label)

        # ADD DIAG LOS
        if diagLOS:
            for diag in diagLOS:
                self.__data2d.synth_diag[diag].plot_LOS(ax, color='darkgrey', lw=1.0)

        # PLOT SEPARATRIX AND WALL
        # Work around for this error message:
        # RuntimeError: Can not put single artist in more than one figure
        from copy import copy
        wallpoly = copy(self.__data2d.mesh_data.wall_poly)
        seppoly = copy(self.__data2d.mesh_data.sep_poly)
        wallpoly2 = patches.Polygon(wallpoly.get_xy(), closed=True, edgecolor=None, facecolor='w', zorder=2)
        seppoly2 = patches.Polygon(seppoly.get_xy(), closed=False, facecolor='None', edgecolor='k', zorder=10, lw=1)

        ax.add_patch(wallpoly2)
        ax.add_patch(seppoly2)

        ax.set_zorder(10)

        if savefig:
            plt.savefig(self.work_dir + self.case + '/' + title + '.png', dpi=plt.gcf().dpi)


    def plot_2d_ff_fb(self, diagLOS, Rrng=None, Zrng=None, min_clip=0.0, max_clip = 1.,
                      savefig=False, writecsv=False):

        idlcmap = get_idlshow_color_cmap()
        bg = patches.Rectangle((Rrng[0], Zrng[0]),
                               Rrng[1]-Rrng[0], Zrng[1]-Zrng[0], fill=True, color='darkgrey', zorder=1)

        fig, ax = plt.subplots(ncols=1, figsize=(6, 4.5))
        fig.patch.set_facecolor('white')
        ax.add_patch(bg)
        ax.set_ylabel('Z (m)')
        ax.set_xlabel('R (m)')

        if Rrng and Zrng:
            ax.set_xlim(Rrng[0], Rrng[1])
            ax.set_ylim(Zrng[0], Zrng[1])
        else:
            ax.set_xlim(1.8, 4.0)
            ax.set_ylim(-2.0, 2.0)

        cell_patches = []
        ff_fb_emiss = []
        polygon_coords = [] # For outputting points to file
        for cell in self.__data2d.cells:
            patch = patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=10)
            cell_patches.append(patch)
            ff_fb_emiss.append(cell.ff_fb_filtered_emiss['ff_fb'])
            polygon_coords.append(patch.get_path().vertices)

        if writecsv:
            filename = self.work_dir + self.case + '/ff_fb_2demiss.polys.npy'
            np.save(filename, polygon_coords)
            filename = self.work_dir + self.case + '/ff_fb_2demiss.intensity'
            np.save(filename+'.npy', ff_fb_emiss)
            np.savetxt(filename+'.csv', ff_fb_emiss, newline='\n')

        # Clip color scale with min and max threshold
        cscale_min = min_clip * np.max(ff_fb_emiss)
        cscale_max = max_clip * np.max(ff_fb_emiss)

        spec_line_clipped = []
        for i, cell_emiss in enumerate(ff_fb_emiss):
            if cell_emiss >= cscale_min and cell_emiss <= cscale_max:
                spec_line_clipped.append(cell_emiss)
            elif cell_emiss >= cscale_max:
                spec_line_clipped.append(cscale_max)
            else:
                spec_line_clipped.append(0.0)
        colors = idlcmap(spec_line_clipped / np.max(spec_line_clipped))

        coll1 = PatchCollection(cell_patches, zorder=10)
        # colors = plt.cm.hot(ff_fb_emiss / np.max(ff_fb_emiss))

        coll1.set_color(colors)
        collplt = ax.add_collection(coll1)
        ax.set_yscale
        # title = self.case + ' Bremss. (ff+fb) 400.96 nm'
        # title = ' Bremss. (ff+fb) 400.96 nm'
        title = ' Bremss. (ff+fb) 400.96 nm'
        ax.set_title(title)
        plt.gca().set_aspect('equal', adjustable='box')

        # ADD COLORBAR
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', size='7%', pad=0.1)

        # Very ugly workaround to scale the colorbar without clobbering the patch collection plot
        # (https://medium.com/data-science-canvas/way-to-show-colorbar-without-calling-imshow-or-scatter)
        sm = plt.cm.ScalarMappable(cmap=idlcmap, norm=plt.Normalize(vmin=0, vmax=np.max(ff_fb_emiss)))
        sm._A = []

        cbar = fig.colorbar(sm, cax=cbar_ax)
        label = '$\mathrm{ph\/s^{-1}\/m^{-3}\/sr^{-1}}$'
        cbar.set_label(label)

        # ADD DIAG LOS
        if diagLOS:
            for diag in diagLOS:
                self.__data2d.synth_diag[diag].plot_LOS(ax, color='darkgrey', lw=1.0)

        # PLOT SEPARATRIX AND WALL
        # Work around for this error message:
        # RuntimeError: Can not put single artist in more than one figure
        from copy import copy
        wallpoly = copy(self.__data2d.mesh_data.wall_poly)
        seppoly = copy(self.__data2d.mesh_data.sep_poly)
        wallpoly2 = patches.Polygon(wallpoly.get_xy(), closed=True, edgecolor=None, facecolor='w', zorder=2)
        seppoly2 = patches.Polygon(seppoly.get_xy(), closed=False, facecolor='None', edgecolor='k', zorder=10, lw=1)

        ax.add_patch(wallpoly2)
        ax.add_patch(seppoly2)

        ax.set_zorder(10)

        if savefig:
            plt.savefig(self.work_dir + self.case + '/' + title + '.png', dpi=plt.gcf().dpi)



    def plot_along_ring(self, params=['ne', 'te'], linestyles=['-', '-'], 
                        colors=['r', 'b'],
                        ring=11, savefig=False):

        ax = self.plot_dict['ring_params']['axs']

#        fig, ax = plt.subplots(ncols=1, figsize=(10, 4.5))
#        ax.set_ylabel('')
        ax.set_xlabel('E2D row')


        _params = []
        _emiss_excits = []
        _emiss_recoms = []
        for param in params:
            _param = []
            _emiss_excit = []
            _emiss_recom = []
            _rows = []            
            for cell in self.__data2d.cells:
                if cell.ring == ring:
#                if cell.R>2.7425 and cell.R<2.7475 and cell.Z>-1.652 and cell.Z<-1.65:
                    _rows.append(cell.row)
                    if param == 'ne':
                        _param.append(cell.ne)
                    elif param == 'n0':
                        _param.append(cell.n0)
                    elif param == 'Te':
                        _param.append(cell.te)
                    elif param == 'Sion':
                        _param.append(cell.Sion)
                    elif param == 'Srec':
                        _param.append(cell.Srec)
                    elif param == 'imp2_radpwr_1+':
                        _param.append(cell.imp2_radpwr_perm3[1])
                    elif param == 'imp2_radpwr_2+':
                        _param.append(cell.imp2_radpwr_perm3[2])
                    elif param == 'imp2_radpwr_3+':
                        _param.append(cell.imp2_radpwr_perm3[3])
                    elif param == 'imp2_radpwr_4+':
                        _param.append(cell.imp2_radpwr_perm3[4])
                    elif param == 'imp2_radpwr_5+':
                        _param.append(cell.imp2_radpwr_perm3[5])
                    elif param == 'imp2_radpwr_6+':
                        _param.append(cell.imp2_radpwr_perm3[6])
                    elif param == 'cNtot':
                        _param.append(np.sum(cell.imp2_den[0:])/cell.ne)
                    elif param == 'NIIemiss':
                        # Reconstruct from PECs - sanity check OK
#                        fPEC_excit = cell.imp_emiss['7']['2']['5002.18']['fPEC_excit']
#                        fPEC_recom = cell.imp_emiss['7']['2']['5002.18']['fPEC_recom']                        
#                        emiss = (1./(4.*np.pi))*1.e-06*np.sum(cell.imp2_den)*cell.ne*(fPEC_excit+fPEC_recom)  
                        emiss_excit = cell.imp_emiss['7']['2']['5002.18']['excit']
                        emiss_recom = cell.imp_emiss['7']['2']['5002.18']['recom']
                        emiss = (emiss_excit + emiss_recom)
                        _param.append(emiss)
                        _emiss_excit.append(emiss_excit)
                        _emiss_recom.append(emiss_recom)
                    elif param == 'NIIIemiss':
                        emiss_excit = cell.imp_emiss['7']['3']['4100.51']['excit']
                        emiss_recom = cell.imp_emiss['7']['3']['4100.51']['recom']
                        emiss = (emiss_excit + emiss_recom)                                          
                        _param.append(emiss)   
                        _emiss_excit.append(emiss_excit)
                        _emiss_recom.append(emiss_recom)                        
                    elif param == 'NIVemiss':
                        emiss_excit = cell.imp_emiss['7']['4']['3481.83']['excit']
                        emiss_recom = cell.imp_emiss['7']['4']['3481.83']['recom']
                        emiss = (emiss_excit + emiss_recom)                         
                        _param.append(emiss)
                        _emiss_excit.append(emiss_excit)
                        _emiss_recom.append(emiss_recom)                        
                    elif param == 'NIIemiss_ionbal':
                        # Reconstruct from PECs
                        fPEC_excit_ionbal = cell.imp_emiss['7']['2']['5002.18']['fPEC_excit_ionbal']
                        fPEC_recom_ionbal = cell.imp_emiss['7']['2']['5002.18']['fPEC_recom_ionbal']                        
                        emiss = (1./(4.*np.pi))*1.e-06*np.sum(cell.imp2_den)*cell.ne*(fPEC_excit_ionbal+fPEC_recom_ionbal)  
                        _param.append(emiss)
                    elif param == 'NIIIemiss_ionbal':
                        # Reconstruct from PECs
                        fPEC_excit_ionbal = cell.imp_emiss['7']['3']['4100.51']['fPEC_excit_ionbal']
                        fPEC_recom_ionbal = cell.imp_emiss['7']['3']['4100.51']['fPEC_recom_ionbal']                        
                        emiss = (1./(4.*np.pi))*1.e-06*np.sum(cell.imp2_den)*cell.ne*(fPEC_excit_ionbal+fPEC_recom_ionbal)  
                        _param.append(emiss)
                    elif param == 'NIVemiss_ionbal':
                        # Reconstruct from PECs
                        fPEC_excit_ionbal = cell.imp_emiss['7']['4']['3481.83']['fPEC_excit_ionbal']
                        fPEC_recom_ionbal = cell.imp_emiss['7']['4']['3481.83']['fPEC_recom_ionbal']                        
                        emiss =(1./(4.*np.pi))*1.e-06*np.sum(cell.imp2_den)*cell.ne*(fPEC_excit_ionbal+fPEC_recom_ionbal)  
                        _param.append(emiss) 
                    elif param == 'cN_NII':
                        fPEC_excit = cell.imp_emiss['7']['2']['5002.18']['fPEC_excit']
                        fPEC_recom = cell.imp_emiss['7']['2']['5002.18']['fPEC_recom']                                                
                        emiss = 1.e-06*(cell.imp_emiss['7']['2']['5002.18']['excit'] + 
                                 cell.imp_emiss['7']['2']['5002.18']['recom'])
                        c_N = 4.*np.pi*emiss/(1.e-06*cell.ne*1.e-06*cell.ne*(fPEC_excit+fPEC_recom))
                        _param.append(c_N)                        
                    elif param == 'cN_NIII':
                        fPEC_excit = cell.imp_emiss['7']['3']['4100.51']['fPEC_excit']
                        fPEC_recom = cell.imp_emiss['7']['3']['4100.51']['fPEC_recom']                                                
                        emiss = 1.e-06*(cell.imp_emiss['7']['3']['4100.51']['excit'] + 
                                 cell.imp_emiss['7']['3']['4100.51']['recom'])
                        c_N = 4.*np.pi*emiss/(1.e-06*cell.ne*1.e-06*cell.ne*(fPEC_excit+fPEC_recom))
                        _param.append(c_N)                        
                    elif param == 'cN_NIV':
                        fPEC_excit = cell.imp_emiss['7']['4']['3481.83']['fPEC_excit']
                        fPEC_recom = cell.imp_emiss['7']['4']['3481.83']['fPEC_recom']                                                
                        emiss = 1.e-06*(cell.imp_emiss['7']['4']['3481.83']['excit'] + 
                                 cell.imp_emiss['7']['4']['3481.83']['recom'])
                        c_N = 4.*np.pi*emiss/(1.e-06*cell.ne*1.e-06*cell.ne*(fPEC_excit+fPEC_recom))
                        _param.append(c_N) 
                    elif param == 'cN_NII_ionbal':
                        fPEC_excit_ionbal = cell.imp_emiss['7']['2']['5002.18']['fPEC_excit_ionbal']
                        fPEC_recom_ionbal = cell.imp_emiss['7']['2']['5002.18']['fPEC_recom_ionbal']                                                
                        emiss = 1.e-06*(cell.imp_emiss['7']['2']['5002.18']['excit'] + 
                                 cell.imp_emiss['7']['2']['5002.18']['recom'])
                        c_N = 4.*np.pi*emiss/(1.e-06*cell.ne*1.e-06*cell.ne*(fPEC_excit_ionbal+fPEC_recom_ionbal))
                        _param.append(c_N)
                    elif param == 'cN_NIII_ionbal':
                        fPEC_excit_ionbal = cell.imp_emiss['7']['3']['4100.51']['fPEC_excit_ionbal']
                        fPEC_recom_ionbal = cell.imp_emiss['7']['3']['4100.51']['fPEC_recom_ionbal']                                                
                        emiss = 1.e-06*(cell.imp_emiss['7']['3']['4100.51']['excit'] + 
                                 cell.imp_emiss['7']['3']['4100.51']['recom'])
                        c_N = 4.*np.pi*emiss/(1.e-06*cell.ne*1.e-06*cell.ne*(fPEC_excit_ionbal+fPEC_recom_ionbal))
                        _param.append(c_N)                        
                    elif param == 'cN_NIV_ionbal':
                        fPEC_excit_ionbal = cell.imp_emiss['7']['4']['3481.83']['fPEC_excit_ionbal']
                        fPEC_recom_ionbal = cell.imp_emiss['7']['4']['3481.83']['fPEC_recom_ionbal']                                                
                        emiss = 1.e-06*(cell.imp_emiss['7']['4']['3481.83']['excit'] + 
                                 cell.imp_emiss['7']['4']['3481.83']['recom'])
                        c_N = 4.*np.pi*emiss/(1.e-06*cell.ne*1.e-06*cell.ne*(fPEC_excit_ionbal+fPEC_recom_ionbal))
                        _param.append(c_N)                          
                    elif param == 'f_N0_ionbal':
                        fPEC_recom_ionbal = cell.imp_emiss['7']['2']['5002.18']['fPEC_recom_ionbal'] 
                        PEC_recom = cell.imp_emiss['7']['2']['5002.18']['PEC_recom']
                        f_N0_ionbal =  fPEC_recom_ionbal / PEC_recom                                          
                        _param.append(f_N0_ionbal)                          
                    elif param == 'f_N1+_ionbal':
                        fPEC_excit_ionbal = cell.imp_emiss['7']['2']['5002.18']['fPEC_excit_ionbal']
                        PEC_excit = cell.imp_emiss['7']['2']['5002.18']['PEC_excit']
                        f_N1_ionbal =  fPEC_excit_ionbal / PEC_excit                                          
                        _param.append(f_N1_ionbal)                          
                    elif param == 'f_N2+_ionbal':
                        fPEC_excit_ionbal = cell.imp_emiss['7']['3']['4100.51']['fPEC_excit_ionbal']
                        PEC_excit = cell.imp_emiss['7']['3']['4100.51']['PEC_excit']
                        f_N2_ionbal =  fPEC_excit_ionbal / PEC_excit                                          
                        _param.append(f_N2_ionbal)                          
                    elif param == 'f_N3+_ionbal':
                        fPEC_excit_ionbal = cell.imp_emiss['7']['4']['3481.83']['fPEC_excit_ionbal']
                        PEC_excit = cell.imp_emiss['7']['4']['3481.83']['PEC_excit']
                        f_N3_ionbal =  fPEC_excit_ionbal / PEC_excit                                          
                        _param.append(f_N3_ionbal)                          
                    elif param == 'f_N0':
                        _param.append(cell.imp2_den[0]/np.sum(cell.imp2_den))                         
                    elif param == 'f_N1+':
                        _param.append(cell.imp2_den[1]/np.sum(cell.imp2_den))                         
                    elif param == 'f_N2+':
                        _param.append(cell.imp2_den[2]/np.sum(cell.imp2_den))                         
                    elif param == 'f_N3+':
                        _param.append(cell.imp2_den[3]/np.sum(cell.imp2_den))                         
                    elif param == 'f_N4+':
                        _param.append(cell.imp2_den[4]/np.sum(cell.imp2_den))                         
                    elif param == 'f_N5+':
                        _param.append(cell.imp2_den[5]/np.sum(cell.imp2_den))                         
                    elif param == 'f_N6+':
                        _param.append(cell.imp2_den[6]/np.sum(cell.imp2_den))                         


            # sort by row
            _rows_idx = np.asarray(np.argsort(_rows))
            _rows = np.asarray(_rows)[_rows_idx]
            _param = np.asarray(_param)[_rows_idx]
            _params.append(_param)
            _emiss_excit = np.asarray(_emiss_excit)[_rows_idx]
            _emiss_excits.append(_emiss_excit)
            _emiss_recom = np.asarray(_emiss_recom)[_rows_idx]
            _emiss_recoms.append(_emiss_recom)

        _params = np.asarray(_params)
        
        for i, param in enumerate(params):
#            ax.plot(_params[i]/np.max(_params[i]), '-', 
#            ax.semilogy(_params[i], '-', 
            ax.plot(_params[i], '-', 
#            ax.plot(_rows, _params[i], '-', 
                    ls=linestyles[i], c=colors[i],label=param)
            ax.plot(_emiss_excits[i], '--', 
#            ax.plot(_rows, _params[i], '-', 
                    ls='--', c=colors[i],label=param)
            ax.plot(_emiss_recoms[i], ':', 
#            ax.plot(_rows, _params[i], '-', 
                    ls=':', c=colors[i],label=param)

        title = self.case + ' ring '+ str(ring)
        ax.set_title(title)
        
        ax.legend(loc='upper right')
#        ax.set_xlim(0,30)
#        ax.set_ylim(0,100)

        if savefig:
            plt.savefig(self.work_dir + self.case + '/' + title + '.png', dpi=plt.gcf().dpi)
            
            
            

    def get_param_at_max_ne_along_los(self, diag, paramstr, nAvgNeighbs=2):

        ne = self.get_line_int_sorted_data_by_chord_id(diag, ['los_1d', 'ne'])
        par = self.get_line_int_sorted_data_by_chord_id(diag, ['los_1d', paramstr])

        ne_max = []
        par_at_ne_max = []

        for i in range(len(ne)):
            if ne[i]:
                ne_los = np.asarray(ne[i])
                par_los = np.asarray(par[i])

                ne_max_idx, val = find_nearest(ne_los, np.max(ne_los))

                # ne_max.append(ne_los[ne_max_idx])
                # par_at_ne_max.append(par_los[ne_max_idx])

                # Find parameter value at position corresponding to max ne along LOS (include nearest neighbours and average)
                if (ne_max_idx + 1) == len(ne_los):
                    ne_max.append(np.average(np.array((ne_los[ne_max_idx - 1], ne_los[ne_max_idx]))))
                    par_at_ne_max.append(np.average(np.array((par_los[ne_max_idx - 1], par_los[ne_max_idx]))))
                elif (ne_max_idx + 2) == len(ne_los):
                    ne_max.append(np.average(np.array((ne_los[ne_max_idx - 1], ne_los[ne_max_idx],
                                                    ne_los[ne_max_idx + 1]))))
                    par_at_ne_max.append(np.average(np.array((par_los[ne_max_idx - 1], par_los[ne_max_idx],
                                                    par_los[ne_max_idx + 1]))))
                else:
                    ne_max.append(np.average(np.array((ne_los[ne_max_idx + 2], ne_los[ne_max_idx],
                                                    ne_los[ne_max_idx + 1]))))
                    par_at_ne_max.append(np.average(np.array((par_los[ne_max_idx + 2], par_los[ne_max_idx],
                                                    par_los[ne_max_idx + 1]))))
            else:
                ne_max.append(0)
                par_at_ne_max.append(0)

        return np.asarray(ne_max), np.asarray(par_at_ne_max)


if __name__=='__main__':

    print('')