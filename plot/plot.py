import os
import pwd
import math
import numpy as np
import json, pprint, pickle
import operator
from functools import reduce
import matplotlib
#matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib import patches, ticker
from collections import OrderedDict
from PESDT import process
from PESDT.atomic import get_ADAS_dict
# from pyproc.process import ProcessEdgeSim

from scipy.constants import Planck, speed_of_light

import sys
sys.path[:0]=['/jet/share/lib/python']
from ppf import * # JET ppf python library

font = {'family': 'normal',
        'weight': 'normal',
        'size': 18}
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

class Plot():
    """
        Class for retrieving, reducing and plotting PESDT saved data
    """
    def __init__(self, work_dir, case, plot_dict=None, icase=1):
        self.work_dir = work_dir
        self.case = case
        self.plot_dict = plot_dict
        self.icase = icase
        # Cherab flags
        self.cherab_bridge = False
        self.cherab_reflection = False
        # Read pickled PESDT object
        try:
            with open(self.work_dir + self.case + '/PESDT.2ddata.pkl', 'rb') as f:
                self.__data2d = pickle.load(f)
        except IOError as e:
            raise
        
        if self.plot_dict.get('synth_data', False):
            try:
                with open(self.work_dir + self.case +  '/PESDT.proc_synth_diag.json', 'r') as f:
                    self.__res_dict = json.load(f)
            except IOError as e:
                raise
        # Read processed synth diag saved data
        else:
            try:
                with open(self.work_dir + self.case +  '/PESDT.synth_diag.json', 'r') as f:
                    self.__res_dict = json.load(f)
            except IOError as e:
                raise

        if plot_dict:
            # Read cherab_bridge synth diag save data
            if 'cherab_bridge_results' in plot_dict:
                self.cherab_bridge = plot_dict['cherab_bridge_results']
                if self.cherab_bridge:
                    if 'cherab_reflections' in plot_dict:
                        self.cherab_reflections = plot_dict['cherab_reflections']
                        if self.cherab_reflections:
                            cherab_res_file = self.work_dir + self.case + '/cherab_refl.synth_diag.json'
                        else:
                            cherab_res_file = self.work_dir + self.case + '/cherab.synth_diag.json'
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
                    self.plot_param_profiles(lineweight=2.0, alpha=0.2500, legend=True)
                if key == 'prof_Hemiss_defs':
                    self.plot_Hemiss_profiles(lineweight=3.0, alpha=0.2500, legend=True,
                                              Watts=False)
                if key == 'prof_Prad_defs':
                    write_ppf = False
                    if 'write_ppf' in val:
                        write_ppf = val['write_ppf']
                    self.plot_Prad_profiles(lineweight=3.0, alpha=0.250, legend=True,
                                            write_ppf=write_ppf)
                if key == 'param_along_LOS':
                    self.plot_params_along_LOS(lineweight=2.0, alpha=0.250, legend=False)
                if key == '2d_defs':
                    diagLOS = val['diagLOS']
                    savefig = val['save']
                    Rrng = val['Rrng']
                    Zrng = val['Zrng']
                    # self.plot_2d_ff_fb(diagLOS, Rrng=Rrng, Zrng=Zrng, savefig=savefig)
                    for at_num in val['lines']:
                        for stage in val['lines'][at_num]:
                            for line in val['lines'][at_num][stage]:
                                self.plot_2d_spec_line(at_num, stage, line, diagLOS, max_abs = 1.5e22,
                                                       Rrng=Rrng, Zrng=Zrng, savefig=savefig)
                if key == '2d_prad':
                    diagLOS = val['diagLOS']
                    savefig = val['save']
                    Rrng = val['Rrng']
                    Zrng = val['Zrng']
                    self.plot_2d_prad(diagLOS, Rrng=Rrng, Zrng=Zrng, savefig=savefig)

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
                            fontsize=20, alpha=0.250, ne_scal=1.0, legend=True):
        '''
        Reads in inputs from the plot_dict dictionary 
        'prof_param_defs':
            'axs': the pyplot axis for plotting
                Note: 'axs' actually controls which plasma parameters are plotted
                if axs is not an np.array (i.e. you have a single axis), plots only the 
                recovered ne profile
                if axs > 1 --> also plot Te
					NOTE: Te plotting does not seem to work...
                if axs > 2 --> also plot ionization
                if axs > 3 --> also plot N0 (atomic hydrogen)
            'diag': diagnostic name (e.g. KS3O)
            'color', 'color2': colors used in plotting
            'zorder': ?
            'coord': The coordinate used for sorting the diag LOS
            'Sion_H_transition' and 'Srec_H_transition': the H line emissions used for 
            assesing the conditions (ionizing or recombining)
            'include_pars_at_max_ne_along_LOS': ?
            'include_target_vals': Plot Te, ne at the target (radial)
            'include_sum_Sion_Srec': ?
        '''
        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PLASMA PARAMS
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

        true_val_col = 'k'

        # Ne
        ne = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'stark', 'fit', 'ne'])
        if not isinstance(axs, np.ndarray):
            axs.plot(x, ne_scal*ne, c=color, ls=linestyle, lw=lineweight, zorder=zorder)
        else:
            axs[0].plot(x, ne_scal*ne, c=color, ls=linestyle, lw=lineweight, zorder=zorder)


        if isinstance(axs, np.ndarray) and len(axs) > 1:
        # Te
            # Te_hi = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_360_400'])
            Te_hi = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_400_500'])
            Te_lo = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_300_360'])
            print(Te_hi)
            print(Te_lo)

            axs[1].plot(x, Te_hi, c=color, ls=linestyle, lw=lineweight, zorder=zorder)
            axs[1].plot(x, Te_lo, c=color, ls=linestyle, lw=lineweight, zorder=zorder)
            
        if isinstance(axs, np.ndarray) and len(axs) > 2:
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

                print('Sion_adf11, tran_str (s^-1): ', Sion_adf11)
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

                print('Srec_adf11, tran_str (s^-1): ', Srec_adf11)
                # Total recombination/ionisation (derived from emisSrec with adf11)
                axs[2].semilogy(x, Srec_scal * Srec_adf11, c=color, ls=ls[itran], lw=lineweight, zorder=zorder,
                                label='Srec_' + tran_str)


            # axs[2].plot(0, 0, c='k', linewidth=2, label='Ionization')
            # axs[2].plot(0, 0, '--', c='k', linewidth=2, label='Recombination')
            if legend:
                leg = axs[2].legend(loc='upper right', prop={'size':14}, frameon=False, labelspacing=0.05)


        if isinstance(axs, np.ndarray) and len(axs) > 3:
            # N0
            n0delL = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Ly_alpha_fit', 'n0delL'])

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
                axs[0].plot(self.data2d.denel_OT['xdata'][:self.data2d.denel_OT['npts']]+self.data2d.osp[0],
                            ne_scal *self.data2d.denel_OT['ydata'][:self.data2d.denel_OT['npts']], 'o', mfc='None',
                            mec=true_val_col, mew=2.0, ms=8, zorder=4)
                # axs[0].plot(self.data2d.denel_IT['xdata'][:self.data2d.denel_IT['npts']]+self.data2d.isp[0],
                #             self.data2d.denel_IT['ydata'][:self.data2d.denel_IT['npts']], 'o', mfc='None',
                #             mec=color, mew=2.0, ms=8)
            if len(axs) > 1:
                axs[1].plot(self.data2d.teve_OT['xdata'][:self.data2d.teve_OT['npts']]+self.data2d.osp[0],
                            self.data2d.teve_OT['ydata'][:self.data2d.teve_OT['npts']], 'o', mfc='None',
                            mec=true_val_col, mew=2.0, ms=8, zorder=4)
                # axs[1].plot(self.data2d.teve_IT['xdata'][:self.data2d.teve_IT['npts']]+self.data2d.isp[0],
                #             self.data2d.teve_IT['ydata'][:self.data2d.teve_IT['npts']], 'o', mfc='None',
                #             mec=color, mew=2.0, ms=8)
            # Ion flux to outer target
            # if len(axs) > 2:
                # axs[2].plot(self.data2d.pflxd_OT['xdata'][:self.data2d.pflxd_OT['npts']]+self.data2d.osp[0],
                #             -1.0*self.data2d.pflxd_OT['ydata'][:self.data2d.pflxd_OT['npts']], 'o', mfc='None',
                #             mec=color, mew=2.0, ms=8)
                # axs[2].plot(self.data2d.pflxd_IT['xdata'][:self.data2d.pflxd_IT['npts']] + self.data2d.osp[0],
                #             -1.0 * self.data2d.pflxd_IT['ydata'][:self.data2d.pflxd_IT['npts']], 'o', mfc='None',
                #             mec=color, mew=2.0, ms=8)
            
            # neutral density
            if len(axs) > 3:
                axs[3].plot(self.data2d.da_OT['xdata'][:self.data2d.da_OT['npts']] + self.data2d.osp[0],
                            self.data2d.da_OT['ydata'][:self.data2d.da_OT['npts']], 'o', mfc='None',
                            mec=true_val_col, mew=2.0, ms=8)

        if self.plot_dict['prof_param_defs']['include_sum_Sion_Srec']:
            Sion = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Sion', 'val'])
            Srec = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Srec', 'val'])
            if len(axs) > 2:
                axs[2].semilogy(x, Sion, '-', c=true_val_col, lw=2.0, zorder=5)
                # axs[2].semilogy(x, Sion, '-', c='k', lw=2.0, zorder=1)
                axs[2].semilogy(x, -1.0*Srec, '--', c=true_val_col, lw=2.0, zorder=5)


            # Output Sion/Srec direct and spectroscopically inferred summations
            R = p2[:, 0]
            inner_idx, = np.where(R < self.__data2d.geom['rpx'])
            outer_idx, = np.where(R >= self.__data2d.geom['rpx'])
            print('')
            print(diag, ' particle balance')
            print('Direct sum of Sion, Srec along LOS [s^-1]')
            print('Total (R < R_xpt) :', np.sum(Sion[inner_idx]), np.sum(Srec[inner_idx]))
            print('Total (R >= R_xpt) :', np.sum(Sion[outer_idx]), np.sum(Srec[outer_idx]))
            print('adf11 Sion, Srec estimates[s^-1]')
            print('Total (R < R_xpt) :', np.sum(Sion_adf11[inner_idx]), np.sum(Srec_adf11[inner_idx]))
            print('Total (R >= R_xpt) :', np.sum(Sion_adf11[outer_idx]), np.sum(Srec_adf11[outer_idx]))
            print('')

        # xpt, isp, osp locations
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].axvline(self.__data2d.geom['rpx'], ls=':', c='darkgrey', linewidth=2.)
                axs[i].axvline(self.__data2d.osp[0], ls=':', c='darkgrey', linewidth=2.)
                axs[i].axvline(self.__data2d.isp[0], ls=':', c='darkgrey', linewidth=2.)

                if ylim:
                    axs[i].set_ylim(ylim[i][0], ylim[i][1])
                if xlim:
                    axs[i].set_xlim(xlim[0], xlim[1])
                else:
                    axs[i].set_xlim(x[0], x[-1])

            # axs[len(axs)-1].set_xlabel('Major radius on tile 5 (m)', fontsize=fontsize)
            axs[0].set_xlabel('R (m)', fontsize=fontsize)
            axs[0].xaxis.set_tick_params(labelsize=fontsize)
            axs[len(axs)-1].set_xlabel('R (m)', fontsize=fontsize)
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
            axs.axvline(self.__data2d.geom['rpx'], ls=':', c='darkgrey', linewidth=2., zorder=1)
            axs.axvline(self.__data2d.osp[0], ls=':', c='darkgrey', linewidth=2., zorder=1)
            axs.axvline(self.__data2d.isp[0], ls=':', c='darkgrey', linewidth=2., zorder=1)
            axs.set_xlabel('Major radius on tile 5 (m)', fontsize=fontsize)
            # axs.set_xlabel('R (m)', fontsize=fontsize)
            axs.set_ylabel(r'$\mathrm{n_{e}\/(10^{20}\/m^{-3})}$', fontsize=fontsize, labelpad=10)
            axs.xaxis.set_tick_params(labelsize=fontsize)
            axs.yaxis.set_tick_params(labelsize=fontsize)


        # axes_dict['main'][3].set_ylabel(r'$\mathrm{n_{H}\/(m^{-3})}$')

    def plot_Hemiss_profiles(self, lineweight=2.0, alpha=0.250, legend=True,
                             linestyle='-', scal=1.e-21,
                             fontsize=18,
                             Watts=False):
        '''
        Plots the synthetic hydrogenic line emissions
        
        
        Reads in inputs from the plot_dict dictionary 
        'prof_Hemiss_defs':
            'axs': the pyplot axis for plotting
					Note: currently len(axs) must be > 2! (i.e., you need more than one line)
            'diag': diagnostic name (e.g. KS3O)
            'color', 'color2': colors used in plotting
            'zorder': ?
            'exrec': ?
            'coord': The coordinate used for sorting the diag LOS
			'lines_lytrap': plot the estimated emission when lyman trapping is taken into account
			'write_csv': writes a text file of the synthetic measurement along diag LOS
        
        '''
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
        cherab_res = self.plot_dict.get('cherab_bridge_results', False)
        write_csv = False
        if 'write_csv' in self.plot_dict['prof_Hemiss_defs']:
            write_csv = self.plot_dict['prof_Hemiss_defs']['write_csv']
        amjuel = self.plot_dict['prof_Hemiss_defs'].get('AMJUEL', False)
        ylim, xlim = None, None
        if 'ylim' in self.plot_dict['prof_Hemiss_defs']:
            ylim = self.plot_dict['prof_Hemiss_defs']['ylim']
        if 'xlim' in self.plot_dict['prof_Hemiss_defs']:
            xlim = self.plot_dict['prof_Hemiss_defs']['xlim']

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
            if amjuel:
                #if cherab_res:
                excit = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'excit'])
                recom = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'recom'])
                h2 = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'h2'])
                h2_pos = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'h2+'])
                h_neg = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'h-'])
                #else:
                #    excit = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'tot'])
                #    recom = np.zeros(np.shape(excit))        
            else:
                excit = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'excit'])
                recom = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'recom'])
            label = '{:5.1f}'.format(float(line)/10.) + ' nm'

            if Watts:
                centre_wav_nm = float(line)/10.
                excit = PhotonToJ.to(excit, centre_wav_nm)
                recom = PhotonToJ.to(recom, centre_wav_nm)
                ylabel = '$\mathrm{W\/m^{-2}\/sr^{-1}}$'
            else:
                ylabel = '$\mathrm{10^{21}\/ph\/s^{-1}\/m^{-2}\/sr^{-1}}$'
                # ylabel = '$\mathrm{ph\/s^{-1}\/m^{-2}\/sr^{-1}}$'

            
            if amjuel:
                axs[i].plot(x, scal*(excit+recom+h2+h2_pos+h_neg), ls=linestyle, lw=lineweight, c=color, zorder=zorder, label=label)
                axs[i].plot(x, scal*(excit), '--', lw=1, c=color, zorder=zorder, label=label + 'excit')
                axs[i].plot(x, scal*(recom), ':', lw=1, c=color, zorder=zorder, label=label + 'recom')
                axs[i].plot(x, scal*(h2), '--', lw=1, c='r', zorder=zorder, label=label + ' D2')
                axs[i].plot(x, scal*(h2_pos), ':', lw=1, c='r', zorder=zorder, label=label + ' D2+')
                axs[i].plot(x, scal*(h_neg),'.', lw=1, c='g', zorder=zorder, label=label + ' D-')
                
            else:
                axs[i].plot(x, scal*(excit+recom), ls=linestyle, lw=lineweight, c=color, zorder=zorder, label=label)

                if excrec:
                    axs[i].plot(x, scal*excit, '--', lw=1, c=color, zorder=zorder, label=label+' excit')
                    axs[i].plot(x, scal*recom, ':', lw=1, c=color, zorder=zorder, label=label+' recom')

            if write_csv:
                filedir = self.work_dir + self.case + '/'
                filename = filedir + self.case + '.' + diag + '.' + 'Hemiss' + '.' + '{:5.1f}'.format(float(line)/10.) + 'nm.txt'
                if amjuel and cherab_res:
                    header = 'line: ' + '{:5.1f}'.format(float(line)/10.) + ' nm, ' + 'cols: ' + coord + ', excit (ph/s/m^2/sr), recom (ph/s/m^2/sr), D2 (ph/s/m^2/sr), D2+ (ph/s/m^2/sr), D- (ph/s/m^2/sr)'
                    np.savetxt(filename, np.transpose((x,excit,recom, h2, h2_pos, h_neg)), header=header, delimiter=',')
                else:
                    header = 'line: ' + '{:5.1f}'.format(float(line)/10.) + ' nm, ' + 'cols: ' + coord + ', excit (ph/s/m^2/sr), recom (ph/s/m^2/sr)'
                    np.savetxt(filename, np.transpose((x,excit,recom)), header=header, delimiter=',')

            if legend:
                leg = axs[i].legend(loc='upper left')
                # leg.get_frame().set_alpha(0.2)
            if i == len(lines.keys())-1:
                # axs[i].set_xlabel('Major radius on tile 5 (m)', fontsize=fontsize)
                axs[i].set_xlabel(coord, fontsize=fontsize)
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
                label = '{:5.1f}'.format(float('1215.67') / 10.) + ' nm w/ ' + 'ad hoc opacity'
                axs[0].plot(x, excit + recom, '-', lw=2, c=color, zorder=zorder, label=label)
                if excrec:
                    axs[0].plot(x, excit, '--', lw=1, c=color, zorder=zorder, label=label + ' excit')
                    axs[0].plot(x, recom, ':', lw=1, c=color, zorder=zorder, label=label + ' recom')
                leg = axs[0].legend(loc='upper left')
                leg.get_frame().set_alpha(0.2)
            if '6564.57'in lines_lytrap:
                excit = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '6564.57', 'excit'])
                recom = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '6564.57', 'recom'])
                label = '{:5.1f}'.format(float('6564.57') / 10.) + ' nm w/ ' + 'ad hoc opacity'
                axs[1].plot(x, excit + recom, '-', lw=2, c=color, zorder=zorder, label=label)

                # EXPERIMENTAL ######################################
                if '1025.72' in lines_lytrap:
                    # axs_twinx = axs[1].twinx()
                    # Plot Ly-beta/D-alpha ratio
                    f_1_2 = 0.41641
                    f_1_3 = 0.079142
                    Arate_1_3 = 0.5575e08  # s^-1
                    Arate_2_3 = 0.4410e08  # s^-1

                    excit_Da = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1215.67', 'excit'])
                    recom_Da = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1215.67', 'recom'])
                    # excit_Da = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1215.2', 'excit'])
                    # recom_Da = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1215.2', 'recom'])
                    Lya = excit_Da + recom_Da

                    excit_Da = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '6564.57', 'excit'])
                    recom_Da = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '6564.57', 'recom'])
                    # excit_Da = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '6561.9', 'excit'])
                    # recom_Da = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '6561.9', 'recom'])
                    Da = excit_Da + recom_Da

                    excit_Lyb = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1025.72', 'excit'])
                    recom_Lyb = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1025.72', 'recom'])
                    # excit_Lyb = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1025.3', 'excit'])
                    # recom_Lyb = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1025.3', 'recom'])

                    Lyb = excit_Lyb + recom_Lyb

                    # label = '{:5.1f}'.format(float('6564.57') / 10.) + ' nm; ' + 'ad hoc opacity'
                    # axs_twinx.plot(x, (Arate_2_3/Arate_1_3)*(excit_Lyb + recom_Lyb)/(excit_Da + recom_Da), '--', lw=2, c=color, zorder=zorder, label=label)
                    # axs_twinx.set_ylim(0,2)
                    from pyADASutils import opacity_utils
                    from pyADASread import adas_adf11_read, adas_adf15_read
                    g_tau_Ly_beta = (Arate_2_3/Arate_1_3)*(Lyb/Da)
                    print(g_tau_Ly_beta)
                    # KT3 g_tau_Ly_beta from oct1317/seq#1
                    g_tau_Ly_beta = [0.46726903,  0.49397789,  0.47733701,  0.6263499,   0.66123852,  0.70666998,
                                     0.77926284,  0.84191311,  0.91273625,  0.94184994,  0.81984646,  0.5897365,
                                     0.36794049,  0.42311035,  0.46638785,  0.45649988,  0.44488203,  0.4756023,
                                     0.54201328,  0.90349965,  0.99123816, 0.98872541]
                    tau_Ly_beta = np.zeros((len(g_tau_Ly_beta)))
                    g_tau_Ly_alpha = np.zeros((len(g_tau_Ly_beta)))
                    for i in range(len(g_tau_Ly_beta)):
                        g_dum, tau_Ly_beta[i] = opacity_utils.get_opacity_from_escape_factor(g_tau_Ly_beta[i])
                    tau_Ly_alpha = tau_Ly_beta * f_1_2 / f_1_3
                    for i in range(len(tau_Ly_alpha)):
                        g_tau_Ly_alpha[i] = opacity_utils.calc_escape_factor(tau_Ly_alpha[i])
                    # axs_twinx.plot(x, g_tau_Ly_beta, '--', lw=2, c=color, zorder=zorder, label=label)
                    # axs_twinx.plot(x, g_tau_Ly_alpha, '--', lw=2, c=color, zorder=zorder, label=label)

                    # Calculate Siz
                    Siz = []
                    Te_arr = np.logspace(np.log10(0.2), np.log10(20), 50)
                    ne_arr = np.logspace(np.log10(1.0e14), np.log10(1.0e15), 10)
                    ne = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'stark', 'fit', 'ne'])
                    Te_hi = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_360_400'])
                    for icoord, coord in enumerate(x):
                        print('Escape factor: R =', coord, ', ', g_tau_Ly_beta[icoord])
                        # opacity estimate only valid for R = 2.7-2.8 due to uncertainties in reflections
                        if coord >= 2.72 and coord <= 2.955:
                            adf11_opa = adas_adf11_read.get_adas_H_adf11_suppressed(Te_arr, ne_arr,
                                                                                    g_tau_Ly_beta[icoord])
                            PEC_dict_opa = adas_adf15_read.get_adas_H_PECs_n5(Te_arr, ne_arr,
                                                                              g_tau_Ly_beta[icoord])
                            # adf11_opa = adas_adf11_read.get_adas_H_adf11_suppressed(Te_arr, ne_arr,
                            #                                                         1.0)
                            # PEC_dict_opa = adas_adf15_read.get_adas_H_PECs_n5(Te_arr, ne_arr,
                            #                                                   1.0)
                        elif coord < 2.72:
                            adf11_opa = adas_adf11_read.get_adas_H_adf11_suppressed(Te_arr, ne_arr, 1.0)
                            PEC_dict_opa = adas_adf15_read.get_adas_H_PECs_n5(Te_arr, ne_arr, 1.0)

                        else:
                            adf11_opa = adas_adf11_read.get_adas_H_adf11_suppressed(Te_arr, ne_arr, 1.0)
                            PEC_dict_opa = adas_adf15_read.get_adas_H_PECs_n5(Te_arr, ne_arr, 1.0)

                        dR = x[1]-x[0]
                        area = 2. * np.pi * coord * dR
                        idxne, ne_val = find_nearest(adf11_opa.ne_arr, 1.0e-06*ne[icoord])
                        idxTe, Te_val = find_nearest(adf11_opa.Te_arr, Te_hi[icoord])
                        Siz.append(
                            4. * np.pi * area * Lya[icoord] * adf11_opa.scd[idxTe, idxne] /
                            PEC_dict_opa['1215.67excit'].pec[idxTe, idxne])
                    print('Siz: ', np.sum(np.asarray(Siz)))
                    print(Siz)
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
            axs[i].axvline(self.__data2d.geom['rpx'], ls=':', c='darkgrey', linewidth=2.)
            axs[i].axvline(self.__data2d.osp[0], ls=':', c='darkgrey', linewidth=2.)
            axs[i].axvline(self.__data2d.isp[0], ls=':', c='darkgrey', linewidth=2.)
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
                # axs[i].set_title(title, rotation=270, x=1.05, y=0.9)
            if i==len(axs)-1:
                axs[i].set_xlabel('Major radius on tile 5 (m)', fontsize=fontsize)
                axs[i].xaxis.set_tick_params(labelsize=fontsize)
                axs[i].yaxis.set_tick_params(labelsize=fontsize)

    def write_bolo_ppf(self, pulse, ppf_data_dict, tstart=52.0, wUid=None):
        # based on Anthony Field's PPF writer for BORP (pyDivertor/bolometry/bolt.py)

        print('Writing synthetic data to PPF: ' + str(pulse) + ' BOLO')

        if wUid == None:
            uid = pwd.getpwuid(os.getuid())[0]  # Get current user's UID
        else:
            uid = wUid

        ppfuid(uid, rw='W')                 # Change to current UID

        ier = ppfgo()                       # Initialise PPF system

        if ier != 0:
            ppferr('PPFGO', ier)

        time, date, ier = pdstd(pulse)        # Get the time and date of the pulse

        if ier != 0:
            ppferr('PDSTD', ier)

        comment = 'Writing synthetic EDGE2D data to ppf'

        ier = ppfopn(pulse, date, time, comment, status=0)

        if ier != 0:
            msg, ier = ppferr('PPFOPN', ier)
            exit('PPFOPN' + ": error code" + msg)

        tvec = np.linspace(tstart, tstart+1, 10)

        dda = 'BOLO'

        for cam, camdata in ppf_data_dict.items():
            #Prepare data
            if cam =='KB5V':
                # pad kb5v channels 25-32
                xpad = np.array((288.760, 280.005, 271.202, 262.441, 253.218, 243.121, 232.948, 222.261))
                x = np.concatenate((camdata['x'], xpad))
                data_pad = np.ones((8))
                data = np.concatenate((camdata['data'], data_pad))
            elif cam == 'KB5H':
                x = camdata['x']
                data = camdata['data']

            data = np.array([data,]*len(tvec))

            dtype = cam

            irdat = ppfwri_irdat(len(x), len(tvec), refx=-1, reft=-1, user=0, system=0)
            ihdat = ppfwri_ihdat('W m-2', 'deg', 's', 'f', 'f', 'f', 'Synthetic ' + dtype)
            iwdat, ier = ppfwri(pulse, dda, dtype, irdat, ihdat, data, x, tvec)

            if ier !=0:
                msg, ier = ppferr('PPFWRI', ier)
                exit('PPFWRI' + ": error code" + msg)
                ier = ppfabo()

                if ier != 0:
                    msg, ier = ppferr('PPFABO', ier)
                    exit('PPFABO' + ": error code" + msg)

                exit()


        # Close the PPF here and clean up!
        seq, ier = ppfclo(pulse, 'pyproc', vers=1)

        if ier != 0:
            msg, ier = ppferr('ppfclo', ier)
            exit('PPFCLO: error code' + msg)

        print('Created PPF for seq.: ', seq)

    def write_B3X4_ppf(self, pulse, dda, tstart=52.0, wUid=None):
        # based on Anthony Field's PPF writer for BORP (pyDivertor/bolometry/bolt.py)

        print('Writing synthetic data to PPF: ' + str(pulse) + ' ' + dda)

        if wUid == None:
            uid = pwd.getpwuid(os.getuid())[0]  # Get current user's UID
        else:
            uid = wUid

        ppfuid(uid, rw='W')  # Change to current UID

        ier = ppfgo()  # Initialise PPF system

        if ier != 0:
            ppferr('PPFGO', ier)

        time, date, ier = pdstd(pulse)  # Get the time and date of the pulse

        if ier != 0:
            ppferr('PDSTD', ier)

        comment = 'Writing synthetic EDGE2D data to ppf'

        ier = ppfopn(pulse, date, time, comment, status=0)

        if ier != 0:
            msg, ier = ppferr('PPFOPN', ier)
            exit('PPFOPN' + ": error code" + msg)

        tvec = np.linspace(tstart, tstart + 1, 10)

        for chan, chan_dict in self.__res_dict[dda].items():
            dtype = chan
            x = 0.0
            Prad_H = chan_dict['los_int']['Prad_perm2']['H']
            Prad_imp1 = np.sum(chan_dict['los_int']['Prad_perm2']['imp1'])
            Prad_imp2 = np.sum(chan_dict['los_int']['Prad_perm2']['imp2'])

            ppf_data = np.repeat(Prad_H+Prad_imp1+Prad_imp2, len(tvec))

            irdat = ppfwri_irdat(1, len(tvec), refx=-1, reft=-1, user=0, system=0)
            ihdat = ppfwri_ihdat('W m-2', 'deg', 's', 'f', 'f', 'f', 'Synthetic ' + dtype)
            iwdat, ier = ppfwri(pulse, dda, dtype, irdat, ihdat, ppf_data, x, tvec)

            if ier != 0:
                msg, ier = ppferr('PPFWRI', ier)
                exit('PPFWRI' + ": error code" + msg)
                ier = ppfabo()

                if ier != 0:
                    msg, ier = ppferr('PPFABO', ier)
                    exit('PPFABO' + ": error code" + msg)

                exit()

        # Close the PPF here and clean up!
        seq, ier = ppfclo(pulse, 'pyproc', vers=1)

        if ier != 0:
            msg, ier = ppferr('ppfclo', ier)
            exit('PPFCLO: error code' + msg)

        print('Created PPF for seq.: ', seq)

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

            ylabel = '$\mathrm{W\/m^{-2}}$'


            Prad_sum = Prad_ff_perm2 + Prad_H_perm2 

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
            header =  coord + ', Prad_free_free, Prad_H (W/m^2)'
            np.savetxt(filename, np.transpose((x,Prad_ff_perm2, Prad_H_perm2)), header=header, delimiter=',')

        if legend:
            leg = axs.legend(loc='upper left')
            # leg.get_frame().set_alpha(0.2)

        # axs.set_xlabel('Major radius on tile 5 (m)', fontsize=fontsize)
        axs.set_xlabel(coord, fontsize=fontsize)

        axs.set_ylabel(ylabel, fontsize=fontsize, labelpad=10)

        # xpt, isp, osp locations
        axs.axvline(self.__data2d.geom['rpx'], ls=':', c='darkgrey', linewidth=2.)
        axs.axvline(self.__data2d.osp[0], ls=':', c='darkgrey', linewidth=2.)
        axs.axvline(self.__data2d.isp[0], ls=':', c='darkgrey', linewidth=2.)
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

    def plot_params_along_LOS(self, lineweight=2.0, alpha=0.250, legend=True):

        dummy = 0

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PARAMS
        axs = self.plot_dict['param_along_LOS']['axs']
        lines = self.plot_dict['param_along_LOS']['lines']
        diag = self.plot_dict['param_along_LOS']['diag']
        chord = self.plot_dict['param_along_LOS']['chord']
        color = self.plot_dict['param_along_LOS']['color']
        zorder = self.plot_dict['param_along_LOS']['zorder']

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

        if isinstance(axs, np.ndarray) and len(axs) > 0:
            # n_e
            axs[0].semilogy(l, ne, c=color, lw=lineweight, zorder=zorder, label=r'$\mathrm{n_{e}}$')
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

        title = diag + ' R = ' + '{:3.2f}'.format(self.__res_dict[diag][chord]['chord']['p2'][0]) + ' m'

        # axs[0].set_xlim(np.min(l), np.max(l)+0.1)
        axs[0].set_xlim(5, 5.25)
        axs[0].set_title(title)

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
                          min_clip=0.0, max_clip = 1.0, max_abs = None, scal_log10=False, savefig=False):

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8))
        fig.patch.set_facecolor('white')

        if Rrng and Zrng:
            ax.set_xlim(Rrng[0], Rrng[1])
            ax.set_ylim(Zrng[0], Zrng[1])
        else:
            ax.set_xlim(1.8, 4.0)
            ax.set_ylim(-2.0, 2.0)

        cell_patches = []
        spec_line = []
        for cell in self.__data2d.cells:
            cell_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=1))
            spec_line.append(cell.H_emiss[line_key]['excit'] +
                                cell.H_emiss[line_key]['recom'])

        coll1 = PatchCollection(cell_patches, zorder=1)
        
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

        colors = plt.cm.hot(spec_line_clipped / np.max(spec_line_clipped))

        coll1.set_color(colors)
        collplt = ax.add_collection(coll1)
        ax.set_yscale
        line_wv = float(line_key) / 10.
        title = self.case + ' ' + process.at_sym[int(at_num) - 1] + ' ' + process.roman[int(ion_stage) - 1] + ' ' + '{:5.1f}'.format(
            line_wv) + ' nm'
        # title = process.at_sym[int(at_num) - 1] + ' ' + process.roman[int(ion_stage) - 1] + ' ' + '{:5.1f}'.format(
        #     line_wv) + ' nm'
        ax.set_title(title, y=1.08, fontsize='16')
        ax.set_ylabel('Z [m]')
        ax.set_xlabel('R [m]')
        plt.gca().set_aspect('equal', adjustable='box')

        # ADD COLORBAR
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', size='7%', pad=0.1)

        # Very ugly workaround to scale the colorbar without clobbering the patch collection plot
        # (https://medium.com/data-science-canvas/way-to-show-colorbar-without-calling-imshow-or-scatter)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hot,
                                   norm=plt.Normalize(vmin=cscale_min,
                                                      vmax=cscale_max))
        sm._A = []
        cbar = fig.colorbar(sm, cax=cbar_ax)
        label = '$\epsilon\mathrm{\/(ph\/s^{-1}\/m^{-3}\/sr^{-1})}$'
        cbar.set_label(label)

        # ADD DIAG LOS
        if diagLOS:
            for diag in diagLOS:
                self.__data2d.synth_diag[diag].plot_LOS(ax, color='w', lw=1.0)

        # PLOT SEPARATRIX AND WALL
        # Work around for this error message:
        # RuntimeError: Can not put single artist in more than one figure
        from copy import copy
        wallpoly = copy(self.__data2d.wall_poly)
        seppoly = copy(self.__data2d.sep_poly)
        ax.add_patch(wallpoly)
        ax.add_patch(seppoly)

        if savefig:
            plt.savefig(self.work_dir + self.case + '/' + title + '.pdf', dpi=plt.gcf().dpi)

    def plot_2d_prad(self, diagLOS, Rrng=None, Zrng=None,
                          min_clip=0.0, max_clip = 0.117, savefig=False):

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
        plt.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.95, wspace=0.05, hspace=0.1)

        # fig.patch.set_facecolor('white')

        if Rrng and Zrng:
            ax.set_xlim(Rrng[0], Rrng[1])
            ax.set_ylim(Zrng[0], Zrng[1])
        else:
            ax.set_xlim(1.8, 4.0)
            ax.set_ylim(-2.0, 2.0)

        cell_patches = []
        prad = []
        for cell in self.__data2d.cells:
            cell_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=1))
            prad.append(cell.ff_radpwr_perm3+cell.H_radpwr_perm3)
    
        prad=np.asarray(prad)*1.0e-06

        # coll1 = PatchCollection(cell_patches, cmap=matplotlib.cm.hot, norm=matplotlib.colors.LogNorm(), zorder=1, lw=0)
        # coll1 = PatchCollection(cell_patches, cmap=matplotlib.cm.hot, zorder=1, lw=0)
        coll1 = PatchCollection(cell_patches, zorder=1)
        
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
        colors = plt.cm.hot(prad_clipped / np.max(prad_clipped))

        coll1.set_color(colors)
        collplt = ax.add_collection(coll1)
        ax.set_yscale
        title = self.case 

        # ax.set_title(title)
        plt.gca().set_aspect('equal', adjustable='box')

        # ADD COLORBAR
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', size='5%', pad=0.3)

        # Very ugly workaround to scale the colorbar without clobbering the patch collection plot
        # (https://medium.com/data-science-canvas/way-to-show-colorbar-without-calling-imshow-or-scatter)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hot,
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
                self.__data2d.synth_diag[diag].plot_LOS(ax, color='w', lw=1.0)

        # PLOT SEPARATRIX AND WALL
        # Work around for this error message:
        # RuntimeError: Can not put single artist in more than one figure
        from copy import copy
        wallpoly = copy(self.__data2d.wall_poly)
        seppoly = copy(self.__data2d.sep_poly)
        ax.add_patch(wallpoly)
        ax.add_patch(seppoly)

        ax.tick_params(axis='both', labelcolor='k', top='off', bottom='off', left='off', right='off')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')


        if savefig:
            plt.savefig(self.work_dir + self.case + '/' + title + '.png', dpi=plt.gcf().dpi)


    def plot_2d_ff_fb(self, diagLOS, Rrng=None, Zrng=None, min_clip=0.0, max_clip = 1.,
                      savefig=False):

        fig, ax = plt.subplots(ncols=1, figsize=(10, 8))
        fig.patch.set_facecolor('white')
        if Rrng and Zrng:
            ax.set_xlim(Rrng[0], Rrng[1])
            ax.set_ylim(Zrng[0], Zrng[1])
        else:
            ax.set_xlim(1.8, 4.0)
            ax.set_ylim(-2.0, 2.0)

        cell_patches = []
        ff_fb_emiss = []
        for cell in self.__data2d.cells:
            cell_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=1))
            ff_fb_emiss.append(cell.ff_fb_filtered_emiss['ff_fb'])

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
        colors = plt.cm.hot(spec_line_clipped / np.max(spec_line_clipped))

        coll1 = PatchCollection(cell_patches, zorder=1)
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
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(vmin=0, vmax=np.max(ff_fb_emiss)))
        sm._A = []

        cbar = fig.colorbar(sm, cax=cbar_ax)
        label = '$\mathrm{ph\/s^{-1}\/m^{-3}\/sr^{-1}}$'
        cbar.set_label(label)

        # ADD DIAG LOS
        if diagLOS:
            for diag in diagLOS:
                self.__data2d.synth_diag[diag].plot_LOS(ax, color='w', lw=1.0)

        # PLOT SEPARATRIX AND WALL
        # Work around for this error message:
        # RuntimeError: Can not put single artist in more than one figure
        from copy import copy
        wallpoly = copy(self.__data2d.wall_poly)
        seppoly = copy(self.__data2d.sep_poly)
        ax.add_patch(wallpoly)
        ax.add_patch(seppoly)

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

    """
    JET region names:
        hfs_sol
        lfs_sol
        hfs_div
        lfs_div
        xpt_conreg
        hfs_lower
        lfs_lower
        rhon_09_10
    """

    # Example for plotting standard pyproc results

    left  = 0.2  # the left side of the subplots of the figure
    right = 0.85    # the right side of the subplots of the figure
    bottom = 0.15   # the bottom of the subplots of the figure
    top = 0.93      # the top of the subplots of the figure
    wspace = 0.18   # the amount of width reserved for blank space between subplots
    hspace = 0.1  # the amount of height reserved for white space between subplots

    fig1, ax1 = plt.subplots(nrows=3, ncols=1, figsize=(6,10), sharex=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    fig2, ax2 = plt.subplots(nrows=3, ncols=1, figsize=(6,10), sharex=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    fig3, ax3 = plt.subplots(nrows=3, ncols=1, figsize=(6,10), sharex=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # fig4, ax4 = plt.subplots(nrows=5, ncols=1, figsize=(6,10), sharex=True, sharey=True)
    # plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # fig5, ax5 = plt.subplots(nrows=2, ncols=1, figsize=(12,10), sharex=True, sharey=True)
    # plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    fig6, ax6 = plt.subplots(nrows=2, ncols=1, figsize=(6,10), sharex=True, sharey=False)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    fig7, ax7 = plt.subplots(nrows=1, ncols=1, figsize=(8,8), sharex=True, sharey=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    workdir = '/work/bloman/pyproc/'
    # case = 'bloman_cmg_catalog_edge2d_jet_81472_sep2618_seq#1'
    case = 'alexc_cmg_catalog_edge2d_jet_91554_may1217_seq#1'
    # case = 'alexc_cmg_catalog_edge2d_jet_84727_dec1416_seq#2'
    # case = 'pheliste_cmg_catalog_edge2d_jet_81472_jun1116_seq#1'
    # case = "common_cmg_jsimpson_edge2d_runs_run1706183"
    # case = 'bloman_cmg_catalog_edge2d_jet_81472_may2018_seq#1'
    # case = 'bloman_cmg_catalog_edge2d_jet_81472_mar2318_seq#3'
    # case = 'common_cmg_bloman_edge2d_runs_runpheliste_jul1816_seq2_mod'
    # case = 'mgroth_cmg_catalog_edge2d_d3d_160299_apr2618_seq#2'
    # case = 'jsimpson_cmg_catalog_edge2d_jet_85274_aug0717_seq#27_mod'

    # Hlines_dict = OrderedDict([
    #     ('1215.2', ['2', '1']),
    #     ('6561.9', ['3', '2']),
    #     ('4339.9', ['5', '2']),
    # ])

    Hlines_dict = OrderedDict([
        ('1215.2', ['2', '1']),
        ('6561.9', ['3', '2']),
        # ('4339.9', ['5', '2']),
        # ('4101.2', ['6', '2']),
        ('3969.5', ['7', '2'])
    ])

    nitrogen_lines_dict = OrderedDict([
        ('2', {
            # '4042.07': ['4f', '3d'],
            '3996.13': ['4f', '3d']
            # '5002.18': ['3d', '3p'],
            # '5005.86': ['3d', '3p']
        }),
        # ('3', {'4100.51':['3p', '3s']}),
        # ('4', {'4058.90':['3d', '3p']})
    ])

    beryllium_lines_dict = Ord002eredDict([
        ('2', {'5272.32':['4s', '3p']})
    ])

    neon_lines_dict = OrderedDict([
        ('2', {'3718.2':['3p', '3s']})
    ])

    tungsten_lines_dict = OrderedDict([
        ('1', {'4053.65':['64a52b56c54d51g51h', '60964a52b56c54d52g']}),
        ('2', {'3604.25':['60964a52b56c54d51h', '60964a52b56c54d51g']}),
    ])

    spec_line_dict = OrderedDict([
        ('1', {'1': Hlines_dict}),
        # ('4', beryllium_lines_dict),
        # ('7', nitrogen_lines_dict)
        # ('10', neon_lines_dict)
        # ('74', tungsten_lines_dict)
    ])

    plot_dict = {
        'spec_line_dict':spec_line_dict,
        # 'prof_param_defs':{'diag': 'KT1V', 'axs': ax1,
        #                    'include_pars_at_max_ne_along_LOS': True,
        #                    'include_sum_Sion_Srec': True,
        #                    'include_target_vals': True,
        #                    'Sion_H_transition':[[2,1], [3,2]],
        #                    'Srec_H_transition':[[7,2]],
        #                    'coord': 'R', # 'angle' 'R' 'Z'
        #                    'color': 'blue', 'zorder': 10},
        'prof_Hemiss_defs':{'diag': 'KT1V',
                            'lines': spec_line_dict['1']['1'],
                            'excrec': True,
                            'axs': ax2,
                            'coord': 'R', # 'angle' 'R' 'Z'
                            'color': 'b',
                            'zorder': 10},
        # 'prof_Prad_defs': {'diag': ['KB5V'], # Allows lists for combined output
        #                    'axs': ax7,
        #                    'coord': 'angle',  # 'angle' 'R' 'Z'
        #                    'color': 'b',
        #                    'write_ppf':False,
        #                    'zorder': 10},
        # 'prof_impemiss_defs':{'diag': 'KT1V',
        #                       'lines': spec_line_dict,
        #                       'excrec': False,
        #                       'coord': 'R', # 'angle' 'R' 'Z'
        #                       'axs': ax3,
        #                       'color': ['r', 'g'],
        #                       'zorder': 10},
        # 'imp_rad_coeff': {'region': 'vessel',
        #                   'atnum': 7,
        #                   'ion_stages': [1, 2, 3, 4],
        #                   'axs': ax4,
        #                   'color': 'r',
        #                   'zorder': 10},
        # 'imp_rad_dist': {'region': 'lfs_div',
        #                   'atnum': 7,
        #                   'te_nbins': 10,
        #                   'axs': ax5,
        #                   'norm':False,
        #                   'ion_stage':1,
        #                   'color': 'r',
        #                   'zorder': 10},
        # 'nii_adas_afg': { 'axs': ax6,
        #                   'writecsv': True,
        #                   'color': 'r',
        #                   'zorder': 10},
        # 'los_param_defs':{'diag':'KT3', 'axs':ax1, 'color':'blue', 'zorder':10},
        # 'los_Hemiss_defs':{'diag':'KT3', 'axs':ax1, 'color':'blue', 'zorder':10},
        # 'param_along_LOS': {'diag': 'KT3', 'chord': '6', 'axs': ax6,
        #                     'lines': spec_line_dict,
        #                     # 'ylim': [[0, 6e20], [0, 7]],  # , [0, 0.08]],#,[1.e21, 1.e23]],#,[1.e16,1.e20]],
        #                     # 'xlim': [2.55, 2.8],
        #                     'color': 'b',
        #                     'zorder': 10},
        # '2d_defs': {'lines': spec_line_dict, 'diagLOS': ['KT3'], 'Rrng': [2.36, 2.96], 'Zrng': [-1.73, -1.29], 'save': True},
        '2d_defs': {'lines': spec_line_dict, 'diagLOS': [], 'Rrng': [2.3, 3.0], 'Zrng': [-1.76, -1.3], 'save': True},
        # '2d_prad': {'diagLOS': [], 'Rrng': [2.31, 3.0], 'Zrng': [-1.75, -1.0], 'save': False}
    }

    pyproc_case = Plot(workdir, case, plot_dict=plot_dict)
    # print available regions
    print('Region powers: ', case)
    for name, region in pyproc_case.data2d.regions.items():
        print('Region, Prad_H, Prad_imp1, Prad_imp2: ', name, region.Prad_H, region.Prad_imp1, region.Prad_imp2)
    print('')

    # pyproc_case.write_B3X4_ppf(90425, 'B3D4', tstart=52.0, wUid=None)
    # pyproc_case.write_B3X4_ppf(90425, 'B3E4', tstart=52.0, wUid=None)

    # Example for plotting cherab_bridge results

    Hlines_dict = OrderedDict([
        ('1215.2', ['2', '1']),
        ('6561.9', ['3', '2']),
        # ('4339.9', ['5', '2']),
        # ('4101.2', ['6', '2']),
        # ('3969.5', ['7', '2'])
    ])

    spec_line_dict = OrderedDict([
        ('1', {'1': Hlines_dict}),
    ])

    plot_dict = {
        'spec_line_dict':spec_line_dict,
        'cherab_bridge_results': True,
        'cherab_reflections': False,
        'cherab_abs_factor': {'1215.2': 115., '6561.9': 615.},
        # 'prof_param_defs': {'diag': 'KT3', 'axs': ax1,
        #                     'include_pars_at_max_ne_along_LOS': False,
        #                     'include_sum_Sion_Srec': False,
        #                     'include_target_vals': False,
        #                     'coord': 'R',  # 'angle' 'R' 'Z'
        #                     'color': 'g', 'zorder': 10},
        'prof_Hemiss_defs':{'diag': 'KT3',
                            'lines': spec_line_dict['1']['1'],
                            'excrec': True,
                            'axs': ax2,
                            'coord': 'R', # 'angle' 'R' 'Z'
                            'color': 'g',
                            'zorder': 10},
    }

    # cherab_bridge_case = Plot(workdir, case, plot_dict=plot_dict)

    plot_dict = {
        'spec_line_dict':spec_line_dict,
        'cherab_bridge_results': True,
        'cherab_reflections': True,
        'cherab_abs_factor': {'1215.2': 115., '6561.9': 615.},
        'prof_param_defs': {'diag': 'KT3', 'axs': ax1,
                            'include_pars_at_max_ne_along_LOS': False,
                            'include_sum_Sion_Srec': False,
                            'include_target_vals': False,
                            'coord': 'R',  # 'angle' 'R' 'Z'
                            'color': 'r', 'zorder': 10},
        'prof_Hemiss_defs':{'diag': 'KT3',
                            'lines': spec_line_dict['1']['1'],
                            'excrec': True,
                            'axs': ax2,
                            'coord': 'R', # 'angle' 'R' 'Z'
                            'color': 'r',
                            'zorder': 10},
    }

    # cherab_bridge_case = Plot(workdir, case, plot_dict=plot_dict)

    # Print out results dictionary tree
    # Plot.pprint_json(pyproc_case.res_dict['KT3']['1']['los_1d'])
    # Plot.pprint_json(o.res_dict['KT3']['1'])

    plt.show()
