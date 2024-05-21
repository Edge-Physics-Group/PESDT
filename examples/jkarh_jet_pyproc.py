
# Import external modules
from pyproc.plot import Plot
import matplotlib
# matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.io as io
# from pyJETPPF.JETPPF import JETPPF
from pyproc.analyse import AnalyseSynthDiag
from collections import OrderedDict

font = {'family': 'normal',
        'weight': 'normal',
        'size': 14}
matplotlib.rc('font', **font)
import matplotlib.font_manager as font_manager
# path = '/usr/share/fonts/msttcore/arial.ttf'
path = '/usr/share/fonts/gnu-free/FreeSans.ttf'
# path = '/home/bloman/fonts/msttcore/arial.ttf'
prop = font_manager.FontProperties(fname=path)
matplotlib.rcParams['font.family'] = prop.get_name()
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = prop.get_name()
matplotlib.rc('lines', linewidth=1.2)
matplotlib.rc('axes', linewidth=1.2)
matplotlib.rc('xtick.major', width=1.2, pad=7)
matplotlib.rc('ytick.major', width=1.2, pad=7)
matplotlib.rc('xtick.minor', width=1.2)
matplotlib.rc('ytick.minor', width=1.2)

def case_defs_jet():

    #~ cases = {
        #~ '1': {
            #~ 'sim_color': 'b', 'sim_zo': 1,
            #~ 'case': 'common_cmg_jkarh_edge2d_runs_run_k_G=1.23e22_D1_Ddiv=1.0_v121218',
        #~ },
        #~ '2': {
            #~ 'sim_color': 'r', 'sim_zo': 1,
            #~ 'case': 'common_cmg_jkarh_edge2d_runs_run_k_G=1.255e22_D9_bv=1_v1061_Ddiv=1.0_vdiv1061',
        #~ },
        #~ '3': {
            #~ 'sim_color': 'g', 'sim_zo': 1,
            #~ 'case': 'common_cmg_jkarh_edge2d_runs_run_k_G=1.31e22_D19_bv=1_v1094_Ddiv=1.0_vdiv1094',
        #~ },
        #~ '4': {
            #~ 'sim_color': 'm', 'sim_zo': 1,
            #~ 'case': 'common_cmg_jkarh_edge2d_runs_run_k_G=1.52e22_D20_bv=1_v1117_Ddiv=1.0_vdiv1117',
        #~ },       
    #~ }
    
    cases = {
        '1': {
            'sim_color': 'b', 'sim_zo': 1,
            'case': 'common_cmg_jkarh_edge2d_runs_run_d_G=1.75e21_D1_Ddiv=1.0_v121218',
        },
        '2': {
            'sim_color': 'r', 'sim_zo': 1,
            'case': 'common_cmg_jkarh_edge2d_runs_run_d_G=1.83e21_D9_bv=1_v1007_Ddiv=1.0_vdiv1007',
        },
        '3': {
            'sim_color': 'g', 'sim_zo': 1,
            'case': 'common_cmg_jkarh_edge2d_runs_run_d_G=1.83e21_D19_bv=1_v990_Ddiv=1.0_vdiv990',
        },
        '4': {
            'sim_color': 'm', 'sim_zo': 1,
            'case': 'common_cmg_jkarh_edge2d_runs_run_d_G=1.90e21_D20_bv=1_v1019_Ddiv=1.0_vdiv1019',
        },       
    }

    return cases


if __name__=='__main__':

    workdir = '/work/jkarh/pyproc/'
    cases = case_defs_jet()

    # setup plot figures
    left  = 0.2  # the left side of the subplots of the figure
    right = 0.85    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.93      # the top of the subplots of the figure
    wspace = 0.25   # the amount of width reserved for blank space between subplots
    hspace = 0.15  # the amount of height reserved for white space between subplots

    Hlines_dict = OrderedDict([
        #('1215.2', ['2', '1']),
        ('6561.9', ['3', '2']),
        #('4339.9', ['5', '2']),
        #('4101.2',['6', '2']),
        ('3969.5', ['7', '2'])
    ])

    spec_line_dict = {
        '1':  # HYDROGEN
            {'1': Hlines_dict}
    }

    fig0, ax0 = plt.subplots(nrows=3, ncols=1, figsize=(6, 12), sharex=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    
    fig1, ax1 = plt.subplots(nrows=len(Hlines_dict), ncols=1, figsize=(6, 12), sharex=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    # fig2, ax2 = plt.subplots(nrows=len(carbon_lines_dict), ncols=1, figsize=(6, 12), sharex=True)
    # plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    #fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), sharex=True)
    #plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    for casekey, case in cases.items():

        plot_dict = {
            'spec_line_dict':spec_line_dict,
			'prof_param_defs':{'diag': 'KT1V', 'axs': ax0,
								'include_pars_at_max_ne_along_LOS': False,
								'include_sum_Sion_Srec': False,
								'include_target_vals': False,
								'Sion_H_transition':[[2,1], [3,2]],
								'Srec_H_transition':[[7,2]],
								'coord': 'R', # 'angle' 'R' 'Z'
								'color': case['sim_color'], 'zorder': 10},
								#'color': 'blue', 'zorder': 10},        
            'prof_Hemiss_defs': {'diag': 'KT1V',
                                 'lines': spec_line_dict['1']['1'],
                                 'excrec': True,
                                 'axs': ax1,
                                 # 'ylim': [[0, 5e21], [0, 2.5e20], [0, 3.5e17]],
                                 # 'xlim': [2.15, 2.90],
                                 'coord': 'R',# 'angle' 'R' 'Z'
                                 'color': case['sim_color'],
                                 'zorder': 10,
                                 'write_csv': False},
            # 'prof_impemiss_defs': {'diag': 'KT1V',
            #                        'lines': spec_line_dict,
            #                        'excrec': True,
            #                        'axs': ax2,
            #                        'ylim': [[0, 8e17], [0, 2e18]],# [0, 8e18]],
            #                        # 'xlim': [2.55, 2.8],
            #                        'coord': 'angle',# 'R' 'Z'
            #                        'color': [case['sim_color']],
            #                        'zorder': case['sim_zo'],
            #                        'write_csv': False},
            # 'prof_Prad_defs': {'diag': ['KB5V'], # must be a list!!!
            #                    'axs': ax3,
            #                    'coord': 'angle',  # 'angle' 'R' 'Z'
            #                    'color': 'b',
            #                    'zorder': 10,
            #                    'write_csv': False},
            #'2d_defs': {'lines': spec_line_dict,
            #            'diagLOS': [],
            #            'Rrng': [2.36, 2.96],
            #            'Zrng': [-1.73, -1.29],
            #            #'max_emiss':1.e22, # adjust color scale
            #            'save': False},
        }

        o = Plot(workdir, case['case'], plot_dict=plot_dict)

    # Print out results dictionary tree
    # Plot.pprint_json(o.res_dict['mds1_hr']['1']['los_int'])

    plt.show()
