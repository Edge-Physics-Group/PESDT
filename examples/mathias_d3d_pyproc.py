
# Import external modules
from pyproc.plot import Plot
import matplotlib
matplotlib.use('TKAgg', force=True)
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

def case_defs_d3d():

    cases = {
        '1': {
            'sim_color': 'b', 'sim_zo': 1,
            'case': 'mgroth_cmg_catalog_edge2d_d3d_160299_apr2618_seq#2',
        },
    }

    return cases


if __name__=='__main__':

    workdir = '/work/bloman/pyproc/'
    cases = case_defs_d3d()
    savefig=True

    # setup plot figures
    left  = 0.2  # the left side of the subplots of the figure
    right = 0.85    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.93      # the top of the subplots of the figure
    wspace = 0.25   # the amount of width reserved for blank space between subplots
    hspace = 0.15  # the amount of height reserved for white space between subplots

    Hlines_dict = OrderedDict([
        ('1215.2', ['2', '1']),
        ('6561.9', ['3', '2']),
        ('4339.9', ['5', '2']),
        ('4101.2',['6', '2']),
        ('3969.5', ['7', '2'])
    ])

    carbon_lines_dict = OrderedDict([
        ('2', {'5143.3': ['3p', '3s']}),
        ('3', {'4650.1': ['3p', '3s']}),
        ('4', {'1549.1': ['2p', '2s']})
    ])

    spec_line_dict = {
        '1':  # HYDROGEN
            {'1': Hlines_dict},
        '6': carbon_lines_dict
    }

    fig1, ax1 = plt.subplots(nrows=len(Hlines_dict), ncols=1, figsize=(6, 12), sharex=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    fig2, ax2 = plt.subplots(nrows=len(carbon_lines_dict), ncols=1, figsize=(6, 12), sharex=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), sharex=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    for casekey, case in cases.items():

        plot_dict = {
            'spec_line_dict':spec_line_dict,
            'prof_Hemiss_defs': {'diag': 'divspred',
                                 'lines': spec_line_dict['1']['1'],
                                 'excrec': True,
                                 'axs': ax1,
                                 # 'ylim': [[0, 5e21], [0, 2.5e20], [0, 3.5e17]],
                                 # 'xlim': [2.15, 2.90],
                                 'coord': 'R',# 'R' 'Z'
                                 'color': case['sim_color'],
                                 'zorder': 10,
                                 'write_csv': False},
            'prof_impemiss_defs': {'diag': 'divspred',
                                   'lines': spec_line_dict,
                                   'excrec': True,
                                   'axs': ax2,
                                   # 'ylim': [[0, 8e17], [0, 2e18]],# [0, 8e18]],
                                   # 'xlim': [2.55, 2.8],
                                   'coord': 'R',# 'R' 'Z'
                                   'color': [case['sim_color']],
                                   'zorder': case['sim_zo'],
                                   'write_csv': False},
            # 'prof_Prad_defs': {'diag': ['bolo1_hr'], # must be a list!!!
            #                    'axs': ax3,
            #                    'coord': 'angle',  # 'angle' 'R' 'Z'
            #                    'color': 'b',
            #                    'zorder': 10,
            #                    'write_csv': False},
            '2d_defs': {'lines': spec_line_dict,
                        'diagLOS': ['divspred'],
                        'Rrng': [1.0, 2.0],
                        'Zrng': [-1.5, -0.5],
                        'save': False},
        }

        o = Plot(workdir, case['case'], plot_dict=plot_dict)

    # Print out results dictionary tree
    # Plot.pprint_json(o.res_dict['mds1_hr']['1']['los_int'])

    plt.show()