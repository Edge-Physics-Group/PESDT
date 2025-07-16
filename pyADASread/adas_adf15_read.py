# -*- coding: utf-8 -*-

import numpy as np
from scipy import interpolate
try:
    from adaslib import *
except ModuleNotFoundError as e:
    print(f"Adaslib unavailable, Error: {e}")
import json, os

# MAIN H SET
line_blocks = {'1215.2excit':1, #LYMAN series exc
               '1025.3excit':2,
               '972.1excit':4,
               '1215.2recom':67, #LYMAN series rec
               '1025.3recom':68,
               '972.1recom':70,
               '6561.9excit':3,  # Balmer series
               '4860.6excit':5,
               '4339.9excit':8,
               '4101.2excit':12,
               '3969.5excit':17,
               '3888.5excit':23,
               '3834.9excit':30,
               '3797.4excit':38,
               '6561.9recom':69,
               '4860.6recom':71,
               '4339.9recom':74,
               '4101.2recom':78,
               '3969.5recom':83,
               '3888.5recom':89,
               '3834.9recom':96,
               '3797.4recom':104}

# Source: Wiese and Fuhr 2009
Arates = {'1215.2':4.6986e08,
           '1025.3':5.5751e07,
           '972.1':1.2785e07,
           '6561.9':4.4101e07,
           '4860.6':8.4193e06,
           '4339.9':2.5304e06,
           '4101.2':9.732e05,
           '3969.5':4.3889e05}

# Reduced set up to n=5 (opacity check)
# C      1   949.743    5(2)0( 24.5)-  1(2)0(  0.5)       EXCIT   1    1   0     4  6  1
# C      2   972.537    4(2)0( 15.5)-  1(2)0(  0.5)       EXCIT   1    1   0     3  4  2
# C      3   1025.72    3(2)0(  8.5)-  1(2)0(  0.5)       EXCIT   1    1   0     2  2  3
# C      4   1215.67    2(2)0(  3.5)-  1(2)0(  0.5)       EXCIT   1    1   0     1  1  4
# C      5   4341.67    5(2)0( 24.5)-  2(2)0(  3.5)       EXCIT   1    1   0     7  7  5
# C      6   4862.65    4(2)0( 15.5)-  2(2)0(  3.5)       EXCIT   1    1   0     6  5  6
# C      7   6564.57    3(2)0(  8.5)-  2(2)0(  3.5)       EXCIT   1    1   0     5  3  7
# C      8   949.743    5(2)0( 24.5)-  1(2)0(  0.5)       RECOM   1    1   0     4  6  1
# C      9   972.537    4(2)0( 15.5)-  1(2)0(  0.5)       RECOM   1    1   0     3  4  2
# C     10   1025.72    3(2)0(  8.5)-  1(2)0(  0.5)       RECOM   1    1   0     2  2  3
# C     11   1215.67    2(2)0(  3.5)-  1(2)0(  0.5)       RECOM   1    1   0     1  1  4
# C     12   4341.67    5(2)0( 24.5)-  2(2)0(  3.5)       RECOM   1    1   0     7  7  5
# C     13   4862.65    4(2)0( 15.5)-  2(2)0(  3.5)       RECOM   1    1   0     6  5  6
# C     14   6564.57    3(2)0(  8.5)-  2(2)0(  3.5)       RECOM   1    1   0     5  3  7
line_blocks_n5 = {'949.743excit':1,
               '972.537excit':2,
               '1025.72excit':3,
               '1215.67excit':4,
               '4341.67excit':5,
               '4862.65excit':6,
               '6564.57excit':7,
               '949.743recom':8,
               '972.537recom':9,
               '1025.72recom':10,
               '1215.67recom':11,
               '4341.67recom':12,
               '4862.65recom':13,
               '6564.57recom':14}

# line_blocks_n5 = {'1215.67excit':4}

Arates_n5 = {'949.743':4.16e06,
               '972.537':1.28e07,
               '1025.72':5.78e07,
               '1215.67':4.7e08,
               '4341.67':2.53e06,
               '4862.65':8.42e06,
               '6564.57':4.41e07}

Ly_beta_esc_fac_cases = {
    '1':{'Ly_beta_esc_fac':0.80, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.80.dat'},
    '2':{'Ly_beta_esc_fac':0.70, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.70.dat'},
    '3':{'Ly_beta_esc_fac':0.65, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.65.dat'},
    '4':{'Ly_beta_esc_fac':0.60, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.60.dat'},
    '5':{'Ly_beta_esc_fac':0.55, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.55.dat'},
    '6':{'Ly_beta_esc_fac':0.50, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.50.dat'},
    '7':{'Ly_beta_esc_fac':0.45, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.45.dat'},
    '8':{'Ly_beta_esc_fac':0.40, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.40.dat'},
    '9':{'Ly_beta_esc_fac':0.35, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.35.dat'},
    '10':{'Ly_beta_esc_fac':0.30, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.30.dat'},
    '11':{'Ly_beta_esc_fac':0.25, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.25.dat'},
    '12':{'Ly_beta_esc_fac':0.20, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.20.dat'},
    '13':{'Ly_beta_esc_fac':0.18, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.18.dat'},
    '14':{'Ly_beta_esc_fac':0.16, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.16.dat'},
    '15':{'Ly_beta_esc_fac':0.14, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.14.dat'},
    '16':{'Ly_beta_esc_fac':0.12, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.12.dat'},
    '17':{'Ly_beta_esc_fac':0.10, 'adf15_file':'/home/bloman/python_tools/pyADASread/adas_data/pec16_h_Ly_beta_esc_fac.10.dat'},
}

# impurity line blocks dictionary: nuclear charge->ionisation stage
# Also includes Hydrogen!
# imp_line_blocks = {
#     '1':
#         {'1': {'file': '/home/adas/adas/adf15/pec12#h/pec12#h_pju#h0.dat',
#                'blocks': line_blocks}},
#     '4': # Be
#         {'2': {'file': '/home/adas/adas/adf15/pec96#be/pec96#be_pju#be1.dat',
#                'blocks': {'5272.32excit': 11, '5272.32recom': 32}}},
#     '7': # N
#         ###################################################
#         # Use Stuart Henderson's latest PEC file for n1 as stage index -1
#         {'-1': {'file': '/home/shenders/adas/adas/adf15/pec16#7/pec98#n_ssh_pju#n1.dat',
#                 'blocks': {'3996.13excit': 15, '3996.13recom': 65,
#                            '4042.07excit': 21, '4042.07recom': 71}},
#          ###################################################
#          '2':{'file':'/home/adas/adas/adf15/pec96#n/pec96#n_vsu#n1.dat',
#               'blocks':{'5002.18excit': 17, '5002.18recom': 67,
#                         '5005.86excit': 4, '5005.86recom': 54}},
#          '3':{'file':'/home/adas/adas/adf15/pec96#n/pec96#n_vsu#n2.dat',
#               'blocks':{'4100.51excit':1, '4100.51recom':51}},
#          '4':{'file':'/home/adas/adas/adf15/pec96#n/pec96#n_vsu#n3.dat',
#               'blocks':{'3481.83excit':10, '3481.83recom':51,
#                         '4058.90excit':17, '4058.90recom':58}}},
#     '6': # C
#         {'1':{'file':'/home/adas/adas/adf15/pec96#c/pec96#c_vsu#c0.dat',
#               'blocks':{'9408.4excit':32, '9408.4recom':81}},
#          '2':{'file':'/home/adas/adas/adf15/pec96#c/pec96#c_vsu#c1.dat',
#               'blocks':{'6581.5excit':1, '6581.5recom':51},
#               'blocks':{'5143.3excit':12, '5143.3recom':62}},
#          '3':{'file':'/home/adas/adas/adf15/pec96#c/pec96#c_vsu#c2.dat',
#               'blocks':{'4650.1excit':2, '4650.1recom':52}},
#          '4':{'file':'/home/adas/adas/adf15/pec96#c/pec96#c_pju#c3.dat',
#                'blocks': {'1549.1excit': 14, '1549.1recom': 55}}},
#     '10': # Ne
#         {'2': {'file': '/home/adas/adas/adf15/pec96#ne/pec96#ne_pju#ne1.dat',
#                'blocks': {'3718.2excit': 17, '3718.2recom': 70}}},
#     '74': # W
#         {'1': {'file': '/home/adas/adas/adf15/pec40#w/pec40#w_ls#w0.dat',
#                'blocks': {'3567.64excit': 44},
#                'blocks': {'3674.90excit': 45},
#                'blocks': {'4053.65excit': 48}},
#          '2': {'file': '/home/adas/adas/adf15/pec40#w/pec40#w_ls#w1.dat',
#                'blocks': {'3604.25excit': 50}}}
# }

line_blocks_lytrap = {
    '1':
        {'1': {'file': '/home/bloman/idl/adas/h_rates/pec16_h0_supress.dat',
               'blocks': line_blocks_n5}},
}

def get_adf15_input_dict(file):
    # Read adf15 input dict
    try:
        with open(file, mode='r', encoding='utf-8') as f:
            # Remove comments
            with open("temp.json", 'w') as wf:
                for line in f.readlines():
                    if line[0:2] == '//' or line[0:1] == '#':
                        continue
                    wf.write(line)
        with open("temp.json", 'r') as f:
            imp_line_blocks = json.load(f)
        os.remove('temp.json')
        return imp_line_blocks
    except IOError as e:
                raise

class PEC:
    def __init__(self, block, pec, Te_arr, ne_arr, info):
        self.block = block # ADAS block composed of wave + excit/recom
        self.pec = pec # 2D PEC array as fn(Te,ne)
        self.Te_arr = Te_arr  # copy of Te array
        self.ne_arr = ne_arr # copy of ne array
        self.info = info # info dict returned by read_adf15

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def get_adas_imp_PECs(imp_dict, imp_line_blocks, Te_arr, ne_arr):

    PEC_dict = dict()
    for species in imp_dict.keys():
        PEC_dict[species] = {}
        for stage in imp_dict[species]:
            PEC_dict[species][stage] = {}
            for key in imp_line_blocks[species][stage]['blocks']:
                coeff, info = get_imp_adf15_block(imp_line_blocks, Te_arr, ne_arr, species, stage, key, imp_line_blocks[species][stage]['file'])
                PEC_dict[species][stage][key] = PEC(key, coeff, Te_arr, ne_arr, info)

    return PEC_dict

def get_adas_imp_PECs_interp(imp_dict, imp_line_blocks, Te_rnge, ne_rnge, npts=100, npts_interp=1000, lytrap_pec_file=False):

    if lytrap_pec_file:
        line_blocks = line_blocks_lytrap
        line_blocks['1']['1']['file'] = lytrap_pec_file
        lytrap = True
    else:
        line_blocks = imp_line_blocks
        lytrap = False

    # NOTE: interp2d only seems to work when x and y have equal num of points
    Te_arr = np.logspace(np.log10(Te_rnge[0]), np.log10(Te_rnge[1]), npts)
    ne_arr = np.logspace(np.log10(ne_rnge[0]), np.log10(ne_rnge[1]), npts)

    PEC_dict = dict()
    for species in imp_dict.keys():
        PEC_dict[species] = {}
        for stage in imp_dict[species]:
            PEC_dict[species][stage] = {}
            for key in imp_dict[species][stage]:

                # Logic to handle Stuart Henderson's custom N II pec file with stage index -1
                # e.g.,
                # "7": {
                #     "2": {
                #         "4042.07": ["4f","3d","-1"] last item corresponds to custom stage key
                #     }}}
                if len(imp_dict[species][stage][key]) > 2:
                    line_block_stage = imp_dict[species][stage][key][2]
                else:
                    line_block_stage = stage

                if key+'excit' in line_blocks[species][line_block_stage]['blocks']:
                    pec_excit, info = get_imp_adf15_block(imp_line_blocks, Te_arr, ne_arr, species, line_block_stage,
                                                          key+'excit', line_blocks[species][line_block_stage]['file'], lytrap=lytrap)
                else:
                    pec_excit = np.zeros((len(Te_arr), len(ne_arr)))
                    info=None
                    print(key+'excit', ' entry does not exist, setting PEC array to zero')
                if key+'recom' in line_blocks[species][line_block_stage]['blocks']:
                    pec_recom, info = get_imp_adf15_block(imp_line_blocks, Te_arr, ne_arr, species, line_block_stage,
                                                          key+'recom', line_blocks[species][line_block_stage]['file'], lytrap=lytrap)
                else:
                    pec_recom = np.zeros((len(Te_arr), len(ne_arr)))
                    info=None
                    print(key+'excit', ' entry does not exist, setting PEC array to zero')

                # linearly interpolate results on finer Te ne grid
                f_excit = interpolate.interp2d(Te_arr, ne_arr, pec_excit, kind='linear')
                f_recom = interpolate.interp2d(Te_arr, ne_arr, pec_recom, kind='linear')
                Te_fine = np.logspace(np.log10(Te_arr[0]), np.log10(Te_arr[-1]), npts_interp)
                ne_fine = np.logspace(np.log10(ne_arr[0]), np.log10(ne_arr[-1]), npts_interp)
                pec_excit_interp = f_excit(Te_fine, ne_fine)
                pec_recom_interp = f_recom(Te_fine, ne_fine)
                PEC_dict[species][stage][key+'excit'] = PEC(key+'excit', pec_excit_interp, Te_fine, ne_fine, info)
                PEC_dict[species][stage][key+'recom'] = PEC(key+'recom', pec_recom_interp, Te_fine, ne_fine, info)

    return PEC_dict

def get_adas_H_PECs(Te_arr, ne_arr):


    PEC_dict = dict()
    for key in line_blocks:
        coeff, info = get_H_adf15_block(Te_arr, ne_arr, key)
        PEC_dict[key] = PEC(key, coeff, Te_arr, ne_arr, info)

    return PEC_dict

def get_adas_H_PECs_interp(line_dict, Te_rnge, ne_rnge, npts=100, npts_interp=1000):

    # NOTE: interp2d only seems to work when x and y have equal num of points
    Te_arr = np.logspace(np.log10(Te_rnge[0]), np.log10(Te_rnge[1]), npts)
    ne_arr = np.logspace(np.log10(ne_rnge[0]), np.log10(ne_rnge[1]), npts)

    PEC_dict = dict()
    for key in line_dict:
        # excitation contribution
        pec_excit, info = get_H_adf15_block(Te_arr, ne_arr, key+'excit')
        # recombination contribution
        pec_recom, info = get_H_adf15_block(Te_arr, ne_arr, key+'recom')

        # sanity check: coarse vs interp
        # ne = 1e14
        # ne_idx, ne_val = find_nearest(ne_arr, ne)
        # plt.loglog(Te_arr, pec_excit[:, ne_idx], 'or')
        # plt.loglog(Te_arr, pec_recom[:, ne_idx], 'ob')

        # linearly interpolate results on finer Te ne grid
        f_excit = interpolate.interp2d(Te_arr, ne_arr, pec_excit, kind='linear')
        f_recom = interpolate.interp2d(Te_arr, ne_arr, pec_recom, kind='linear')
        Te_fine = np.logspace(np.log10(Te_arr[0]), np.log10(Te_arr[-1]), npts_interp)
        ne_fine = np.logspace(np.log10(ne_arr[0]), np.log10(ne_arr[-1]), npts_interp)
        pec_excit_interp = f_excit(Te_fine, ne_fine)
        pec_recom_interp = f_recom(Te_fine, ne_fine)
        PEC_dict[key+'excit'] = PEC(key+'excit', pec_excit_interp, Te_fine, ne_fine, info)
        PEC_dict[key+'recom'] = PEC(key+'recom', pec_recom_interp, Te_fine, ne_fine, info)

        # sanity check: coarse vs interp
        # ne_idx, ne_val = find_nearest(PEC_dict[key+'excit'].ne_arr, ne)
        # plt.loglog(Te_fine, PEC_dict[key+'excit'].pec[:, ne_idx], '.-r')
        # plt.loglog(Te_fine, PEC_dict[key+'recom'].pec[:, ne_idx], '.-b')
        # plt.show()

    return PEC_dict

def get_adas_H_PECs_n5(Te_arr, ne_arr, Ly_beta_esc_fac=1.0):


    PEC_dict = dict()
    for key in line_blocks_n5:
        coeff, info = get_H_adf15_block_n5(Te_arr, ne_arr, key, Ly_beta_esc_fac)
        PEC_dict[key] = PEC(key, coeff, Te_arr, ne_arr, info)

    return PEC_dict

def get_H_adf15_block(Te_arr, ne_arr, block_key):

    file = '/home/adas/adas/adf15/pec12#h/pec12#h_pju#h0.dat'

    # opacity check - up to n=5 only
    # file = '/home/bloman/python_bal/analysis/adas_data/pec12_h0_supress.dat'

    #Te_fine = np.logspace(np.log10(0.2), np.log10(100), 10)
    #ne_fine = np.logspace(np.log10(1.0e12), np.log10(1.0e15), 5)

    # return 2D coeff(te, dens) in units ph s-1 cm3
    if block_key in line_blocks:
        print('Getting ADF15 data ', block_key, ' from read_adf15...')
        coeff, info = read_adf15(file=file, block=line_blocks[block_key], te=Te_arr, dens=ne_arr, all=True, return_info = True)
    else:
        print(block_key, ' entry does not exist')


    return coeff, info

def get_imp_adf15_block(imp_line_blocks, Te_arr, ne_arr, nuc_charge, ion_stage, block_key, adf15file, lytrap=False):

    # return 2D coeff(te, dens) in units ph s-1 cm3
    if lytrap:
        if block_key in line_blocks_lytrap[nuc_charge][ion_stage]['blocks']:
            try:
                f = open(adf15file)
                f.close()
            except FileNotFoundError:
                print('File', adf15file, ' not found.')
                raise
            print('Getting ADF15 data ', adf15file, nuc_charge, ion_stage, block_key, ' from read_adf15...')
            coeff, info = read_adf15(file=adf15file, block=line_blocks_lytrap[nuc_charge][ion_stage]['blocks'][block_key], te=Te_arr, dens=ne_arr, all=True, return_info = True)
            return coeff, info
        else:
            print(block_key, ' entry does not exist')
    else:
        if block_key in imp_line_blocks[nuc_charge][ion_stage]['blocks']:
            try:
                f = open(adf15file)
                f.close()
            except FileNotFoundError:
                print('File', adf15file, ' not found.')
                raise
            print('Getting ADF15 data ', adf15file, nuc_charge, ion_stage, block_key, ' from read_adf15...')
            coeff, info = read_adf15(file=adf15file, block=imp_line_blocks[nuc_charge][ion_stage]['blocks'][block_key], te=Te_arr, dens=ne_arr, all=True, return_info = True)
            return coeff, info
        else:
            print(block_key, ' entry does not exist')




def get_H_adf15_block(Te_arr, ne_arr, block_key):

    file = '/home/adas/adas/adf15/pec12#h/pec12#h_pju#h0.dat'

    # opacity check - up to n=5 only
    # file = '/home/bloman/python_bal/analysis/adas_data/pec12_h0_supress.dat'

    #Te_fine = np.logspace(np.log10(0.2), np.log10(100), 10)
    #ne_fine = np.logspace(np.log10(1.0e12), np.log10(1.0e15), 5)

    # return 2D coeff(te, dens) in units ph s-1 cm3
    if block_key in line_blocks:
        print('Getting ADF15 data ', block_key, ' from read_adf15...')
        coeff, info = read_adf15(file=file, block=line_blocks[block_key], te=Te_arr, dens=ne_arr, all=True, return_info = True)
    else:
        print(block_key, ' entry does not exist')


    return coeff, info

def get_H_adf15_block_n5(Te_arr, ne_arr, block_key, Ly_beta_esc_fac):

    # opacity check - up to n=5 only

    if Ly_beta_esc_fac < 0.9:
        # GET THE APPROPRIATE HI AND LO PEC FILES ENCLOSING THE INPUT LY_BETA_ESC_FAC, THEN INTERPOLATE LINEARLY
        tmp_esc_fac_hi = 1.0
        tmp_esc_fac_lo = 0.0
        case_hi = None
        case_lo = None
        for key in Ly_beta_esc_fac_cases.keys():
            if Ly_beta_esc_fac <= Ly_beta_esc_fac_cases[key]['Ly_beta_esc_fac'] and Ly_beta_esc_fac_cases[key]['Ly_beta_esc_fac'] <= tmp_esc_fac_hi:
                tmp_esc_fac_hi = Ly_beta_esc_fac_cases[key]['Ly_beta_esc_fac']
                case_hi = Ly_beta_esc_fac_cases[key]
            if Ly_beta_esc_fac > Ly_beta_esc_fac_cases[key]['Ly_beta_esc_fac'] and Ly_beta_esc_fac_cases[key]['Ly_beta_esc_fac'] > tmp_esc_fac_lo:
                tmp_esc_fac_lo = Ly_beta_esc_fac_cases[key]['Ly_beta_esc_fac']
                case_lo = Ly_beta_esc_fac_cases[key]

        if case_hi and case_lo:
            # interpolate the PECs linearly
            if block_key in line_blocks_n5:
                # return 2D coeff(te, dens) in units ph s-1 cm3
                # print('Getting ADF15 data ', block_key, ' from read_adf15...')
                coeff_hi, info = read_adf15(file=case_hi['adf15_file'], block=line_blocks_n5[block_key], te=Te_arr, dens=ne_arr, all=True, return_info = True)
                coeff_lo, info = read_adf15(file=case_lo['adf15_file'], block=line_blocks_n5[block_key], te=Te_arr, dens=ne_arr, all=True, return_info = True)
                coeff = coeff_lo + (Ly_beta_esc_fac - case_lo['Ly_beta_esc_fac'])*(coeff_hi-coeff_lo)/(case_hi['Ly_beta_esc_fac'] - case_lo['Ly_beta_esc_fac'])
                # print('adf15 suppress file hi: ', case_hi['adf15_file'])
                # print('adf15 suppress file lo: ', case_lo['adf15_file'])
                # print('Done')

                return coeff, info
            else:
                print(block_key, ' entry does not exist')

        else:
            if case_hi:
                file = case_hi['adf15_file']
            elif case_lo:
                file = case_lo['adf15_file']
            else: # default
                file = '/home/bloman/python_tools/pyADASread/adas_data/pec12_h0_n5.dat'
    else:
        file = '/home/bloman/python_tools/pyADASread/adas_data/pec12_h0_n5.dat'

    # return 2D coeff(te, dens) in units ph s-1 cm3
    if block_key in line_blocks_n5:
        # print('Getting ADF15 data ', block_key, ' from read_adf15...')
        coeff, info = read_adf15(file=file, block=line_blocks_n5[block_key], te=Te_arr, dens=ne_arr, all=True, return_info = True)
        # print('Done')
    else:
        print(block_key, ' entry does not exist')

    return coeff, info

def get_imp_line_emiss(line_key, PEC_dict, Te_eV, ne_cm3, nparent_cm3, nion_cm3):
    # CALCULATE LINE EMISSIVITY FROM ADAS PECs#

    idxTe, valTe = find_nearest(PEC_dict[line_key+'recom'].Te_arr, Te_eV)
    idxne, valne = find_nearest(PEC_dict[line_key+'recom'].ne_arr, ne_cm3)

    # RECOMBINATION COMPONENT
    recom_emiss =  PEC_dict[line_key+'recom'].pec[idxTe, idxne] * ne_cm3 * nparent_cm3 * (1./(4.*np.pi)) # ph s-1 cm-3 sr-1
    recom_emiss *= 1.0e06 # ph s-1 m-3 sr-1

    # EXCITATION COMPONENT (assuming quasi-neutral plasma ne=ni)
    excit_emiss =  PEC_dict[line_key+'excit'].pec[idxTe, idxne] * ne_cm3 * nion_cm3 * (1./(4.*np.pi)) # ph s-1 cm-3 sr-1
    excit_emiss *= 1.0e06 # ph s-1 m-3 sr-1

    return excit_emiss, recom_emiss

def get_H_pop(line_key, PEC_dict, Te_eV, ne_cm3, ni_cm3, n0_cm3):
    # OUTPUT: EXCITED STATE POPULATION, (m^-3)

    idxTe, valTe = find_nearest(PEC_dict[line_key+'recom'].Te_arr, Te_eV)
    idxne, valne = find_nearest(PEC_dict[line_key+'recom'].ne_arr, ne_cm3)
    Arate = Arates_n5[line_key]

    # RECOMBINATION COMPONENT
    n_recom =  (PEC_dict[line_key+'recom'].pec[idxTe, idxne] * ne_cm3 * ni_cm3 ) / Arate # cm-3
    n_recom *= 1.0e06 # m-3

    # EXCITATION COMPONENT (assuming quasi-neutral plasma ne=ni)
    n_excit =  (PEC_dict[line_key+'excit'].pec[idxTe, idxne] * ne_cm3 * n0_cm3 ) / Arate # cm-3
    n_excit *= 1.0e06 # m-3

    return n_recom, n_excit

def get_H_line_emiss(line_key, PEC_dict, Te_eV, ne_cm3, ni_cm3, n0_cm3):
    # CALCULATE LINE EMISSIVITY FROM ADAS PECs#

    idxTe, valTe = find_nearest(PEC_dict[line_key+'recom'].Te_arr, Te_eV)
    idxne, valne = find_nearest(PEC_dict[line_key+'recom'].ne_arr, ne_cm3)

    # RECOMBINATION COMPONENT (assuming quasi-neutral plasma ne=ni)
    recom_emiss =  PEC_dict[line_key+'recom'].pec[idxTe, idxne] * ne_cm3 * ni_cm3 * (1./(4.*np.pi)) # ph s-1 cm-3 sr-1
    recom_emiss *= 1.0e06 # ph s-1 m-3 sr-1

    # EXCITATION COMPONENT
    excit_emiss =  PEC_dict[line_key+'excit'].pec[idxTe, idxne] * ne_cm3 * n0_cm3 * (1./(4.*np.pi)) # ph s-1 cm-3 sr-1
    excit_emiss *= 1.0e06 # ph s-1 m-3 sr-1

    return excit_emiss, recom_emiss

def get_H_line_intensities(line_dict, PEC_dict, ne_cm3, Te_eV, n0_cm3, delL_m, recom_only = False, line_ratio=False):

    # CALCULATE LINE INTENSITY FROM ADAS PECs#
    PECrec_emiss = [] # ph s-1 cm-3
    PECexc_emiss = []
    PEC_nupp_list = [] # upper excited state

    if line_ratio: delL_m = 1.0 # delL cancels so set to 1 in case passed value is 0

    for idx,key in enumerate(line_dict.keys()):
        idxTe, valTe = find_nearest(PEC_dict[key+'recom'].Te_arr, Te_eV)
        idxne, valne = find_nearest(PEC_dict[key+'recom'].ne_arr, ne_cm3)

        # RECOMBINATION COMPONENT (assuming quasi-neutral plasma ne=ni)
        recom_emiss =  PEC_dict[key+'recom'].pec[idxTe, idxne] * ne_cm3 * ne_cm3 # ph s-1 cm-3
        recom_emiss *= 1.0e06 # ph s-1 m-3
        PECrec_emiss.append(recom_emiss)
        PEC_nupp_list.append(line_dict[key][0])

        # EXCITATION COMPONENT (assuming quasi-neutral plasma ne=ni)
        excit_emiss =  PEC_dict[key+'excit'].pec[idxTe, idxne] * ne_cm3 * n0_cm3 # ph s-1 cm-3
        excit_emiss *= 1.0e06 # ph s-1 m-3
        PECexc_emiss.append(excit_emiss)

        from operator import add
        if recom_only == True:
            PECtot_emiss = np.asarray(PECrec_emiss)
        else:
            PECtot_emiss = np.asarray(PECexc_emiss) + np.asarray(PECrec_emiss)

    PEC_nupp = np.asarray(PEC_nupp_list)

    # Intensity output in units of ph s-1 m-2 sr-1
    PECrec_I = np.asarray(PECrec_emiss) * delL_m * (1./(4.*np.pi))
    PECexc_I = np.asarray(PECexc_emiss) * delL_m * (1./(4.*np.pi))
    PECtot_I = PECtot_emiss * delL_m * (1./(4.*np.pi))

    # return sorted arrays by nupp
    nupp_idx = np.argsort(PEC_nupp)
    # print PEC_nupp[nupp_idx]

    PECrec_I = PECrec_I[nupp_idx]
    PECexc_I = PECexc_I[nupp_idx]
    PECtot_I = PECtot_I[nupp_idx]

    if line_ratio:
        return PEC_nupp[nupp_idx], PECexc_I/PECexc_I[0], PECrec_I/PECrec_I[0], PECtot_I/PECtot_I[0]
    else:
        return PEC_nupp[nupp_idx], PECexc_I, PECrec_I, PECtot_I

if __name__ == "__main__":
    print('adas_adf15_read')
