# -*- coding: utf-8 -*-

import numpy as np
from adaslib import *
from adas4xx import *
from scipy.interpolate import interp1d, interp2d
import json, os


# adf11_files_dict = {
#     '1':
#         {'acd':'/home/adas/adas/adf11/acd12/acd12_h.dat',
#          'scd':'/home/adas/adas/adf11/scd12/scd12_h.dat',
#          'ccd':'/home/adas/adas/adf11/ccd12/ccd12_h.dat',
#          'plt':'/home/adas/adas/adf11/plt12/plt12_h.dat',
#          'prb':'/home/adas/adas/adf11/prb12/prb12_h.dat',
#          'prc':'/home/adas/adas/adf11/prc12/prc12_h.dat'},
#     '4':
#         {'acd': '/home/adas/adas/adf11/acd96/acd96_be.dat',
#          'scd': '/home/adas/adas/adf11/scd96/scd96_be.dat',
#          'ccd': '/home/adas/adas/adf11/ccd89/ccd89_be.dat',
#          'plt': '/home/adas/adas/adf11/plt96/plt96_be.dat',
#          'prb': '/home/adas/adas/adf11/prb96/prb96_be.dat',
#          'prc': '/home/adas/adas/adf11/prc89/prc89_be.dat'},
#     '6':
#         {'acd': '/home/adas/adas/adf11/acd96/acd96_c.dat',
#          'scd': '/home/adas/adas/adf11/scd96/scd96_c.dat',
#          'ccd': '/home/adas/adas/adf11/ccd89/ccd89_c.dat',
#          'plt': '/home/adas/adas/adf11/plt96/plt96_c.dat',
#          'prb': '/home/adas/adas/adf11/prb96/prb96_c.dat',
#          'prc': '/home/adas/adas/adf11/prc89/prc89_c.dat'},
#     '7':
#         {'acd': '/home/adas/adas/adf11/acd96/acd96_n.dat',
#          'scd': '/home/adas/adas/adf11/scd96/scd96_n.dat',
#          # 'ccd': '/home/adas/adas/adf11/ccd96/ccd96_n.dat',
#          'plt': '/home/adas/adas/adf11/plt96/plt96_n.dat',
#          'prb': '/home/adas/adas/adf11/prb96/prb96_n.dat',
#          'prc': '/home/adas/adas/adf11/prc89/prc89_n.dat'},
#     '10':
#         {'acd': '/home/adas/adas/adf11/acd96/acd96_ne.dat',
#          'scd': '/home/adas/adas/adf11/scd96/scd96_ne.dat',
#          # 'ccd': '/home/adas/adas/adf11/ccd96/ccd96_ne.dat',
#          'plt': '/home/adas/adas/adf11/plt96/plt96_ne.dat',
#          'prb': '/home/adas/adas/adf11/prb96/prb96_ne.dat',
#          'prc': '/home/adas/adas/adf11/prc89/prc89_ne.dat'},
#     '74':
#         {'acd': '/home/adas/adas/adf11/acd50/acd50_w.dat',
#          'scd': '/home/adas/adas/adf11/scd50/scd50_w.dat',
#          'plt': '/home/adas/adas/adf11/plt50/plt50_w.dat',
#          'prb': '/home/adas/adas/adf11/prb50/prb50_w.dat'}
# }

Ly_beta_esc_fac_cases = {
    '0':{'Ly_beta_esc_fac':1.00, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.100.dat',
                                 'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.100.dat'},
    '1':{'Ly_beta_esc_fac':0.80, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.80.dat',
                                 'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.80.dat'},
    '2':{'Ly_beta_esc_fac':0.70, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.70.dat',
                                 'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.70.dat'},
    '3':{'Ly_beta_esc_fac':0.65, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.65.dat',
                                 'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.65.dat'},
    '4':{'Ly_beta_esc_fac':0.60, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.60.dat',
                                 'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.60.dat'},
    '5':{'Ly_beta_esc_fac':0.55, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.55.dat',
                                 'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.55.dat'},
    '6':{'Ly_beta_esc_fac':0.50, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.50.dat',
                                 'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.50.dat'},
    '7':{'Ly_beta_esc_fac':0.45, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.45.dat',
                                 'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.45.dat'},
    '8':{'Ly_beta_esc_fac':0.40, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.40.dat',
                                 'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.40.dat'},
    '9':{'Ly_beta_esc_fac':0.35, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.35.dat',
                                 'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.35.dat'},
    '10':{'Ly_beta_esc_fac':0.30, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.30.dat',
                                  'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.30.dat'},
    '11':{'Ly_beta_esc_fac':0.25, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.25.dat',
                                  'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.25.dat'},
    '12':{'Ly_beta_esc_fac':0.20, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.20.dat',
                                  'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.20.dat'},
    '13':{'Ly_beta_esc_fac':0.18, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.18.dat',
                                  'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.18.dat'},
    '14':{'Ly_beta_esc_fac':0.16, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.16.dat',
                                  'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.16.dat'},
    '15':{'Ly_beta_esc_fac':0.14, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.14.dat',
                                  'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.14.dat'},
    '16':{'Ly_beta_esc_fac':0.12, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.12.dat',
                                  'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.12.dat'},
    '17':{'Ly_beta_esc_fac':0.10, 'acd_file':'/home/bloman/python_tools/pyADASread/adas_data/acd16_h_Ly_beta_esc_fac.10.dat',
                                  'scd_file':'/home/bloman/python_tools/pyADASread/adas_data/scd16_h_Ly_beta_esc_fac.10.dat'},
}

def get_adf11_input_dict(file):
    # Read adf11 input dict
    try:
        with open(file, mode='r', encoding='utf-8') as f:
            # Remove comments
            with open("temp.json", 'w') as wf:
                for line in f.readlines():
                    if line[0:2] == '//' or line[0:1] == '#':
                        continue
                    wf.write(line)
        with open("temp.json", 'r') as f:
            adf11_files_dict = json.load(f)
        os.remove('temp.json')
        return adf11_files_dict
    except IOError as e:
                raise

class ADF11:
    def __init__(self, acd, scd, ccd, Te_arr, ne_arr, plt=None, prb=None, prc=None):
        self.acd = acd # 2D ACD array as fn(Te,ne) units: cm3 s-1
        self.scd = scd # 2D SCD array as fn(Te,ne) units: cm3 s-1
        self.ccd = ccd # 2D CCD array as fn(Te,ne) units: cm3 s-1
        self.plt = plt # 2D PLT array as fn(Te,ne) for all ion stages, units: W cm3
        self.prb = prb # 2D PRB array as fn(Te,ne) for all ion stages, units: W cm3
        self.prc = prc # 2D PRC array as fn(Te,ne) for all ion stages, units: W cm3
        self.Te_arr = Te_arr  # copy of Te array
        self.ne_arr = ne_arr # copy of ne array

class ADF11_imp:
    def __init__(self, Te_arr, ne_arr, acd, scd, ccd, plt, prb, prc, ion_bal_frac, ion_bal_pwr):
        self.Te_arr = Te_arr  # copy of Te array
        self.ne_arr = ne_arr # copy of ne array
        self.acd = acd # 2D ACD array as fn(Te,ne) units: cm3 s-1
        self.scd = scd # 2D SCD array as fn(Te,ne) units: cm3 s-1
        self.ccd = ccd # 2D CCD array as fn(Te,ne) units: cm3 s-1
        self.plt = plt # 2D PLT array as fn(Te,ne) for all ion stages, units: W cm3
        self.prb = prb # 2D PRB array as fn(Te,ne) for all ion stages, units: W cm3
        self.prc = prc # 2D PRC array as fn(Te,ne) for all ion stages, units: W cm3
        # ADAS405 ionisation balance
        self.ion_bal_frac = ion_bal_frac # dict
        self.ion_bal_pwr = ion_bal_pwr # dict

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def get_adas_imp_adf11(adf11_files_dict, at_num, Te_arr, ne_arr, tint=None):

    # tint: integration time (s)
    
    at_num_to_elem = {'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'Ne':10, 'Ar':18, 'Kr':36, 'W':74}
    for elem in at_num_to_elem:
        if at_num_to_elem[elem] == at_num:
            print('Getting ADF11 data from run_adas405 and read_adf11')
            # IONISATION BALANCE
            if at_num == 1:
                year=12
            elif at_num == 74:
                year=50
            else:
                year = 96
                
            if tint:
                ion_bal_frac, ion_bal_pwr = run_adas406(elem=elem, uid='adas', year=96, dens=ne_arr, te=Te_arr, tint=tint, all=True)
            else:
                ion_bal_frac, ion_bal_pwr = run_adas405(elem=elem, uid='adas', year=96, dens=ne_arr, te=Te_arr, all=True)

            # INDIVIDUAL COEFF FOR EACH ION STAGE
            acd = np.zeros((len(Te_arr), len(ne_arr), at_num))
            scd = np.zeros((len(Te_arr), len(ne_arr), at_num))
            ccd = np.zeros((len(Te_arr), len(ne_arr), at_num))
            plt = np.zeros((len(Te_arr), len(ne_arr), at_num))
            prb = np.zeros((len(Te_arr), len(ne_arr), at_num))
            prc = np.zeros((len(Te_arr), len(ne_arr), at_num))

            items = ['acd', 'scd', 'ccd', 'plt', 'prb', 'prc']
            for coeff in items:
                if coeff in adf11_files_dict[str(at_num)]:
                    if coeff == 'acd':
                        print('acd file: ', adf11_files_dict[str(at_num)][coeff])
                        for ion_stage in range(at_num):
                            acd[:, :, ion_stage] = read_adf11(file=adf11_files_dict[str(at_num)][coeff],
                                                              adf11type=coeff, is1=ion_stage + 1, te=Te_arr,
                                                              dens=ne_arr, all=True)
                    elif coeff == 'scd':
                        print('scd file: ', adf11_files_dict[str(at_num)][coeff])
                        for ion_stage in range(at_num):
                            scd[:, :, ion_stage] = read_adf11(file=adf11_files_dict[str(at_num)][coeff],
                                                              adf11type=coeff, is1=ion_stage + 1, te=Te_arr,
                                                              dens=ne_arr, all=True)
                    elif coeff == 'ccd':
                        print('ccd file: ', adf11_files_dict[str(at_num)][coeff])
                        for ion_stage in range(at_num):
                            ccd[:, :, ion_stage] = read_adf11(file=adf11_files_dict[str(at_num)][coeff],
                                                              adf11type=coeff, is1=ion_stage + 1, te=Te_arr,
                                                              dens=ne_arr, all=True)
                    elif coeff == 'plt':
                        print('plt file: ', adf11_files_dict[str(at_num)][coeff])
                        for ion_stage in range(at_num):
                            plt[:, :, ion_stage] = read_adf11(file=adf11_files_dict[str(at_num)][coeff],
                                                              adf11type=coeff, is1=ion_stage + 1, te=Te_arr,
                                                              dens=ne_arr, all=True)
                    elif coeff == 'prb':
                        print('prb file: ', adf11_files_dict[str(at_num)][coeff])
                        for ion_stage in range(at_num):
                            prb[:, :, ion_stage] = read_adf11(file=adf11_files_dict[str(at_num)][coeff],
                                                              adf11type=coeff, is1=ion_stage + 1, te=Te_arr,
                                                              dens=ne_arr, all=True)
                    elif coeff == 'prc':
                        print('prc file: ', adf11_files_dict[str(at_num)][coeff])
                        for ion_stage in range(at_num):
                            prc[:, :, ion_stage] = read_adf11(file=adf11_files_dict[str(at_num)][coeff],
                                                              adf11type=coeff, is1=ion_stage + 1, te=Te_arr,
                                                              dens=ne_arr, all=True)

            # plt_file = '/home/adas/adas/adf11/plt96/plt96_' + elem.lower() + '.dat'
            # prb_file = '/home/adas/adas/adf11/prb96/prb96_' + elem.lower() + '.dat'
            # prc_file = '/home/adas/adas/adf11/prc89/prc89_' + elem.lower() + '.dat'

            # for ion_stage in range(at_num):
            #     print('at_num, stage:', at_num, ion_stage)
            #     plt[:,:,ion_stage] = read_adf11(file=plt_file, adf11type='plt', is1=ion_stage+1, te=Te_arr, dens=ne_arr, all=True)
            #     prb[:,:,ion_stage] = read_adf11(file=prb_file, adf11type='prb', is1=ion_stage+1, te=Te_arr, dens=ne_arr, all=True)
            #     prc[:,:,ion_stage] = read_adf11(file=prc_file, adf11type='prc', is1=ion_stage+1, te=Te_arr, dens=ne_arr, all=True)

            adf11 = ADF11_imp(Te_arr, ne_arr, acd, scd, ccd, plt, prb, prc, ion_bal_frac, ion_bal_pwr)

            print('Done')
            return adf11

def get_adas_H_adf11(Te_arr, ne_arr, pwr=False, year=12, custom_dir=None):

    if custom_dir:
        location = custom_dir
    else:
        location = '/home/adas/adas/adf11'

    print('Getting ADF11 data from read_adf11 year', str(year), ' (ccd prc year 96)...')
    # EFFECTIVE RECOMBINATION COEFFICIENT (cm3 s-1)
    file = location + '/acd'+str(year)+'/acd'+str(year)+'_h.dat'
    acd = read_adf11(file=file, adf11type='acd', is1=1, te=Te_arr, dens=ne_arr, all=True)
    # EFFECTIVE IONISATION COEFFICIENT (cm3 s-1)
    file = location + '/scd'+str(year)+'/scd'+str(year)+'_h.dat'
    scd = read_adf11(file=file, adf11type='scd', is1=1, te=Te_arr, dens=ne_arr, all=True)

    # ccd only exists fo year 96
    file = location + '/ccd96/ccd96_h.dat'
    ccd = read_adf11(file=file, adf11type='ccd', is1=1, te=Te_arr, dens=ne_arr, all=True)

    # Also get rad pwr los coeff
    if pwr:
        plt_file =  location + '/plt'+str(year)+'/plt'+str(year)+'_h.dat'
        prb_file =  location + '/prb'+str(year)+'/prb'+str(year)+'_h.dat'
        prc_file =  location + '/prc96/prc96_h.dat'
        plt = read_adf11(file=plt_file, adf11type='plt', is1=1, te=Te_arr, dens=ne_arr, all=True)
        prb = read_adf11(file=prb_file, adf11type='prb', is1=1, te=Te_arr, dens=ne_arr, all=True)
        prc = read_adf11(file=prb_file, adf11type='prc', is1=1, te=Te_arr, dens=ne_arr, all=True)
        adf11 = ADF11(acd, scd, ccd, Te_arr, ne_arr, plt=plt, prb=prb, prc=prc)
    else:
        adf11 = ADF11(acd, scd, ccd, Te_arr, ne_arr)
    print('Done')

    return adf11

def get_adas_H_adf11_interp(Te_rnge, ne_rnge, npts=100, npts_interp=1000, pwr=False, year=12, custom_dir=None):

    if custom_dir:
        location = custom_dir
    else:
        location = '/home/adas/adas/adf11'

    # NOTE: interp2d only seems to work when x and y have equal num of points
    Te_arr = np.logspace(np.log10(Te_rnge[0]), np.log10(Te_rnge[1]), npts)
    ne_arr = np.logspace(np.log10(ne_rnge[0]), np.log10(ne_rnge[1]), npts)

    print('Getting ADF11 data from read_adf11 year', str(year), ' (ccd prc year 96)...')

    # EFFECTIVE RECOMBINATION COEFFICIENT (cm3 s-1)
    file = location + '/acd'+str(year)+'/acd'+str(year)+'_h.dat'
    acd = read_adf11(file=file, adf11type='acd', is1=1, te=Te_arr, dens=ne_arr, all=True)
    # EFFECTIVE IONISATION COEFFICIENT (cm3 s-1)
    file = location + '/scd'+str(year)+'/scd'+str(year)+'_h.dat'
    scd = read_adf11(file=file, adf11type='scd', is1=1, te=Te_arr, dens=ne_arr, all=True)

    # ccd only exists fo year 96
    file = location + '/ccd96/ccd96_h.dat'
    ccd = read_adf11(file=file, adf11type='ccd', is1=1, te=Te_arr, dens=ne_arr, all=True)

    # linearly interpolate results on finer Te ne grid
    f_acd = interp2d(Te_arr, ne_arr, acd, kind='linear')
    f_scd = interp2d(Te_arr, ne_arr, scd, kind='linear')
    f_ccd = interp2d(Te_arr, ne_arr, ccd, kind='linear')
    Te_fine = np.logspace(np.log10(Te_arr[0]), np.log10(Te_arr[-1]), npts_interp)
    ne_fine = np.logspace(np.log10(ne_arr[0]), np.log10(ne_arr[-1]), npts_interp)
    acd_interp = f_acd(Te_fine, ne_fine)
    scd_interp = f_scd(Te_fine, ne_fine)
    ccd_interp = f_ccd(Te_fine, ne_fine)

    # Also get rad pwr los coeff
    if pwr:
        plt_file =  location + '/plt'+str(year)+'/plt'+str(year)+'_h.dat'
        prb_file =  location + '/prb'+str(year)+'/prb'+str(year)+'_h.dat'
        prc_file =  location + '/prc96/prc96_h.dat'
        plt = read_adf11(file=plt_file, adf11type='plt', is1=1, te=Te_arr, dens=ne_arr, all=True)
        prb = read_adf11(file=prb_file, adf11type='prb', is1=1, te=Te_arr, dens=ne_arr, all=True)
        prc = read_adf11(file=prc_file, adf11type='prc', is1=1, te=Te_arr, dens=ne_arr, all=True)

        # linearly interpolate results on finer Te ne grid
        f_plt = interp2d(Te_arr, ne_arr, plt, kind='linear')
        f_prb = interp2d(Te_arr, ne_arr, prb, kind='linear')
        f_prc = interp2d(Te_arr, ne_arr, prc, kind='linear')
        plt_interp = f_plt(Te_fine, ne_fine)
        prb_interp = f_prb(Te_fine, ne_fine)
        prc_interp = f_prc(Te_fine, ne_fine)

        adf11 = ADF11(acd_interp, scd_interp, ccd_interp, Te_fine, ne_fine, plt=plt_interp, prb=prb_interp, prc=prc_interp)
    else:
        adf11 = ADF11(acd_interp, scd_interp, ccd_interp, Te_fine, ne_fine)

    return adf11

def get_adas_H_adf11_suppressed(Te_arr, ne_arr, Ly_beta_esc_fac):

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
            # interpolate the coeffs linearly
            # print('Getting ADF11 ACD,SCD data from read_adf11...')
            acd_hi = read_adf11(file=case_hi['acd_file'], adf11type='acd', is1=1, te=Te_arr, dens=ne_arr, all=True)
            acd_lo = read_adf11(file=case_lo['acd_file'], adf11type='acd', is1=1, te=Te_arr, dens=ne_arr, all=True)
            acd = acd_lo + (Ly_beta_esc_fac - case_lo['Ly_beta_esc_fac'])*(acd_hi-acd_lo)/(case_hi['Ly_beta_esc_fac'] - case_lo['Ly_beta_esc_fac'])
            scd_hi = read_adf11(file=case_hi['scd_file'], adf11type='scd', is1=1, te=Te_arr, dens=ne_arr, all=True)
            scd_lo = read_adf11(file=case_lo['scd_file'], adf11type='scd', is1=1, te=Te_arr, dens=ne_arr, all=True)
            scd = scd_lo + (Ly_beta_esc_fac - case_lo['Ly_beta_esc_fac'])*(scd_hi-scd_lo)/(case_hi['Ly_beta_esc_fac'] - case_lo['Ly_beta_esc_fac'])
            # print('adf11 scd suppress file hi: ', case_hi['scd_file'])
            # print('adf11 scd suppress file lo: ', case_lo['scd_file'])
            # print('Done')

            adf11 = ADF11(acd, scd, None, Te_arr, ne_arr)
    
            return adf11

        else:
            if case_hi:
                acd_file = case_hi['acd_file']
                scd_file = case_hi['scd_file']
            elif case_lo:
                acd_file = case_lo['acd_file']
                scd_file = case_lo['scd_file']
            else: # default
                acd_file = '/home/adas/adas/adf11/acd12/acd12_h.dat'
                scd_file = '/home/adas/adas/adf11/scd12/scd12_h.dat'
    else:
        acd_file = Ly_beta_esc_fac_cases['0']['acd_file']
        scd_file = Ly_beta_esc_fac_cases['0']['scd_file']
        # acd_file = '/home/adas/adas/adf11/acd12/acd12_h.dat'
        # scd_file = '/home/adas/adas/adf11/scd12/scd12_h.dat'

    # return 2D coeff(te, dens) in units ph s-1 cm3
    # print('Getting ADF11 ACD,SCD data from read_adf11...')
    acd = read_adf11(file=acd_file, adf11type='acd', is1=1, te=Te_arr, dens=ne_arr, all=True)
    scd = read_adf11(file=scd_file, adf11type='scd', is1=1, te=Te_arr, dens=ne_arr, all=True)
    adf11 = ADF11(acd, scd, None, Te_arr, ne_arr)
    # print('Done')

    return adf11

if __name__ == "__main__":

    print('adas_adf11_read')