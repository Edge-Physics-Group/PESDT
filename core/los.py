import numpy as np
from pyADASread import adas_adf11_read, adas_adf15_read, continuo_read
import logging
logger = logging.getLogger(__name__)

class LOS:

    def __init__(self, diag, los_poly = None, chord_num=None, p1 = None, w1 = None, p2orig = None, p2 = None,
                 w2orig = None, w2 = None, l12 = None, theta=None, los_angle=None, spec_line_dict=None,
                 spec_line_dict_lytrap=None, data_source = "AMJUEL"):

        self.diag = diag

        self.los_poly = los_poly
        self.chord_num = chord_num
        self.p1 = p1
        self.w1 = w1
        self.p2unmod = p2orig
        self.p2 = p2
        self.w2unmod = w2orig
        self.w2 = w2
        self.l12 = l12
        self.theta = theta
        self.los_angle = los_angle
        self.cells = []
        self.spec_line_dict = spec_line_dict
        self.spec_line_dict_lytrap = spec_line_dict_lytrap
        self.data_source = data_source

        # center line shapely object
        self.shply_cenline = None
        # intersection with separatrix below x-point (for identifying PFR-SOL regions)
        self.shply_intersects_w_sep = None

        # LINE INTEGRATED QUANTITIES - DICT
        # {'param':val, 'units':''}
        self.los_int = {}

        # 1D QUANTITIES ALONG LOS - DICT
        # {'param':val, 'units':''}
        self.los_1d = {}

        # SYNTHETIC SPECTRA - DICT
        # e.g. features: 'stark', 'ff_fb_continuum'
        # {'feature':{'wave':1d_arr, 'intensity':1d_arr, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}}
        self.los_int_spectra = {}
        # {'feature':{'wave':2d_arr[dl_idx,arr], 'intensity':2d_arr[dl_idx,arr], 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}}
        self.los_1d_spectra = {}
        # Stark broadening coefficients (Lomanowski et al 2015, NF)
        self.lorentz_5_2_coeffs = {}
        if self.spec_line_dict['1']['1']:
            for key in self.spec_line_dict['1']['1']:
                # H 6-2
                if self.spec_line_dict['1']['1'][key][1] == '2' and self.spec_line_dict['1']['1'][key][0] == '6':
                    self.lorentz_5_2_coeffs[key] = {'C':3.954E-16, 'a':0.7149, 'b':0.028}


    def orthogonal_polys(self):
        for cell in self.cells:
            # FOR EACH CELL IN LOS, COMPUTE EQUIVALENT ORTHOGONAL POLY

            cell_area = cell.poly.area
            cell.dist_to_los_v1 = np.sqrt(((self.p1[0] - cell.R) ** 2) + ((self.p1[1] - cell.Z) ** 2))


            # Handle case of uniform (or close to uniform) LOS beam width (e.g., KG1V)
            if np.abs(self.w2-self.w1) <= 1.0e-03:
                cell.los_ortho_width = self.w2
            # Similar triangles method
            else:
                dw = (cell.dist_to_los_v1 / self.l12 ) * ((self.w2 - self.w1) / 2.0)
                cell.los_ortho_width = self.w1 + 2.0 * dw

            cell.los_ortho_delL = cell_area / cell.los_ortho_width

    def calc_int_and_1d_los_quantities_1(self):

        # SUM H EMISSION METHOD 1: TRAVERSE CELLS AND SUM EMISSIVITIES*ORTHO_DELL
        self.los_int['H_emiss'] = {}
        for key in self.spec_line_dict['1']['1']:
            sum_excit = 0
            sum_recom = 0
            for cell in self.cells:
                sum_excit += (cell.H_emiss[key]['excit'] * cell.los_ortho_delL)
                sum_recom += (cell.H_emiss[key]['recom'] * cell.los_ortho_delL)
            self.los_int['H_emiss'].update({key:{'excit':sum_excit, 'recom':sum_recom, 'units':'ph.s^-1.m^-2.sr^-1'}})

        # Same for Ly-opacity
        if self.spec_line_dict_lytrap:
            for key in self.spec_line_dict_lytrap['1']['1']:
                sum_excit = 0
                sum_recom = 0
                for cell in self.cells:
                    sum_excit += (cell.H_emiss[key]['excit'] * cell.los_ortho_delL)
                    sum_recom += (cell.H_emiss[key]['recom'] * cell.los_ortho_delL)
                self.los_int['H_emiss'].update({key:{'excit':sum_excit, 'recom':sum_recom, 'units':'ph.s^-1.m^-2.sr^-1'}})

        # SUM TOTAL RECOMBINATION/IONISATION
        self.los_int['Srec'] = {}
        self.los_int['Sion'] = {}
        sum_Srec = 0
        sum_Sion = 0
        for cell in self.cells:
            sum_Srec += (cell.Srec * cell.poly.area * 2.*np.pi*cell.R)
            sum_Sion += (cell.Sion * cell.poly.area * 2.*np.pi*cell.R)
        self.los_int['Srec'].update({'val':sum_Srec, 'units':'s^-1'})
        self.los_int['Sion'].update({'val':sum_Sion, 'units':'s^-1'})

        # SUM RADIATED POWER (total and per m^3)
        self.los_int['Prad'] = {}
        self.los_int['Prad_perm2'] = {}

        if self.spec_line_dict_lytrap:
            self.los_int['Prad_Lytrap'] = {}
            self.los_int['Prad_Lytrap_perm2'] = {}
            sum_Prad_H_Lytrap = 0
            sum_Prad_H_Lytrap_perm2 = 0

        sum_Prad_H = 0
        sum_Prad_H_perm2 = 0
        sum_Prad_ff = 0
        sum_Prad_ff_perm2 = 0
        for cell in self.cells:
            sum_Prad_H += (cell.H_radpwr)
            sum_Prad_H_perm2 += (cell.H_radpwr_perm3*cell.los_ortho_delL)

            if self.spec_line_dict_lytrap:
                sum_Prad_H_Lytrap += (cell.H_radpwr_Lytrap)
                sum_Prad_H_Lytrap_perm2 += (cell.H_radpwr_Lytrap_perm3 * cell.los_ortho_delL)

            sum_Prad_ff += (cell.ff_radpwr)
            sum_Prad_ff_perm2 += (cell.ff_radpwr_perm3*cell.los_ortho_delL)
        self.los_int['Prad'].update({'H':sum_Prad_H,  'units':'W'})
        self.los_int['Prad_perm2'].update({'H':sum_Prad_H_perm2,  'units':'W m^-2'})
        self.los_int['Prad'].update({'ff':sum_Prad_ff,  'units':'W'})
        self.los_int['Prad_perm2'].update({'ff':sum_Prad_ff_perm2,  'units':'W m^-2'})

        if self.spec_line_dict_lytrap:
            self.los_int['Prad_Lytrap'].update({'H': sum_Prad_H_Lytrap, 'units': 'W'})
            self.los_int['Prad_Lytrap_perm2'].update({'H': sum_Prad_H_Lytrap_perm2, 'units': 'W m^-2'})

    def calc_int_and_1d_los_quantities_2(self):
        ####################################################
        # COMPUTE AVERAGED QUANTITIES ALONG LOS
        ####################################################
        self.los_1d['dl'] = 0.01 # m
        self.los_1d['l'] = np.arange(0, self.l12, self.los_1d['dl'])
        for item in ['ne', 'n0', 'te', 'ortho_delL']:
            self.los_1d[item] = np.zeros((len(self.los_1d['l'])))

        # H EMISS LOS 1D METHOD 2: SIMILAR TO METHOD 1 ABOVE, EXCEPT 1D PROFILE INFO IS ALSO STORED ALONG LOS
        recom = {}
        excit = {}
        recom_pervol = {} # emissivity per m^-3
        excit_pervol = {} # emissivity per m^-3

        for key in self.spec_line_dict['1']['1']:
            recom[key] = np.zeros((len(self.los_1d['l'])))
            excit[key] = np.zeros((len(self.los_1d['l'])))
            recom_pervol[key] = np.zeros((len(self.los_1d['l'])))
            excit_pervol[key] = np.zeros((len(self.los_1d['l'])))

        

        for dl_idx, dl_val in enumerate(self.los_1d['l']):
            # temp storage
            ne_tmp = []
            n0_tmp = []
            te_tmp = []
            delL_tmp = []
            H_emiss_excit_tmp = {}
            H_emiss_recom_tmp = {}
            H_emiss_pervol_excit_tmp = {}
            H_emiss_pervol_recom_tmp = {}
            for key in self.spec_line_dict['1']['1']:
                H_emiss_excit_tmp[key] = []
                H_emiss_recom_tmp[key] = []
                H_emiss_pervol_excit_tmp[key] = []
                H_emiss_pervol_recom_tmp[key] = []
                            
            for cell in self.cells:
                if cell.dist_to_los_v1 >= dl_val and cell.dist_to_los_v1 < dl_val + self.los_1d['dl']:
                    ne_tmp.append(cell.ne)
                    n0_tmp.append(cell.n0)
                    te_tmp.append(cell.te)
                    delL_tmp.append(cell.los_ortho_delL)
                    for key in self.spec_line_dict['1']['1']:
                        H_emiss_excit_tmp[key].append(cell.H_emiss[key]['excit']*cell.los_ortho_delL)
                        H_emiss_recom_tmp[key].append(cell.H_emiss[key]['recom']*cell.los_ortho_delL)
                        H_emiss_pervol_excit_tmp[key].append(cell.H_emiss[key]['excit'])
                        H_emiss_pervol_recom_tmp[key].append(cell.H_emiss[key]['recom'])
            if ne_tmp:
                self.los_1d['ortho_delL'][dl_idx] = np.sum(np.asarray(delL_tmp))
                delL_norm = np.asarray(delL_tmp) / self.los_1d['ortho_delL'][dl_idx]
                self.los_1d['ne'][dl_idx] = np.average(np.asarray(ne_tmp),weights = delL_norm)
                self.los_1d['n0'][dl_idx] = np.average(np.asarray(n0_tmp),weights = delL_norm)
                self.los_1d['te'][dl_idx] = np.average(np.asarray(te_tmp),weights = delL_norm)
                for key in self.spec_line_dict['1']['1']:
                    excit[key][dl_idx] = np.sum(np.asarray(H_emiss_excit_tmp[key]))
                    recom[key][dl_idx] = np.sum(np.asarray(H_emiss_recom_tmp[key]))
                    excit_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_excit_tmp[key]), weights = delL_norm)
                    recom_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_recom_tmp[key]), weights = delL_norm )
                
        # REMOVE ZEROS FROM COMPUTED LOS ARRAYS
        nonzero_idx = np.nonzero(self.los_1d['ne'])
        for item in ['l', 'ne', 'n0', 'te', 'ortho_delL']:
            self.los_1d[item] = self.los_1d[item][nonzero_idx]
            # convert numpy arrays to lists for JSON serialization
            self.los_1d[item] = list(self.los_1d[item])

        self.los_1d['H_emiss'] = {}
        self.los_1d['H_emiss_per_vol'] = {}
        for key in self.spec_line_dict['1']['1']:
            excit[key] = excit[key][nonzero_idx]
            recom[key] = recom[key][nonzero_idx]
            excit_pervol[key] = excit_pervol[key][nonzero_idx]
            recom_pervol[key] = recom_pervol[key][nonzero_idx]
            self.los_1d['H_emiss'].update({key:{'excit':excit[key], 'recom':recom[key], 'units':'ph s^-1 m^-2 sr^-1'}})
            self.los_1d['H_emiss_per_vol'].update({key:{'excit':excit_pervol[key], 'recom':recom_pervol[key], 'units':'ph s^-1 m^-3 sr^-1'}})
            # convert numpy arrays to lists for JSON serialization
            self.los_1d['H_emiss'][key]['excit'] = list(self.los_1d['H_emiss'][key]['excit'])
            self.los_1d['H_emiss'][key]['recom'] = list(self.los_1d['H_emiss'][key]['recom'])
            self.los_1d['H_emiss_per_vol'][key]['excit'] = list(self.los_1d['H_emiss_per_vol'][key]['excit'])
            self.los_1d['H_emiss_per_vol'][key]['recom'] = list(self.los_1d['H_emiss_per_vol'][key]['recom'])

    def calc_int_and_1d_los_quantities_AMJUEL_1(self):
        # SUM H EMISSION METHOD 1: TRAVERSE CELLS AND SUM EMISSIVITIES*ORTHO_DELL
        self.los_int['H_emiss'] = {}
        for key in self.spec_line_dict['1']['1']:
            sum_exc = 0
            sum_rec = 0
            sum_h2 = 0
            sum_h2_pos = 0
            sum_h_neg = 0
            sum_tot = 0
            for cell in self.cells:
                sum_tot += (cell.H_emiss[key]['tot'] * cell.los_ortho_delL)
                sum_exc += (cell.H_emiss[key]['excit'] * cell.los_ortho_delL)
                sum_rec += (cell.H_emiss[key]['recom'] * cell.los_ortho_delL)
                sum_h2  += (cell.H_emiss[key]['h2'] * cell.los_ortho_delL)
                sum_h2_pos += (cell.H_emiss[key]['h2+'] * cell.los_ortho_delL)
                sum_h_neg += (cell.H_emiss[key]['h-'] * cell.los_ortho_delL)
            self.los_int['H_emiss'].update({key:{'tot':sum_tot, 'excit': sum_exc, 'recom': sum_rec,
                                            'h2': sum_h2, 'h2+': sum_h2_pos, 'h-': sum_h_neg, 'units':'ph.s^-1.m^-2.sr^-1'}})

    def calc_int_and_1d_los_quantities_AMJUEL_2(self):
        ####################################################
        # COMPUTE AVERAGED QUANTITIES ALONG LOS
        ####################################################
        self.los_1d['dl'] = 0.01 # m
        self.los_1d['l'] = np.arange(0, self.l12, self.los_1d['dl'])
        for item in ['ne', 'n0', 'te', 'ortho_delL']:
            self.los_1d[item] = np.zeros((len(self.los_1d['l'])))
        # H EMISS LOS 1D METHOD 2: SIMILAR TO METHOD 1 ABOVE, EXCEPT 1D PROFILE INFO IS ALSO STORED ALONG LOS
        # Added for processing emission calculated from AMJUEL, initially only for the total emissivity
        em = {}
        em_pervol = {}
        recom = {}
        excit = {}
        h2 = {}
        h2_pos = {}
        h_neg = {}
        h2_pervol = {}
        h2_pos_pervol = {}
        h_neg_pervol = {}
        recom_pervol = {} # emissivity per m^-3
        excit_pervol = {} # emissivity per m^-3

        for key in self.spec_line_dict['1']['1']:
            em[key] = np.zeros((len(self.los_1d['l'])))
            em_pervol[key] = np.zeros((len(self.los_1d['l'])))
            recom[key] = np.zeros((len(self.los_1d['l'])))
            excit[key] = np.zeros((len(self.los_1d['l'])))
            h2[key] = np.zeros((len(self.los_1d['l'])))
            h2_pos[key] = np.zeros((len(self.los_1d['l'])))
            h_neg[key] = np.zeros((len(self.los_1d['l'])))
            recom_pervol[key] = np.zeros((len(self.los_1d['l'])))
            excit_pervol[key] = np.zeros((len(self.los_1d['l'])))
            h2_pervol[key] = np.zeros((len(self.los_1d['l'])))
            h2_pos_pervol[key] = np.zeros((len(self.los_1d['l'])))
            h_neg_pervol[key] = np.zeros((len(self.los_1d['l'])))

        for dl_idx, dl_val in enumerate(self.los_1d['l']):
            # temp storage
            ne_tmp = []
            n0_tmp = []
            te_tmp = []
            delL_tmp = []
            H_emiss_em_tmp = {}
            H_emiss_em_pervol_tmp = {}
            H_emiss_excit_tmp = {}
            H_emiss_recom_tmp = {}
            H_emiss_h2_tmp = {}
            H_emiss_h2_pos_tmp = {}
            H_emiss_h_neg_tmp = {}
            H_emiss_pervol_excit_tmp = {}
            H_emiss_pervol_recom_tmp = {}
            H_emiss_pervol_h2_tmp = {}
            H_emiss_pervol_h2_pos_tmp = {}
            H_emiss_pervol_h_neg_tmp = {}
                
            for key in self.spec_line_dict['1']['1']:
                H_emiss_em_tmp[key] = []
                H_emiss_em_pervol_tmp[key] = []
                H_emiss_excit_tmp[key] = []
                H_emiss_recom_tmp[key] = []
                H_emiss_h2_tmp[key] = []
                H_emiss_h2_pos_tmp[key] = []
                H_emiss_h_neg_tmp[key] = []
                H_emiss_pervol_excit_tmp[key] = []
                H_emiss_pervol_recom_tmp[key] = []
                H_emiss_pervol_h2_tmp[key] = []
                H_emiss_pervol_h2_pos_tmp[key] = []
                H_emiss_pervol_h_neg_tmp[key] = []
                            
            for cell in self.cells:
                if cell.dist_to_los_v1 >= dl_val and cell.dist_to_los_v1 < dl_val + self.los_1d['dl']:
                    ne_tmp.append(cell.ne)
                    n0_tmp.append(cell.n0)
                    te_tmp.append(cell.te)
                    delL_tmp.append(cell.los_ortho_delL)
                    for key in self.spec_line_dict['1']['1']:
                        H_emiss_em_tmp[key].append(cell.H_emiss[key]['tot']*cell.los_ortho_delL)
                        H_emiss_em_pervol_tmp[key].append(cell.H_emiss[key]['tot'])
                        H_emiss_excit_tmp[key].append(cell.H_emiss[key]['excit']*cell.los_ortho_delL)
                        H_emiss_recom_tmp[key].append(cell.H_emiss[key]['recom']*cell.los_ortho_delL)
                        H_emiss_h2_tmp[key].append(cell.H_emiss[key]['h2']*cell.los_ortho_delL)
                        H_emiss_h2_pos_tmp[key].append(cell.H_emiss[key]['h2+']*cell.los_ortho_delL)
                        H_emiss_h_neg_tmp[key].append(cell.H_emiss[key]['h-']*cell.los_ortho_delL)
                        H_emiss_pervol_excit_tmp[key].append(cell.H_emiss[key]['excit'])
                        H_emiss_pervol_recom_tmp[key].append(cell.H_emiss[key]['recom'])
                        H_emiss_pervol_h2_tmp[key].append(cell.H_emiss[key]['h2'])
                        H_emiss_pervol_h2_pos_tmp[key].append(cell.H_emiss[key]['h2+'])
                        H_emiss_pervol_h_neg_tmp[key].append(cell.H_emiss[key]['h-'])
            if ne_tmp:
                self.los_1d['ortho_delL'][dl_idx] = np.sum(np.asarray(delL_tmp))
                delL_norm = np.asarray(delL_tmp) / self.los_1d['ortho_delL'][dl_idx]
                self.los_1d['ne'][dl_idx] = np.average(np.asarray(ne_tmp),weights = delL_norm)
                self.los_1d['n0'][dl_idx] = np.average(np.asarray(n0_tmp),weights = delL_norm)
                self.los_1d['te'][dl_idx] = np.average(np.asarray(te_tmp),weights = delL_norm)
                for key in self.spec_line_dict['1']['1']:
                    em[key][dl_idx] = np.sum(np.asarray(H_emiss_em_tmp[key]))
                    em_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_em_pervol_tmp[key]), weights = delL_norm )
                    recom[key][dl_idx] = np.sum(np.asarray(H_emiss_recom_tmp[key]))
                    excit[key][dl_idx] = np.sum(np.asarray(H_emiss_excit_tmp[key]))
                    h2[key][dl_idx] = np.sum(np.asarray(H_emiss_h2_tmp[key]))
                    h2_pos[key][dl_idx] = np.sum(np.asarray(H_emiss_h2_pos_tmp[key]))
                    h_neg[key][dl_idx] = np.sum(np.asarray(H_emiss_h_neg_tmp[key]))
                    recom_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_recom_tmp[key]), weights = delL_norm )
                    excit_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_excit_tmp[key]), weights = delL_norm )
                    h2_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_h2_tmp[key]), weights = delL_norm )
                    h2_pos_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_h2_pos_tmp[key]), weights = delL_norm )
                    h_neg_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_h_neg_tmp[key]), weights = delL_norm )
                
        # REMOVE ZEROS FROM COMPUTED LOS ARRAYS
        nonzero_idx = np.nonzero(self.los_1d['ne'])
        for item in ['l', 'ne', 'n0', 'te', 'ortho_delL']:
            self.los_1d[item] = self.los_1d[item][nonzero_idx]
            # convert numpy arrays to lists for JSON serialization
            self.los_1d[item] = list(self.los_1d[item])

        self.los_1d['H_emiss'] = {}
        self.los_1d['H_emiss_per_vol'] = {}
        for key in self.spec_line_dict['1']['1']:
            em[key] = em[key][nonzero_idx]
            em_pervol[key] = em_pervol[key][nonzero_idx]
            excit[key] = excit[key][nonzero_idx]
            excit_pervol[key] = excit_pervol[key][nonzero_idx]
            recom[key] = recom[key][nonzero_idx]
            recom_pervol[key] = recom_pervol[key][nonzero_idx]
            h2[key] = h2[key][nonzero_idx]
            h2_pervol[key] = h2_pervol[key][nonzero_idx]
            h2_pos[key] = h2_pos[key][nonzero_idx]
            h2_pos_pervol[key] = h2_pos_pervol[key][nonzero_idx]
            h_neg[key] = h_neg[key][nonzero_idx]
            h_neg_pervol[key] = h_neg_pervol[key][nonzero_idx]
            self.los_1d['H_emiss'].update({key:{'tot':em[key], 'excit': excit[key], 'recom': recom[key],
                                          'h2': h2[key], 'h2+': h2_pos[key], 'h-': h_neg[key],'units':'ph s^-1 m^-2 sr^-1'}})
            self.los_1d['H_emiss_per_vol'].update({key:{'tot':em_pervol[key], 'excit': excit_pervol[key], 'recom': recom_pervol[key],
                                          'h2': h2_pervol[key], 'h2+': h2_pos_pervol[key], 'h-': h_neg_pervol[key], 'units':'ph s^-1 m^-3 sr^-1'}})
            # convert numpy arrays to lists for JSON serialization
            self.los_1d['H_emiss'][key]['tot'] = list(self.los_1d['H_emiss'][key]['tot'])
            self.los_1d['H_emiss_per_vol'][key]['tot'] = list(self.los_1d['H_emiss_per_vol'][key]['tot'])
            self.los_1d['H_emiss'][key]['excit'] = list(self.los_1d['H_emiss'][key]['excit'])
            self.los_1d['H_emiss_per_vol'][key]['excit'] = list(self.los_1d['H_emiss_per_vol'][key]['excit'])
            self.los_1d['H_emiss'][key]['recom'] = list(self.los_1d['H_emiss'][key]['recom'])
            self.los_1d['H_emiss_per_vol'][key]['recom'] = list(self.los_1d['H_emiss_per_vol'][key]['recom'])
            self.los_1d['H_emiss'][key]['h2'] = list(self.los_1d['H_emiss'][key]['h2'])
            self.los_1d['H_emiss_per_vol'][key]['h2'] = list(self.los_1d['H_emiss_per_vol'][key]['h2'])
            self.los_1d['H_emiss'][key]['h2+'] = list(self.los_1d['H_emiss'][key]['h2+'])
            self.los_1d['H_emiss_per_vol'][key]['h2+'] = list(self.los_1d['H_emiss_per_vol'][key]['h2+'])
            self.los_1d['H_emiss'][key]['h-'] = list(self.los_1d['H_emiss'][key]['h-'])
            self.los_1d['H_emiss_per_vol'][key]['h-'] = list(self.los_1d['H_emiss_per_vol'][key]['h-'])


    def calc_int_and_1d_los_synth_spectra(self):

        ###############################################################
        # FREE-FREE AND FREE-BOUND CONTINUUM
        ###############################################################
        logger.info(f"Calculating ff fb continuum spectra for chord {self.chord_num}")
        wave_nm = np.linspace(300, 500, 50)

        # METHOD 1: SUM CELL-WISE (OK, good agreement with METHOD 2 below, so comment out)
        # sum_ff_fb = np.zeros((len(wave_nm)))
        # for cell in self.cells:
        #     # call adas continuo function (return units: ph s-1 m3 sr-1 nm-1)
        #     ff_only, ff_fb_tot = continuo_read.adas_continuo_py(wave_nm, cell.te, 1, 1)
        #     # convert to spectral radiance: ph s-1 m-2 sr-1 nm-1
        #     sum_ff_fb += ff_fb_tot * cell.ne * cell.ne * cell.los_ortho_delL

        # METHOD 2: SUM BASED ON AVERAGED 1D LOS PARAMS
        # TODO: ADD ZEFF CAPABILITY
        dl_ff_fb_abs = np.zeros((len(self.los_1d['l']), len(wave_nm)))
        for dl_idx, dl_val in enumerate(self.los_1d['l']):
            # call adas continuo function (return units: ph s-1 m3 sr-1 nm-1)
            ff_only, ff_fb_tot = continuo_read.adas_continuo_py(wave_nm, self.los_1d['te'][dl_idx], 1, 1)
            # convert to spectral radiance: ph s-1 m-2 sr-1 nm-1
            dl_ff_fb_abs[dl_idx] = ff_fb_tot * self.los_1d['ne'][dl_idx] * self.los_1d['ne'][dl_idx] * self.los_1d['ortho_delL'][dl_idx]

        # store spectra
        self.los_1d_spectra['ff_fb_continuum'] = {'wave':wave_nm, 'intensity':dl_ff_fb_abs, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
        self.los_int_spectra['ff_fb_continuum'] = {'wave':wave_nm, 'intensity':np.sum(dl_ff_fb_abs, axis=0), 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
        # convert numpy array to list for JSON serialization
        self.los_1d_spectra['ff_fb_continuum']['wave'] = list(self.los_1d_spectra['ff_fb_continuum']['wave'])
        self.los_1d_spectra['ff_fb_continuum']['intensity'] = list(self.los_1d_spectra['ff_fb_continuum']['intensity'])
        self.los_int_spectra['ff_fb_continuum']['wave'] = list(self.los_int_spectra['ff_fb_continuum']['wave'])
        self.los_int_spectra['ff_fb_continuum']['intensity'] = list(self.los_int_spectra['ff_fb_continuum']['intensity'])

        ###############################################################
        # STARK BROADENED H6-2 LINE
        ###############################################################
        logger.info(f"Calculating Stark H6-2 spectra for chord {self.chord_num}")
        # Generate modified lorentzian profile for each dl position along los (cf Lomanowski et al 2015, NF)
        for key in self.spec_line_dict['1']['1']:
            # H 6-2
            if self.spec_line_dict['1']['1'][key][1] == '2' and self.spec_line_dict['1']['1'][key][0] == '6':
                cwl = float(key) / 10.0 # nm
                # wave_nm = np.linspace(cwl-5, cwl+5, 10000)
                wave_nm = np.linspace(cwl-15, cwl+15, 1000)
                dl_stark = np.zeros((len(self.los_1d['l']), len(wave_nm)))
                for dl_idx, dl_val in enumerate(self.los_1d['l']):
                    stark_fwhm = ( self.lorentz_5_2_coeffs[key]['C']*np.power(self.los_1d['ne'][dl_idx], self.lorentz_5_2_coeffs[key]['a']) /
                                   np.power(self.los_1d['te'][dl_idx], self.lorentz_5_2_coeffs[key]['b']) )
                    dl_stark[dl_idx] = 1. / ( np.power(np.abs(wave_nm-cwl), 5./2.) + np.power(stark_fwhm / 2., 5./2.) )
                    # normalise by emissivity
                    dl_emiss = self.los_1d['H_emiss'][key]['excit'][dl_idx] + self.los_1d['H_emiss'][key]['recom'][dl_idx]
                    if self.data_source == "AMJUEL":
                        dl_emiss += self.los_1d['H_emiss'][key]['h2'][dl_idx]
                        dl_emiss += self.los_1d['H_emiss'][key]['h2+'][dl_idx]
                        dl_emiss += self.los_1d['H_emiss'][key]['h-'][dl_idx]
                    wv_area = np.trapz(dl_stark[dl_idx], x = wave_nm)
                    amp_scal = dl_emiss / wv_area
                    dl_stark[dl_idx] *= amp_scal

                # store spectra
                self.los_1d_spectra['stark'] = {'wavelength': key,'cwl':cwl, 'wave':wave_nm, 'intensity':dl_stark, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
                self.los_int_spectra['stark'] = {'wavelength': key,'cwl':cwl, 'wave':wave_nm, 'intensity':np.sum(dl_stark, axis=0), 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
                # convert numpy array to list for JSON serialization
                self.los_1d_spectra['stark']['wave'] = list(self.los_1d_spectra['stark']['wave'])
                self.los_1d_spectra['stark']['intensity'] = list(self.los_1d_spectra['stark']['intensity'])
                self.los_int_spectra['stark']['wave'] = list(self.los_int_spectra['stark']['wave'])
                self.los_int_spectra['stark']['intensity'] = list(self.los_int_spectra['stark']['intensity'])
