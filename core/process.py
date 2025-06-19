
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import pickle
import json

from matplotlib.collections import PatchCollection
from matplotlib import patches
from scipy.interpolate import interp1d
from .synth_diag import SynthDiag
from .utils.utils import isclose, interp_nearest_neighb, find_nearest
from .utils.amread import calc_photon_rate

from .utils.machine_defs import get_DIIIDdefs, get_JETdefs
from pyADASread import adas_adf11_read, adas_adf15_read, continuo_read
from .edge_code_formats import BackgroundPlasma, Cell, Edge2D, SOLPS
from .cherab_plasma import CherabPlasma


import logging
logger = logging.getLogger(__name__)
    
class ProcessEdgeSim:
    '''
    Class to read and store background plasma results from supported edge codes

    First all properties to be loaded are initialized as "None"
    Currently the code is very edge2d spesific, i.e. reading SOLPS does not actually work

    '''
    def __init__(self, ADAS_dict, edge_code_defs, use_AMJUEL = False, AMJUEL_date = 2016, ADAS_dict_lytrap=None,
                 machine='JET', pulse=90531, spec_line_dict=None, spec_line_dict_lytrap = None, 
                 diag_list=None, calc_synth_spec_features=None, save_synth_diag=False,
                 synth_diag_save_file=None, data2d_save_file=None, recalc_h2_pos=True, 
                 run_cherab = False, input_dict = None, **kwargs):

        self.ADAS_dict = ADAS_dict
        self.AMJUEL_date = AMJUEL_date
        self.use_AMJUEL = use_AMJUEL
        self.ADAS_dict_lytrap = ADAS_dict_lytrap
        self.spec_line_dict = spec_line_dict
        self.spec_line_dict_lytrap = spec_line_dict_lytrap
        self.edge_code = edge_code_defs['code']
        self.sim_path = edge_code_defs['sim_path']
        self.machine = machine
        self.pulse = pulse
        self.recalc_h2_pos = recalc_h2_pos
        self.input_dict = input_dict
        self.regions = {}

        # Dictionary for storing synthetic diagnostic objects
        self.synth_diag = {}
        
        if self.machine == 'JET':
            self.defs = get_JETdefs(pulse_ref=self.pulse)
        elif self.machine == 'DIIID':
            self.defs = get_DIIIDdefs()
        else:
            raise Exception("Unsupported machine. Currently supported machines are JET and DIIID")

        
        logger.info(f"Loading {self.edge_code} BG plasma from {self.sim_path}.")
        if self.edge_code == "edge2d":
            self.data = Edge2D(self.sim_path)
        elif self.edge_code == "solps":
            self.data = SOLPS(self.sim_path)
        elif self.edge_code == "oedge":
            '''
            TODO
            '''
        elif self.edge_code == "custom":
            try:
                from custom import Custom
                self.data = Custom(self.sim_path)
            except ImportError:
                logger.info("Could not load custom data format. Try creating a folder named \"custom\", \n and in the folder have \"__init__.py\", where you import your custom BG-plasma format.")
                raise ImportError
        else:
            logger.info("Edge code not supported")
            raise Exception("Edge code not supported")
        logger.info("   Data loaded")
        
        self.te = self.data.te
        self.ne = self.data.ne
        self.ni = self.data.ni
        self.n0 = self.data.n0
        self.n2 = self.data.n2
        self.n2p = self.data.n2p

        self.cells = self.data.cells   

        logger.info("Emission evaluation")
        # TODO: Add opacity calcs
        if run_cherab:
            logger.info("   Calculate emission via Cherab")
            # Currently the run cherab function uses the synth_diag to get the instrument and LOS details, so that needs to be generated
            self.run_cherab_bridge()
        else:
            logger.info("   Calcualte emission via cone integration")
            self.calc_H_emiss()
            self.calc_H_rad_power()
            self.calc_ff_fb_emiss()

            if diag_list:
                logger.info('       diag_list', diag_list)
                for key in diag_list:
                    if key in self.defs.diag_dict.keys():
                        self.synth_diag[key] = SynthDiag(self.defs, diag=key,
                                                        spec_line_dict = self.spec_line_dict,
                                                        spec_line_dict_lytrap=self.spec_line_dict_lytrap, 
                                                        use_AMJUEL = self.use_AMJUEL)
                        for chord in self.synth_diag[key].chords:
                            # Basic LOS implementation using 2D polygons - no reflections
                            self.los_intersect(chord)
                            chord.orthogonal_polys()
                            if self.use_AMJUEL:
                                chord.calc_int_and_1d_los_quantities_AMJUEL_1()
                            else:
                                chord.calc_int_and_1d_los_quantities_1()
                            if calc_synth_spec_features:
                                # Derived ne, te require information along LOS, calc emission again using _2 functions
                                if self.use_AMJUEL:
                                    chord.calc_int_and_1d_los_quantities_AMJUEL_2()
                                else:
                                    chord.calc_int_and_1d_los_quantities_2()
                                logger.info('       Calculating synthetic spectra for diag: ', key)
                                chord.calc_int_and_1d_los_synth_spectra()

            if save_synth_diag:
                if self.synth_diag:
                    self.save_synth_diag_data(savefile=synth_diag_save_file)

        if data2d_save_file:
            # pickle serialization of e2deirpostproc object
            output = open(data2d_save_file, 'wb')
            pickle.dump(self, output)
            output.close()

    def run_cherab_bridge(self): 
        # Inputs from cherab_bridge_input_dict
        import_jet_surfaces = self.input_dict['cherab_options']['import_jet_surfaces']
        include_reflections = self.input_dict['cherab_options']['include_reflections']
        spectral_bins = self.input_dict['cherab_options']['spectral_bins']
        pixel_samples = self.input_dict['cherab_options']['pixel_samples']
        spec_line_dict = self.input_dict['spec_line_dict']
        diag_list = self.input_dict['diag_list']
        use_AMJUEL = self.input_dict['run_options']['use_AMJUEL']
        recalc_h2_pos = self.input_dict['run_options'].get("recalc_h2_pos", True)
        calc_stark_ne = self.input_dict['cherab_options'].get('calculate_stark_ne', False)
        stark_transition = self.input_dict['cherab_options'].get('stark_transition', None)
        ff_fb = self.input_dict['cherab_options'].get('ff_fb_emission', False)
        #sion_H_transition = input_dict['cherab_options']['Sion_H_transition']
        #srec_H_transition = input_dict['cherab_options']['Srec_H_transition']
        # Generate cherab plasma
        transitions = [(int(val[0]), int(val[1])) for _, val in spec_line_dict['1']['1'].items()]
        plasma = CherabPlasma(self, self.ADAS_dict, 
                              include_reflections = include_reflections,
                              import_jet_surfaces = import_jet_surfaces, 
                              use_AMJUEL=use_AMJUEL, 
                              recalc_h2_pos = recalc_h2_pos,
                              transitions=transitions)

        # Create output dict
        self.outdict = {}

        diag_dict = get_JETdefs().diag_dict

        # Loop through diagnostics, their LOS, integrate over Lyman/Balmer
        for diag_key in diag_list:
            self.outdict[diag_key] = {}
            for diag_chord, los_p2 in enumerate(diag_dict[diag_key]['p2']):
                los_p2 = los_p2.tolist()
                los_p1 = diag_dict[diag_key]['p1'][0].tolist()
                los_w1 = 0.0 # unused
                los_w2 = diag_dict[diag_key]['w'][0][1]
                H_lines = spec_line_dict['1']['1']

                self.outdict[diag_key][str(diag_chord)] = {
                    'chord':{'p1':los_p1, 'p2':los_p2, 'w1':los_w1, 'w2':los_w2}
                }
                self.outdict[diag_key][str(diag_chord)]['spec_line_dict'] = spec_line_dict

                self.outdict[diag_key][str(diag_chord)]['los_int'] = {'H_emiss': {}}

                logger.info(diag_key, los_p2)
                for H_line_key, val in H_lines.items():
                
                    transition = (int(val[0]), int(val[1]))
                    wavelength = float(H_line_key)/10. #nm
                    min_wavelength = (wavelength)-1.0
                    max_wavelength = (wavelength)+1.0

                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition, include_excitation=True,  use_AMJUEL=use_AMJUEL)
                    exc_radiance, exc_radiance_std = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, wavelength=wavelength, pixel_samples=pixel_samples)

                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition, include_recombination=True, use_AMJUEL=use_AMJUEL)
                    rec_radiance, rec_radiance_std = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, wavelength=wavelength, pixel_samples=pixel_samples)
                    
                    if use_AMJUEL:
                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition, include_H2=True, use_AMJUEL=use_AMJUEL)
                        h2_radiance, h2_radiance_std = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, wavelength=wavelength, pixel_samples=pixel_samples)

                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition, include_H2_pos= True, use_AMJUEL=use_AMJUEL)
                        h2_pos_radiance, h2_pos_radiance_std = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, wavelength=wavelength, pixel_samples=pixel_samples)
                        
                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition, include_H3_pos=True, use_AMJUEL=use_AMJUEL)
                        h3_pos_radiance, h3_pos_radiance_std  = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2,wavelength=wavelength,  pixel_samples=pixel_samples)
                        
                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition, include_H_neg=True, use_AMJUEL=use_AMJUEL)
                        h_neg_radiance, h_neg_radiance_std = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, wavelength=wavelength, pixel_samples=pixel_samples)
                                                            
                        self.outdict[diag_key][str(diag_chord)]['los_int']['H_emiss'][H_line_key] = {
                            'excit':(np.array(exc_radiance)).tolist(),
                            'recom':(np.array(rec_radiance)).tolist(),
                            'h2': (np.array(h2_radiance)).tolist(),
                            'h2+': (np.array(h2_pos_radiance)).tolist(),
                            'h3+': (np.array(h3_pos_radiance)).tolist(),
                            'h-': (np.array(h_neg_radiance)).tolist(),
                            'units':'ph.s^-1.m^-2.sr^-1'
                        }
                    else:
                        self.outdict[diag_key][str(diag_chord)]['los_int']['H_emiss'][H_line_key] = {
                            'excit':(np.array(exc_radiance)).tolist(),
                            'recom':(np.array(rec_radiance)).tolist(),
                            'units':'ph.s^-1.m^-2.sr^-1'
                        }

                    if calc_stark_ne:
                        if transition == tuple(stark_transition):
                            print('Stark transition')
                            plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                    include_excitation=True, include_recombination=True, 
                                                    include_H2_pos= True, include_H2=True, include_H_neg=True,
                                                    include_H3_pos=True, use_AMJUEL=use_AMJUEL,
                                                    include_stark=True)
                            spec_bins = 50
                            radiance,  wave_arr = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, #spectrum,
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spec_bins,
                                                                                pixel_samples=pixel_samples,
                                                                                display_progress=False,no_avg = True)

                            self.outdict[diag_key][str(diag_chord)]['los_int']['stark']={'cwl': wavelength, 'wave': (np.array(wave_arr)).tolist(),
                                                                            'intensity': (np.array(radiance)).tolist(),
                                                                            'units': 'nm, ph s^-1 m^-2 sr^-1 nm^-1'}

                    # Free-free + free-bound using adaslib/continuo
                if ff_fb:
                    plasma.define_plasma_model(atnum=1, ion_stage=0,
                                                include_excitation=False, include_recombination=False, use_AMJUEL=use_AMJUEL,
                                                include_stark=False, include_ff_fb=True)
                    min_wave = 300
                    max_wave = 500
                    spec_bins = 50
                    radiance,  wave_arr = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, #spectrum,
                                                                            min_wave, max_wave,
                                                                            spectral_bins=spec_bins,
                                                                            pixel_samples=pixel_samples,
                                                                            display_progress=False, no_avg= True)

                    self.outdict[diag_key][str(diag_chord)]['los_int']['ff_fb_continuum'] = {
                            'wave': (np.array(wave_arr)).tolist(),
                            'intensity': (np.array(radiance)).tolist(),
                            'units': 'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
        #return outdict

    def __getstate__(self):
        """
            For removing the large ADAS_dict from the object for pickling
            See: https://docs/python/org/2/library/pickle.html#example
        """
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['ADAS_dict']
        return odict

    def __setstate__(self, dict):
        # TODO: Read external ADAS_dict object and add to dict for unpickling
        self.__dict__.update(dict)
    '''
    def interp_outlier_cells(self):
        """
            Replace outlier cell plasma properties with interpolated values from nearest specified neighbours.
            Uses average of neighbouring cell values weighted by distance to these cells.
        """
        for outlier_cell in self.outlier_cell_dict:
            for cell in self.cells:
                if isclose(cell.R, outlier_cell['outlier_RZ'][0], rel_tol=1e-09, abs_tol=rz_match_tol) and \
                        isclose(cell.Z, outlier_cell['outlier_RZ'][1], rel_tol=1e-09, abs_tol=rz_match_tol):
                    R = cell.R
                    Z = cell.Z
                    neighbs_RZ = []
                    neighb_ne = []
                    neighb_ni = []
                    neighb_n0 = []
                    neighb_n2 = []
                    neighb_Te = []
                    neighb_Ti = []
                    for i, neighb in enumerate(outlier_cell['neighbs_RZ']):
                        for _cell in self.cells:
                            if isclose(_cell.R, neighb[i][0], rel_tol=1e-09, abs_tol=rz_match_tol) and \
                                    isclose(_cell.Z, neighb[i][1], rel_tol=1e-09, abs_tol=rz_match_tol):
                                neighbs_RZ.append([_cell.R, _cell.Z])
                                neighb_ne.append(_cell.ne)
                                neighb_ni.append(_cell.ni)
                                neighb_n0.append(_cell.n0)
                                neighb_n2.append(_cell.n2)
                                neighb_Te.append(_cell.Te)
                                neighb_Ti.append(_cell.Ti)
                                break

                    # now interpolate
                    cell.ne = interp_nearest_neighb([R,Z], neighbs_RZ, neighb_ne)
                    cell.ni = interp_nearest_neighb([R,Z], neighbs_RZ, neighb_ni)
                    cell.n0 = interp_nearest_neighb([R,Z], neighbs_RZ, neighb_n0)
                    cell.n2 = interp_nearest_neighb([R,Z], neighbs_RZ, neighb_n2)
                    cell.Te = interp_nearest_neighb([R,Z], neighbs_RZ, neighb_Te)
                    cell.Ti = interp_nearest_neighb([R,Z], neighbs_RZ, neighb_Ti)
                break
    '''

    def save_synth_diag_data(self, savefile=None):
        # output = open(savefile, 'wb')
        outdict = {}
        for diag_key in self.synth_diag:
            outdict[diag_key] = {}
            for chord in self.synth_diag[diag_key].chords:
                outdict[diag_key].update({chord.chord_num:{}})
                # outdict[diag_key][chord.chord_num].update({'H_emiss':chord.los_int['H_emiss']})
                outdict[diag_key][chord.chord_num]['spec_line_dict'] = self.spec_line_dict
                outdict[diag_key][chord.chord_num]['spec_line_dict_lytrap'] = self.spec_line_dict_lytrap
                outdict[diag_key][chord.chord_num]['los_1d'] = chord.los_1d
                outdict[diag_key][chord.chord_num]['los_int'] = chord.los_int
                for spectrum in chord.los_int_spectra:
                    outdict[diag_key][chord.chord_num]['los_int'].update({spectrum:chord.los_int_spectra[spectrum]})
                    # outdict[diag_key][chord.chord_num]['los_1d'].update({spectrum: chord.los_1d_spectra[spectrum]})
                # outdict[diag_key][chord.chord_num]['Srec'] = chord.los_int['Srec']
                # outdict[diag_key][chord.chord_num]['Sion'] = chord.los_int['Sion']
                if chord.shply_intersects_w_sep and diag_key=='KT3':
                    outdict[diag_key][chord.chord_num].update({'chord':{'p1':chord.p1, 'p2':chord.p2unmod, 'w1': chord.w1,'w2':chord.w2unmod, 'sep_intersect_below_xpt':[chord.shply_intersects_w_sep.coords.xy[0][0],chord.shply_intersects_w_sep.coords.xy[1][0]]}})
                else:
                    outdict[diag_key][chord.chord_num].update({'chord':{'p1':chord.p1, 'p2':chord.p2unmod, 'w1': chord.w1,'w2':chord.w2unmod, 'sep_intersect_below_xpt':None}})
                if chord.los_angle:
                    outdict[diag_key][chord.chord_num]['chord']['los_angle'] = chord.los_angle

        # SAVE IN JSON FORMAT TO ENSURE PYTHON 2/3 COMPATIBILITY
        with open (savefile, mode='w', encoding='utf-8') as f:
            json.dump(outdict, f, indent=2)

        logger.info('Saving synthetic diagnostic data to:', savefile)

        # pickle.dump(outdict, output)
        #
        # output.close()


    def calc_H_emiss(self):
        '''
        Calculate the hydrogenic emission for spectral lines defined in the input

        If Use_AMJUEL flag set to true, calculate the emission with contributions from molecules and H-
        Otherwise used ADAS rates for contributions from el-impact excitation and recombination.

        TODO: addability to define path to AMJUEL.tex. Currently assume that the file is located in the home dir.

        '''
        logger.info('Calculating H emission...')
        if self.use_AMJUEL:
            logger.info('Using AMJUEL data')
            debug = True
            h3 = False
            if self.AMJUEL_date >= 2017:
                h3 = True
            for line_key in self.spec_line_dict['1']['1']:
                E_excit, E_recom, E_mol, E_h2_pos, E_h3_pos, E_h_neg, E_tot = calc_photon_rate(self.spec_line_dict['1']['1'][line_key], self.te, self.ne, self.n0, mol_n_density=self.n2,molp_n_density=self.n2p,p_density=self.ni, h3=h3, recalc_h2_pos=self.recalc_h2_pos, debug=debug)
                for k in range(len(self.cells)):
                    self.cells[k].H_emiss[line_key] = {'excit': E_excit[k], 'recom': E_recom[k], "h2": E_mol[k], "h2+": E_h2_pos[k], "h3+": E_h3_pos[k], "h-": E_h_neg[k], "tot": E_tot[k], 
                    "units": 'ph.s^-1.m^-3.sr^-1'}
                                      
        else: 
            logger.info("Using Adas")
            for cell in self.cells:
                for line_key in self.spec_line_dict['1']['1']:
                    E_excit, E_recom= adas_adf15_read.get_H_line_emiss(line_key, self.ADAS_dict['adf15']['1']['1'], cell.te, cell.ne*1.0E-06, cell.ni*1.0E-06, cell.n0*1.0E-06)
                    cell.H_emiss[line_key] = {'excit':E_excit, 'recom':E_recom, 'units':'ph.s^-1.m^-3.sr^-1'}

        if self.spec_line_dict_lytrap:
            logger.info('Calculating H emission for Ly trapping...')
            for cell in self.cells:
                for line_key in self.spec_line_dict_lytrap['1']['1']:
                    E_excit, E_recom= adas_adf15_read.get_H_line_emiss(line_key, self.ADAS_dict_lytrap['adf15']['1']['1'], cell.te, cell.ne*1.0E-06, cell.ni*1.0E-06, cell.n0*1.0E-06)
                    cell.H_emiss[line_key] = {'excit':E_excit, 'recom':E_recom, 'units':'ph.s^-1.m^-3.sr^-1'}


    def calc_ff_fb_filtered_emiss(self, filter_wv_nm, filter_tran):
        # TODO: ADD ZEFF CAPABILITY

        wave_nm = np.linspace(filter_wv_nm[0], filter_wv_nm[-1], 10)

        logger.info('Calculating FF+FB filtered emission...')
        for cell in self.cells:
            ff_only, ff_fb_tot = continuo_read.adas_continuo_py(wave_nm, cell.te, 1, 1)
            f = interp1d(wave_nm, ff_fb_tot)
            ff_fb_tot_interp = f(filter_wv_nm)
            # convert to spectral emissivity: ph s-1 m-3 sr-1 nm-1
            ff_fb_tot_interp = ff_fb_tot_interp * cell.ne * cell.ne
            # multiply by filter transmission
            ff_fb_tot_interp *= filter_tran
            # Integrate over wavelength
            ff_fb_tot_emiss = np.trapz(ff_fb_tot_interp, filter_wv_nm)
            cell.ff_fb_filtered_emiss = {'ff_fb':ff_fb_tot_emiss, 'units':'ph.s^-1.m^-3.sr^-1',
                                'filter_wv_nm':filter_wv_nm, 'filter_tran':filter_tran}

    def calc_ff_fb_emiss(self):
        # TODO: ADD ZEFF !

        wave_nm = np.logspace((0.001), np.log10(100000), 500)

        logger.info('Calculating FF+FB emission...')
        sum_ff_radpwr = 0
        for cell in self.cells:
            ff_only, ff_fb = continuo_read.adas_continuo_py(wave_nm, cell.te, 1, 1, output_in_ph_s=False)

            # convert to spectral emissivity (from W m^3 sr^-1 nm^-1 to W m^-3 nm^-1)
            ff_only = ff_only * cell.ne * cell.ne * 4. * np.pi
            ff_fb = ff_fb * cell.ne * cell.ne * 4. * np.pi

            # Integrate over wavelength
            cell.ff_radpwr_perm3 = np.trapz(ff_only, wave_nm) # W m^-3
            cell.ff_fb_radpwr_perm3 = np.trapz(ff_fb, wave_nm) # W m^-3

            cell_vol = cell.poly.area * 2.0 * np.pi * cell.R  # m^3
            cell.ff_radpwr = cell.ff_radpwr_perm3 * cell_vol
            cell.ff_fb_radpwr = cell.ff_fb_radpwr_perm3 * cell_vol

            sum_ff_radpwr += cell.ff_radpwr
        logger.info('Total ff radiated power:', sum_ff_radpwr, ' [W]')

    def calc_H_rad_power(self):
        # Te_rnge = [0.2, 5000]
        # ne_rnge = [1.0e11, 1.0e15]
        # self.H_adf11 = adas_adf11_utils.get_adas_H_adf11_interp(Te_rnge, ne_rnge, npts=self.ADAS_npts, npts_interp=1000, pwr=True)
        logger.info('Calculating H radiated power...')
        sum_pwr = 0
        for cell in self.cells:
            iTe, vTe = find_nearest(self.ADAS_dict['adf11']['1'].Te_arr, cell.te)
            ine, vne = find_nearest(self.ADAS_dict['adf11']['1'].ne_arr, cell.ne*1.0e-06)
            # plt/prb absolute rad pow contr in units W.cm^3
            plt_contr = self.ADAS_dict['adf11']['1'].plt[iTe,ine]*(1.0e-06*cell.n0)*(1.0e-06*cell.ne) #W.cm^-3
            prb_contr = self.ADAS_dict['adf11']['1'].prb[iTe,ine]*(1.0e-06*cell.ni)*(1.0e-06*cell.ne) #W.cm^-3
            cell_vol = cell.poly.area * 2.0 * np.pi * cell.R # m^3
            cell.H_radpwr = (plt_contr+prb_contr) * 1.e06 * cell_vol # Watts
            cell.H_radpwr_perm3 = (plt_contr+prb_contr) * 1.e06 # Watts m^-3

            sum_pwr += np.sum(np.asarray(cell.H_radpwr)) # sanity check. compare to eproc
        self.Prad_H = sum_pwr
        logger.info('Total H radiated power:', sum_pwr, ' [W]')

        if self.spec_line_dict_lytrap:
            logger.info('Calculating H radiated power for Ly trapping...')
            sum_pwr = 0
            for cell in self.cells:
                iTe, vTe = find_nearest(self.ADAS_dict_lytrap['adf11']['1'].Te_arr, cell.te)
                ine, vne = find_nearest(self.ADAS_dict_lytrap['adf11']['1'].ne_arr, cell.ne * 1.0e-06)
                # plt/prb absolute rad pow contr in units W.cm^3
                plt_contr = self.ADAS_dict_lytrap['adf11']['1'].plt[iTe, ine] * (1.0e-06 * cell.n0) * (
                            1.0e-06 * cell.ne)  # W.cm^-3
                prb_contr = self.ADAS_dict_lytrap['adf11']['1'].prb[iTe, ine] * (1.0e-06 * cell.ni) * (
                            1.0e-06 * cell.ne)  # W.cm^-3
                cell_vol = cell.poly.area * 2.0 * np.pi * cell.R  # m^3
                cell.H_radpwr_Lytrap = (plt_contr + prb_contr) * 1.e06 * cell_vol  # Watts
                cell.H_radpwr_Lytrap_perm3 = (plt_contr + prb_contr) * 1.e06  # Watts m^-3
                sum_pwr += np.sum(np.asarray(cell.H_radpwr_Lytrap))  # sanity check. compare to eproc
            self.Prad_H_Lytrap = sum_pwr
            logger.info('Total H radiated power w/ Ly trapping:', sum_pwr, ' [W]')


    def los_intersect(self, los):
        for cell in self.cells:
            # check if cell lies within los.poly
            if los.los_poly.contains(cell.poly):
                los.cells.append(cell)
            # check if cell interstects with los.poly
            elif los.los_poly.intersects(cell.poly):
                clipped_poly = los.los_poly.intersection(cell.poly)
                if clipped_poly.geom_type == 'Polygon':
                    centroid_p = clipped_poly.centroid
                    clipped_cell = Cell(centroid_p.x, centroid_p.y, poly=clipped_poly, te=cell.te, ne=cell.ne, ni=cell.ni, n0=cell.n0, Srec=cell.Srec, Sion=cell.Sion)
                    clipped_cell.H_emiss = cell.H_emiss
                    area_ratio = clipped_cell.poly.area /  cell.poly.area
                    clipped_cell.H_radpwr = cell.H_radpwr * area_ratio
                    clipped_cell.H_radpwr_perm3 = cell.H_radpwr_perm3
                    if self.spec_line_dict_lytrap:
                        clipped_cell.H_radpwr_Lytrap = cell.H_radpwr_Lytrap * area_ratio
                        clipped_cell.H_radpwr_Lytrap_perm3 = cell.H_radpwr_Lytrap_perm3
                    clipped_cell.ff_radpwr = np.asarray(cell.ff_radpwr) * area_ratio
                    clipped_cell.ff_radpwr_perm3 = cell.ff_radpwr_perm3
                    clipped_cell.ff_fb_radpwr = np.asarray(cell.ff_fb_radpwr) * area_ratio
                    clipped_cell.ff_fb_radpwr_perm3 = cell.ff_fb_radpwr_perm3
                    los.cells.append(clipped_cell)

        # Intersection of los centerline with separatrix: returns points where los crosses sep
        # generate los centerline shapely object
#        los.shply_cenline = LineString([(los.p1),(los.p2)])
#        if los.los_poly.intersects(self.shply_sep_poly_below_xpt):
#            los.shply_intersects_w_sep = None#los.shply_cenline.intersection(self.shply_sep_poly_below_xpt)


    def calc_qpol_div(self):

        """ TODO: CHECK THIS PROCEDURE WITH DEREK/JAMES"""

        # LFS
        endidx = self.data.qpartot_LFS['npts']
        xidx, = np.where(np.array(self.data.qpartot_LFS['xdata'][0:endidx]) > 0.0)
        # CONVERT QPAR TO QPOL (QPOL = QPAR * Btheta/Btot)
        qpol_LFS = np.array(self.data.qpartot_LFS['ydata'])[xidx] * np.array(self.data.bpol_btot_LFS['ydata'])[xidx]
        # calculate DR from neighbours
        dR_LFS = np.zeros((len(xidx)))
        for idx, val in enumerate(xidx):
            left_neighb = np.sqrt(((self.data.qpartot_LFS_rmesh['ydata'][val]-
                                    self.data.qpartot_LFS_rmesh['ydata'][val-1])**2)+
                                  ((self.data.qpartot_LFS_zmesh['ydata'][val]-
                                    self.data.qpartot_LFS_zmesh['ydata'][val-1])**2))
            if val != xidx[-1]:
                right_neighb = np.sqrt(((self.data.qpartot_LFS_rmesh['ydata'][val+1]-
                                         self.data.qpartot_LFS_rmesh['ydata'][val])**2)+
                                       ((self.data.qpartot_LFS_zmesh['ydata'][val+1]-
                                         self.data.qpartot_LFS_zmesh['ydata'][val])**2))
            else:
                right_neighb = left_neighb
            dR_LFS[idx] = (left_neighb+right_neighb)/2.0
        area = 2. * np.pi * np.array(self.data.qpartot_LFS_rmesh['ydata'])[xidx] * dR_LFS
        self.qpol_div_LFS = np.sum(qpol_LFS*area)

        # HFS
        endidx = self.data.qpartot_HFS['npts']
        xidx, = np.where(np.array(self.data.qpartot_HFS['xdata'])[0:endidx] > 0.0)
        # CONVERT QPAR TO QPOL (QPOL = QPAR * Btheta/Btot)
        qpol_HFS = np.array(self.data.qpartot_HFS['ydata'])[xidx] * np.array(self.data.bpol_btot_HFS['ydata'])[xidx]
        # calculate dR from neighbours
        dR_HFS = np.zeros((len(xidx)))
        for idx, val in enumerate(xidx):
            left_neighb = np.sqrt(((self.data.qpartot_HFS_rmesh['ydata'][val]-
                                    self.data.qpartot_HFS_rmesh['ydata'][val-1])**2)+
                                  ((self.data.qpartot_HFS_zmesh['ydata'][val]-
                                    self.data.qpartot_HFS_zmesh['ydata'][val-1])**2))
            if val != xidx[-1]:
                right_neighb = np.sqrt(((self.data.qpartot_HFS_rmesh['ydata'][val+1]-
                                         self.data.qpartot_HFS_rmesh['ydata'][val])**2)+
                                       ((self.data.qpartot_HFS_zmesh['ydata'][val+1]-
                                         self.data.qpartot_HFS_zmesh['ydata'][val])**2))
            else:
                right_neighb = left_neighb
            dR_HFS[idx] = (left_neighb+right_neighb)/2.0
        area = 2. * np.pi * np.array(self.data.qpartot_HFS_rmesh['ydata'])[xidx] * dR_HFS
        self.qpol_div_HFS = np.sum(qpol_HFS*area)

        print('Pdiv_LFS (MW): ', self.qpol_div_LFS*1.e-06, 'Pdiv_HFS (MW): ', 
              self.qpol_div_HFS*1.e-06, 'POWSOL (MW): ', 
              self.data.powsol['data'][ self.data.powsol['npts']-1]*1.e-06)


    def calc_region_aggregates(self):

        # Calculates for each region:
        #   radiated power
        #   ionisation and recombination
        for regname, region in self.regions.items():
            for cell in self.cells:
                if region.cell_in_region(cell, self.data.shply_sep_poly):
                    region.cells.append(cell)
                    region.Prad_units = 'W'
                    region.Prad_H += cell.H_radpwr
                    if self.spec_line_dict_lytrap:
                        region.Prad_H_Lytrap += cell.H_radpwr_Lytrap
                    # ionization/recombination * cell volume
                    region.Sion += cell.Sion * 2.*np.pi*cell.R * cell.poly.area
                    region.Srec += cell.Srec * 2.*np.pi*cell.R * cell.poly.area

    def plot_region(self, name='LFS_DIV'):

        fig, ax = plt.subplots(ncols=1)
        region_patches = []
        for regname, region in self.regions.items():
            if regname == name:
                for cell in region.cells:
                    region_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=False))

        # region_patches.append(patches.Polygon(self.shply_sep_poly.exterior.coords, closed=False))
        coll = PatchCollection(region_patches)
        ax.add_collection(coll)
        ax.set_xlim(1.8, 4.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_title(name)
        ax.add_patch(self.sep_poly)
        ax.add_patch(self.wall_poly)
        plt.axes().set_aspect('equal')

if __name__=='__main__':
    print('To run PESDT, use the "PESDT_run.py" script in the root of PESDT')
