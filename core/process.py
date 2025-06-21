
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
    def __init__(self, ADAS_dict, edge_code_defs, data_source = "AMJUEL", AMJUEL_date = 2016, ADAS_dict_lytrap=None,
                 machine='JET', pulse=90531, spec_line_dict=None, spec_line_dict_lytrap = None, 
                 diag_list=None, calc_synth_spec_features=None, save_synth_diag=False,
                 synth_diag_save_file=None, data2d_save_file=None, recalc_h2_pos=True, 
                 run_cherab = False, input_dict = None, **kwargs):

        self.ADAS_dict = ADAS_dict
        self.AMJUEL_date = AMJUEL_date
        self.data_source = data_source
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
            self.run_cherab_raytracing()
        else:
            logger.info("   Calcualte emission via cone integration")
            self.calc_H_emiss()
            self.calc_H_rad_power()
            self.calc_ff_fb_emiss()

            if diag_list:
                logger.info(f"       diag_list: {diag_list}")
                for key in diag_list:
                    if key in self.defs.diag_dict.keys():
                        self.synth_diag[key] = SynthDiag(self.defs, diag=key,
                                                        spec_line_dict = self.spec_line_dict,
                                                        spec_line_dict_lytrap=self.spec_line_dict_lytrap, 
                                                        data_source = self.data_source)
                        for chord in self.synth_diag[key].chords:
                            # Basic LOS implementation using 2D polygons - no reflections
                            self.los_intersect(chord)
                            chord.orthogonal_polys()
                            if self.data_source == "AMJUEL":
                                chord.calc_int_and_1d_los_quantities_AMJUEL_1()
                            else:
                                chord.calc_int_and_1d_los_quantities_1()
                            if calc_synth_spec_features:
                                # Derived ne, te require information along LOS, calc emission again using _2 functions
                                if self.data_source == "AMJUEL":
                                    chord.calc_int_and_1d_los_quantities_AMJUEL_2()
                                else:
                                    chord.calc_int_and_1d_los_quantities_2()
                                logger.info(f"       Calculating synthetic spectra for diag:  {key}")
                                chord.calc_int_and_1d_los_synth_spectra()

            if save_synth_diag:
                if self.synth_diag:
                    self.save_synth_diag_data(savefile=synth_diag_save_file)

        if data2d_save_file:
            # pickle serialization of e2deirpostproc object
            output = open(data2d_save_file, 'wb')
            pickle.dump(self, output)
            output.close() 

    def run_cherab_raytracing(self):
        # === Load Configuration ===
        cherab_opts = self.input_dict["cherab_options"]
        run_opts = self.input_dict["run_options"]
        spec_line_dict = self.input_dict["spec_line_dict"]
        diag_list = self.input_dict["diag_list"]

        pixel_samples = cherab_opts.get("pixel_samples", 1000)
        include_reflections = cherab_opts.get("include_reflections", True)
        import_jet_surfaces = cherab_opts.get("import_jet_surfaces", True)
        calc_stark_ne = cherab_opts.get("calculate_stark_ne", False)
        stark_transition = cherab_opts.get("stark_transition", None)
        ff_fb = cherab_opts.get("ff_fb_emission", False)

        data_source = run_opts.get("data_source", "AMJUEL")
        recalc_h2_pos = run_opts.get("recalc_h2_pos", True)

        diag_def = get_JETdefs().diag_dict
        transitions = [(int(v[0]), int(v[1])) for _, v in spec_line_dict['1']['1'].items()]
        
        # === Initialize Plasma ===
        plasma = CherabPlasma(self, self.ADAS_dict,
                            include_reflections=include_reflections,
                            import_jet_surfaces=import_jet_surfaces,
                            data_source=data_source,
                            recalc_h2_pos=recalc_h2_pos,
                            transitions=transitions)
        
        # === Setup Observers ===
        instrument_los_dict = {}
        for diag in diag_list:
            los_points = []
            p1 = diag_def[diag]["p1"][0].tolist()
            w1 = 0.0
            w2 = diag_def[diag]["w"][0][1]
            for p2 in diag_def[diag]["p2"]:
                los_points.append((p1, p2.tolist(), w1, w2))
            instrument_los_dict[diag] = los_points

        plasma.setup_observers(instrument_los_dict, pixel_samples=pixel_samples)

        # === Setup Spectral Observers if needed ===
        # Figure out Stark wavelength from spec_line_dict
        if calc_stark_ne:
            stark_wavelength_nm = None
            for line_key, val in spec_line_dict["1"]["1"].items():
                if (int(val[0]), int(val[1])) == tuple(stark_transition):
                    stark_wavelength_nm = float(line_key) / 10.0
                    break

            if stark_wavelength_nm is not None:
                min_wave = stark_wavelength_nm - 1.0
                max_wave = stark_wavelength_nm + 1.0
                plasma.setup_spectral_observers(instrument_los_dict,
                                                min_wavelength_nm=min_wave,
                                                max_wavelength_nm=max_wave,
                                                destination="stark",
                                                pixel_samples=pixel_samples,
                                                spectral_bins= 1,
                                                spectral_rays= 1)

        if ff_fb:
            plasma.setup_spectral_observers(instrument_los_dict,
                                            min_wavelength_nm=300,
                                            max_wavelength_nm=500,
                                            destination="continuum",
                                            pixel_samples=pixel_samples,
                                            spectral_bins= 1,
                                            spectral_rays= 1)

        self.outdict = {}

        # === Process Each Instrument ===
        for diag in diag_list:
            self.outdict[diag] = {}

            p1 = diag_def[diag]["p1"][0].tolist()
            w1 = 0.0
            w2 = diag_def[diag]["w"][0][1]

            los_coords = []
            for p2 in diag_def[diag]["p2"]:
                los_coords.append({"p1": p1, "p2": p2.tolist(), "w1": w1, "w2": w2})
            self.outdict[diag]["chord"] = los_coords

            H_lines = spec_line_dict['1']['1']
            self.outdict[diag]["units"] = "ph s^-1 m^-2 sr^-1"

            for line_key, trans in H_lines.items():
                transition = (int(trans[0]), int(trans[1]))
                wavelength = line_key
                self.outdict[diag][wavelength] = {}

                # Excitation
                logger.info("Excitation")
                plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                        include_excitation=True, data_source=data_source)
                excit = plasma.integrate_instrument(diag)
                self.outdict[diag][wavelength]["excit"] = [x[0] for x in excit]

                # Recombination
                logger.info("Recombination")
                plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                        include_recombination=True, data_source=data_source)
                recom = plasma.integrate_instrument(diag)
                self.outdict[diag][wavelength]["recom"] = [x[0] for x in recom]

                # Molecular / negative H species
                if data_source == "AMJUEL":
                    logger.info("H2")
                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                            include_H2=True, data_source=data_source)
                    self.outdict[diag][wavelength]["h2"] = [x[0] for x in plasma.integrate_instrument(diag)]
                    logger.info("H2+")
                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                            include_H2_pos=True, data_source=data_source)
                    self.outdict[diag][wavelength]["h2+"] = [x[0] for x in plasma.integrate_instrument(diag)]
                    logger.info("H3+")
                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                            include_H3_pos=True, data_source=data_source)
                    self.outdict[diag][wavelength]["h3+"] = [x[0] for x in plasma.integrate_instrument(diag)]
                    logger.info("H-")
                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                            include_H_neg=True, data_source=data_source)
                    self.outdict[diag][wavelength]["h-"] = [x[0] for x in plasma.integrate_instrument(diag)]

                # === Optional Stark Spectrum ===
                if calc_stark_ne and transition == tuple(stark_transition):
                    logger.info("Stark")
                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                            include_excitation=True, include_recombination=True,
                                            include_H2=True, include_H2_pos=True,
                                            include_H3_pos=True, include_H_neg=True,
                                            include_stark=True, data_source=data_source)

                    spec, wl = plasma.integrate_instrument_spectral(diag, destination="stark")
                    self.outdict[diag]["stark"] = {
                        "intensity": spec,
                        "wave": wl[0], # same wavelengths for all chords
                        "units": "nm, ph s^-1 m^-2 sr^-1 nm^-1",
                        "wavelength": wavelength,
                        "cwl": 0.1*float(wavelength)
                        }
            # === Optional FF+FB Spectrum ===
            if ff_fb:
                logger.info("Continuum")
                plasma.define_plasma_model(atnum=1, ion_stage=0, data_source=data_source, include_ff_fb=True)
                spec, wl = plasma.integrate_instrument_spectral(diag, destination="continuum")
                self.outdict[diag]["ff_fb_continuum"] = {
                    "wave": wl,
                    "intensity": spec,
                    "units": "nm, ph s^-1 m^-2 sr^-1 nm^-1"
                }


    def run_cherab_raytracing_old(self): 
        # Inputs from cherab_bridge_input_dict
        import_jet_surfaces = self.input_dict['cherab_options'].get('import_jet_surfaces', True)
        include_reflections = self.input_dict['cherab_options'].get('include_reflections', True)
        pixel_samples = self.input_dict['cherab_options'].get('pixel_samples', 1000)
        spec_line_dict = self.input_dict['spec_line_dict']
        diag_list = self.input_dict['diag_list']
        data_source = self.input_dict['run_options'].get('data_source', "AMJUEL")
        recalc_h2_pos = self.input_dict['run_options'].get("recalc_h2_pos", True)
        calc_stark_ne = self.input_dict['cherab_options'].get('calculate_stark_ne', False)
        stark_transition = self.input_dict['cherab_options'].get('stark_transition', None)
        ff_fb = self.input_dict['cherab_options'].get('ff_fb_emission', False)

        # Generate cherab plasma
        transitions = [(int(val[0]), int(val[1])) for _, val in spec_line_dict['1']['1'].items()]
        plasma = CherabPlasma(self, self.ADAS_dict, 
                              include_reflections = include_reflections,
                              import_jet_surfaces = import_jet_surfaces, 
                              data_source=data_source, 
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

                print(diag_key, los_p2)
                for H_line_key, val in H_lines.items():
                
                    transition = (int(val[0]), int(val[1]))
                    wavelength = float(H_line_key)/10. #nm
                    

                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition, include_excitation=True,  data_source=data_source)
                    exc_radiance, exc_radiance_std = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, wavelength=wavelength, pixel_samples=pixel_samples)

                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition, include_recombination=True, data_source=data_source)
                    rec_radiance, rec_radiance_std = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, wavelength=wavelength, pixel_samples=pixel_samples)
                    
                    if data_source == "AMJUEL":
                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition, include_H2=True, data_source=data_source)
                        h2_radiance, h2_radiance_std = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, wavelength=wavelength, pixel_samples=pixel_samples)

                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition, include_H2_pos= True, data_source=data_source)
                        h2_pos_radiance, h2_pos_radiance_std = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, wavelength=wavelength, pixel_samples=pixel_samples)
                        
                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition, include_H3_pos=True, data_source=data_source)
                        h3_pos_radiance, h3_pos_radiance_std  = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2,wavelength=wavelength,  pixel_samples=pixel_samples)
                        
                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition, include_H_neg=True, data_source=data_source)
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
                            min_wavelength = (wavelength)-1.0
                            max_wavelength = (wavelength)+1.0
                            print('Stark transition')
                            plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                    include_excitation=True, include_recombination=True, 
                                                    include_H2_pos= True, include_H2=True, include_H_neg=True,
                                                    include_H3_pos=True, data_source=data_source,
                                                    include_stark=True)
                            spec_bins = 50
                            radiance,  wave_arr = plasma.integrate_los_spectral(los_p1, los_p2, los_w1, los_w2, 
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spec_bins,
                                                                                pixel_samples=pixel_samples,
                                                                                display_progress=False)

                            self.outdict[diag_key][str(diag_chord)]['los_int']['stark']={'cwl': wavelength, 'wave': (np.array(wave_arr)).tolist(),
                                                                            'intensity': (np.array(radiance)).tolist(),
                                                                            'units': 'nm, ph s^-1 m^-2 sr^-1 nm^-1'}

                    # Free-free + free-bound using adaslib/continuo
                if ff_fb:
                    plasma.define_plasma_model(atnum=1, ion_stage=0, data_source=data_source, include_ff_fb=True)
                    min_wave = 300
                    max_wave = 500
                    spec_bins = 50
                    radiance,  wave_arr = plasma.integrate_los_spectral(los_p1, los_p2, los_w1, los_w2, #spectrum,
                                                                            min_wave, max_wave,
                                                                            spectral_bins=spec_bins,
                                                                            pixel_samples=pixel_samples,
                                                                            display_progress=False)

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

    def save_synth_diag_data_old(self, savefile=None):
        # output = open(savefile, 'wb')
        outdict = {}
        for diag_key in self.synth_diag:
            outdict[diag_key] = {}
            for chord in self.synth_diag[diag_key].chords:
                outdict[diag_key].update({chord.chord_num:{}})
                outdict[diag_key][chord.chord_num]['spec_line_dict'] = self.spec_line_dict
                outdict[diag_key][chord.chord_num]['los_1d'] = chord.los_1d
                outdict[diag_key][chord.chord_num]['los_int'] = chord.los_int
                for spectrum in chord.los_int_spectra:
                    outdict[diag_key][chord.chord_num]['los_int'].update({spectrum:chord.los_int_spectra[spectrum]})
                outdict[diag_key][chord.chord_num].update({'chord':{'p1':chord.p1, 'p2':chord.p2unmod, 'w1': chord.w1,'w2':chord.w2unmod, 'sep_intersect_below_xpt':None}})

        # SAVE IN JSON FORMAT TO ENSURE PYTHON 2/3 COMPATIBILITY
        with open (savefile, mode='w', encoding='utf-8') as f:
            json.dump(outdict, f, indent=2)

        logger.info(f"Saving synthetic diagnostic data to:{savefile}")

    def save_synth_diag_data(self, savefile=None):
        """
        Save synthetic diagnostic data in CHERAB-style format with:
        - Emissivity data: dict[diag][wavelength][source]
        - Chord geometry & per-chord data: dict[diag]["chord"] = [dicts]
        - FF spectra and LOS 1D profiles per chord included
        """
        outdict = {}

        for diag_key, diag_obj in self.synth_diag.items():
            diag_dict = {"units": "ph.s^-1.m^-2.sr^-1", 
                         "chord": [], 
                         "stark": {"cwl": 410.12, "wavelength": "4101.2", "transition": [6,2], "wave": [], "intensity": []}, 
                         "ff_fb_continuum": {"wave": [], "intensity": []}}

            # Prepare structure for each wavelength and emission type
            for chord in diag_obj.chords:
                for wl, src_vals in chord.los_int.get("H_emiss", {}).items():
                    if wl not in diag_dict:
                        diag_dict[wl] = {}
                    for src_key in src_vals:
                        if src_key not in diag_dict[wl] and src_key != "units":
                            diag_dict[wl][src_key] = []
            
            # Get stark and ff_fb wave vectors
            diag_dict["stark"]["cwl"] = diag_obj.chords[0].los_int_spectra.get('stark', {}).get("cwl", 410.12)
            diag_dict["stark"]["wavelength"] = diag_obj.chords[0].los_int_spectra.get('stark', {}).get("wavelength", "4101.2")
            diag_dict["stark"]["wave"] = (diag_obj.chords[0].los_int_spectra.get('stark', {}).get("wave", []))
            diag_dict["ff_fb_continuum"]["wave"] = (diag_obj.chords[0].los_int_spectra.get('ff_fb_continuum', {}).get("wave", []))

            # Store data per chord
            for chord in diag_obj.chords:
                # Fill in wavelength-integrated emission values
                for wl, src_vals in chord.los_int.get("H_emiss", {}).items():
                    for src_key, val in src_vals.items():
                        if src_key != "units":
                            diag_dict[wl][src_key].append(val)
                # Fill stark emission and ff_fb_continuum
                diag_dict["stark"]["intensity"].append(chord.los_int_spectra.get('stark', {}).get("intensity", []))
                diag_dict["ff_fb_continuum"]["intensity"].append(chord.los_int_spectra.get('ff_fb_continuum', {}).get("intensity", []))

                # Create chord dictionary with geometry + los_1d + spectrum
                chord_entry = {
                    "p1": chord.p1,
                    "p2": chord.p2unmod,
                    "w1": chord.w1,
                    "w2": chord.w2unmod,
                    "los_1d": chord.los_1d,                         # 1D line-of-sight emission profile
                }
                diag_dict["chord"].append(chord_entry)

            outdict[diag_key] = diag_dict

        # Save to JSON
        with open(savefile, mode="w", encoding="utf-8") as f:
            json.dump(outdict, f, indent=2)

        logger.info(f"Saved synthetic diagnostic data to: {savefile}")


    def calc_H_emiss(self):
        '''
        Calculate the hydrogenic emission for spectral lines defined in the input

        If data source is AMJUEL, calculate the emission with contributions from molecules and H-
        Otherwise used ADAS rates for contributions from el-impact excitation and recombination.

        TODO: addability to define path to AMJUEL.tex. Currently assume that the file is located in the home dir.

        '''
        logger.info('Calculating H emission...')
        if self.data_source == "AMJUEL":
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
        logger.info(f"Total ff radiated power: {sum_ff_radpwr} [W] ")

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
        logger.info(f"Total H radiated power: {sum_pwr} [W] ")

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
            logger.info(f"Total H radiated power w/ Ly trapping: {sum_pwr} [W] ")
      

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

if __name__=='__main__':
    print('To run PESDT, use the "PESDT_run.py" script in the root of PESDT')
