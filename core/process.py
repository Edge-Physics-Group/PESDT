
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import pickle, json, os, errno

from matplotlib.collections import PatchCollection
from matplotlib import patches
from scipy.interpolate import interp1d
from .synth_diag import SynthDiag
from .utils.utils import isclose, interp_nearest_neighb, find_nearest
from .utils.amread import calc_photon_rate
from .utils import get_ADAS_dict
from .utils.machine_defs import get_DIIIDdefs, get_JETdefs
from pyADASread import adas_adf11_read, adas_adf15_read, continuo_read
from .edge_code_formats import BackgroundPlasma, Cell, Edge2D, SOLPS, OEDGE
from .cherab import CherabPlasma
from .analyse import recover_line_int_ff_fb_Te, recover_line_int_Stark_ne, recover_line_int_particle_bal, recover_delL_atomden_product


import logging
logger = logging.getLogger(__name__)
    
class ProcessEdgeSim:
    '''
    Class to read and store background plasma results from supported edge codes, and run


    '''
    def __init__(self, input_dict):
        self.input_dict = input_dict
        # READ ENV VARIABLES
        self.cache_dir = os.environ.get("PESDTCacheDir", os.path.join(os.path.expanduser('~'), "PESDTCache/"))
        # Read input deck and populate run parameters
        self.parse_input_deck()
        # Read neutral data source
        self.load_neutral_data_sources()
        # Dictionary for storing synthetic diagnostic objects
        self.synth_diag = {}
        # Load machine definitions
        if self.machine == 'JET':
            self.defs = get_JETdefs(pulse_ref=self.pulse)
        elif self.machine == 'DIIID':
            self.defs = get_DIIIDdefs()
        else:
            raise Exception("Unsupported machine. Currently supported machines are JET and DIIID")

        self.load_edge_data()
        

        logger.info("Emission evaluation")
        if self.run_cherab:
            logger.info("   Calculate emission via Cherab")
            # Currently the run cherab function uses the synth_diag to get the instrument and LOS details, so that needs to be generated
            self.run_cherab_raytracing()
            if self.calc_synth_spec_features:
                try:
                    recover_line_int_Stark_ne(self.outdict)
                    if input_dict['cherab_options'].get('ff_fb_emission', False):
                        recover_line_int_ff_fb_Te(self.outdict)
                except Exception as e:
                    # SafeGuard for possible issues, so that not all comp. time is lost 
                    logger.error(f'Something went wrong with AnalyseSynthDiag, error{e}')
                    pass
            # SAVE IN JSON FORMAT TO ENSURE PYTHON 2/3 COMPATIBILITY
            if self.input_dict["cherab_options"].get('include_reflections', False):
                logger.info("Saving cherab reflections")
                savefile = self.savedir + '/cherab_refl.synth_diag.json'
            else:
                logger.info("Saving cherab no reflections")
                savefile = self.savedir + '/cherab.synth_diag.json'
            with open(savefile, mode='w', encoding='utf-8') as f:
                json.dump(self.outdict, f, indent=2)
        else:
            logger.info("   Calculate emission via cone integration")
            self.run_cone_integration()
            if self.input_dict['run_options']['analyse_synth_spec_features']:
            # Read synth diag saved data
                try:
                    self.analyse_synth_spectra(self.outdict)
                except IOError as e:
                    raise
            with open(self.synth_diag_save_file, mode='w', encoding='utf-8') as f:
                json.dump(self.outdict, f, indent=2)
            logger.info(f"Saved synthetic diagnostic data to: {self.synth_diag_save_file}")

        if self.data2d_save_file:
            # pickle serialization of e2deirpostproc object
            output = open(self.data2d_save_file, 'wb')
            pickle.dump(self, output)
            output.close() 

    def parse_input_deck(self):
        """
        Reads the input deck and populates run parameters
        """

        tmpstr = self.input_dict['edge_code']['sim_path'].replace('/','_')
        logger.info(f"{self.input_dict['edge_code']['sim_path']}")
        if tmpstr[:3] == '_u_':
            tmpstr = tmpstr[3:]
        elif tmpstr[:6] == '_work_':
            tmpstr = tmpstr[6:]
        else:
            tmpstr = tmpstr[1:]

        self.savedir = self.input_dict['save_dir'] + tmpstr + '/'

        # Create dir from tran file, if it does not exist
        try:
            os.makedirs(self.savedir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.data2d_save_file = self.savedir +'PESDT.2ddata.pkl'
        self.synth_diag_save_file = self.savedir + 'PESDT.synth_diag.json'
        self.spec_line_dict = self.input_dict['spec_line_dict']

        # Option to run cherab
        self.run_cherab = self.input_dict["run_options"].get('run_cherab', False)

        # Option to use cherab ne and Te fits rather than pyproc's. Use case - Lyman opacity adas data is not suppored
        # by cherab-bridge, so import cherab plasma parameter profiles with reflections impact here instead and apply
        # to Siz and Srec estimates with modified Ly-trapping adas data
        self.cherab_ne_Te_KT3_resfile = self.input_dict['run_options'].get('use_cherab_resfile_for_KT3_ne_Te_fits', None)
        self.diag_list = self.input_dict['diag_list']
        self.calc_synth_spec_features = self.input_dict['run_options']['analyse_synth_spec_features']
        self.AMJUEL_date = self.input_dict['run_options'].get("AMJUEL_date", 2016) # Default to <2017 (no H3+)
        self.data_source = self.input_dict["run_options"].get("data_source", "AMJUEL")
        self.edge_code = self.input_dict['edge_code']['code']
        self.sim_path = self.input_dict['edge_code']['sim_path']
        self.machine = self.input_dict.get('machine', "JET")
        self.pulse = self.input_dict.get('pulse', 81472)
        self.recalc_h2_pos = self.input_dict['run_options'].get('recalc_h2_pos', True)
        self.run_cherab = self.input_dict['run_options'].get('run_cherab', False)

        # Location of adf15 and adf11 ADAS data modified for Ly-series opacity with escape factor method
        self.adas_lytrap = self.input_dict.get('read_ADAS_lytrap', None)
        if self.adas_lytrap is not None:
            self.spec_line_dict_lytrap = self.adas_lytrap['spec_line_dict']
        else:
            self.spec_line_dict_lytrap = None

    def load_neutral_data_sources(self):
        """
        Reads the
        """
        if self.data_source == "AMJUEL":
            pass # we have amread, and the user needs to supply the .tex file
        elif self.data_source == "YACORA":
            # Look for YACORA rates in the home folder, unless specified otherwise in the input
            self.YACORA_RATES_PATH = self.input_dict.get("YACORA_RATES_PATH", os.path.join(os.path.expanduser("~"), "YACORA_RATES/" ))
        elif self.data_source == "ADAS":
            
            # Look for Lyman trapping modified ADAS data
            if self.adas_lytrap is not None:
                self.ADAS_dict_lytrap = get_ADAS_dict(self.savedir,
                                                    self.spec_line_dict_lytrap,
                                                    restore=not self.input_dict['read_ADAS_lytrap']['read'],
                                                    adf11_year = self.adas_lytrap['adf11_year'],
                                                    lytrap_adf11_dir=self.adas_lytrap['adf11_dir'],
                                                    lytrap_pec_file=self.adas_lytrap['pec_file'])

            # Also get standard ADAS data
            self.ADAS_dict = get_ADAS_dict(self.cache_dir,
                                        self.spec_line_dict, adf11_year=12, restore=not self.input_dict['read_ADAS'])


        logger.info(f"diag_list: {self.input_dict['diag_list']}")

    def load_edge_data(self):
        logger.info(f"Loading {self.edge_code} BG plasma from {self.sim_path}.")
        if self.edge_code == "edge2d":
            try:
                self.data = Edge2D(self.sim_path)
            except Exception as e:
                logger.warning(f"   Error: {e}")
                logger.info("   Trying to load with eproc")
                from .edge_code_formats.edge2d_format_old import Edge2D_old
                self.data = Edge2D_old(self.sim_path)
        elif self.edge_code == "solps":
            self.data = SOLPS(self.sim_path)
        elif self.edge_code == "oedge":
            self.data = OEDGE(self.sim_path)
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

    def run_cherab_raytracing(self):
        # === Load Configuration ===
        cherab_opts = self.input_dict["cherab_options"]
        run_opts = self.input_dict["run_options"]
        spec_line_dict = self.input_dict["spec_line_dict"]
        diag_list = self.input_dict["diag_list"]

        # === Run Options ===
        data_source = run_opts.get("data_source", "AMJUEL")
        recalc_h2_pos = run_opts.get("recalc_h2_pos", True)

        # === Cherab Options ===
        num_processes = cherab_opts.get("num_processes", 1)
        pixel_samples = cherab_opts.get("pixel_samples", 1000)
        include_reflections = cherab_opts.get("include_reflections", True)
        import_jet_surfaces = cherab_opts.get("import_jet_surfaces", True)
        calc_stark_ne = cherab_opts.get("calculate_stark_ne", False)
        stark_transition = cherab_opts.get("stark_transition", None)
        stark_bins = cherab_opts.get("stark_spectral_bins", 50)
        ff_fb = cherab_opts.get("ff_fb_emission", False)
        ff_fb_bins = cherab_opts.get("ff_fb_spectral_bins", 20)
        mol_exc_emission = cherab_opts.get("mol_exc_emission", False)
        mol_exc_emission_bands = cherab_opts.get("mol_exc_emission_bands", None)

        

        diag_def = get_JETdefs().diag_dict
        transitions = [(int(v[0]), int(v[1])) for _, v in spec_line_dict['1']['1'].items()]
        
        # === Create insrument LOS database ===
        instrument_los_dict = {}
        for diag in diag_list:
            los_points = []
            p1 = diag_def[diag]["p1"][0].tolist()
            w1 = 0.0
            w2 = diag_def[diag]["w"][0][1]
            for p2 in diag_def[diag]["p2"]:
                los_points.append((p1, p2.tolist(), w1, w2))
            instrument_los_dict[diag] = los_points

        # === Initialize Plasma ===
        plasma = CherabPlasma(self, 
                            include_reflections=include_reflections,
                            import_jet_surfaces=import_jet_surfaces,
                            data_source=data_source,
                            recalc_h2_pos=recalc_h2_pos,
                            transitions=transitions,
                            instrument_los_dict = instrument_los_dict,
                            mol_exc_bands= mol_exc_emission_bands)
        
        # === Setup Observers ===
        plasma.setup_observers(instrument_los_dict, pixel_samples=pixel_samples, num_processes = num_processes)

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
                                                spectral_bins= stark_bins,
                                                spectral_rays= 1,
                                                num_processes = num_processes)

        if ff_fb:
            plasma.setup_spectral_observers(instrument_los_dict,
                                            min_wavelength_nm=300,
                                            max_wavelength_nm=500,
                                            destination="continuum",
                                            pixel_samples=pixel_samples,
                                            spectral_bins= ff_fb_bins,
                                            spectral_rays= 1,
                                            num_processes = num_processes)

        self.outdict = {"description": f"CHERAB, REFLECTIONS: {include_reflections}, JET-MESH: {import_jet_surfaces}, DATA SOURCE: {data_source}"}
        

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
                if data_source in ["ADAS", "AMJUEL"]:
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

                if data_source == "YACORA":
                    logger.info("YACORA - ATOMIC")
                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                            include_excitation=True, data_source=data_source)
                    excit = plasma.integrate_instrument(diag)
                    self.outdict[diag][wavelength]["excit"] = [x[0] for x in excit]
                    logger.info("YACORA - MOLECULAR")
                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                            include_H2=True, data_source=data_source)
                    self.outdict[diag][wavelength]["h2"] = [x[0] for x in plasma.integrate_instrument(diag)]
                    
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
                    "wave": wl[0],
                    "intensity": spec,
                    "units": "nm, ph s^-1 m^-2 sr^-1 nm^-1"
                }
            # === Optional molecular band emission ===
            if mol_exc_emission:
                for band in mol_exc_emission_bands:
                    logger.info(f"Molecular Excitation Emission for {band} band")
                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=band, data_source=data_source, include_mol_exc = True)
                    self.outdict[diag][band] = [x[0] for x in plasma.integrate_instrument(diag)]

    def run_cone_integration(self):
        """
        Do a cone integral over the LOS of the diagnostic instruments
        """
        # Populate the H emission cache of the cells
        self.calc_H_emiss()
        # Only ADAS contains the total radiated power data
        if self.data_source == "ADAS":
            self.calc_H_rad_power()
        # Populate the ff-fb cache of the cells
        self.calc_ff_fb_emiss()

        if self.diag_list:
            logger.info(f"       diag_list: {self.diag_list}")
            for key in self.diag_list:
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
                        if self.calc_synth_spec_features:
                            # Derived ne, te require information along LOS, calc emission again using _2 functions
                            if self.data_source == "AMJUEL":
                                chord.calc_int_and_1d_los_quantities_AMJUEL_2()
                            else:
                                chord.calc_int_and_1d_los_quantities_2()
                            logger.info(f"       Calculating synthetic spectra for diag:  {key}")
                            chord.calc_int_and_1d_los_synth_spectra()

        self.gen_synth_diag_data()

    def __getstate__(self):
        """
            For removing the large ADAS_dict from the object for pickling
            See: https://docs/python/org/2/library/pickle.html#example
        """
        odict = self.__dict__.copy() # copy the dict since we change it
        if self.data_source == "ADAS":
            del odict['ADAS_dict']
        return odict

    def __setstate__(self, dict):
        # TODO: Read external ADAS_dict object and add to dict for unpickling
        self.__dict__.update(dict)

    def gen_synth_diag_data(self):
        """
        Generate synthetic diagnostic data in CHERAB-style format with:
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

        self.outdict = outdict

    def calc_H_emiss(self):
        '''
        Calculate the hydrogenic emission for spectral lines defined in the input

        If data source is AMJUEL, calculate the emission with contributions from molecules and H-
        Otherwise used ADAS rates for contributions from el-impact excitation and recombination.
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

    def analyse_synth_spectra(self, res_dict, stark_ne = True, cont_Te = True, line_int_part_bal = False, delL_atomden = False):

        sion_H_transition = self.input_dict['run_options'].get('Sion_H_transition',[[2,1],[3,2]])
        srec_H_transition = self.input_dict['run_options'].get('Srec_H_transition',[[5,2]])

        # Estimate parameters and update res_dict. Call order matters since ne and Te
        # are needed as constraints
        #
        # Electron density estimate from Stark broadening of H6-2 line
        if stark_ne:
            recover_line_int_Stark_ne(res_dict)

        # Electron temperature estimate from ff+fb continuum
        if cont_Te:
            recover_line_int_ff_fb_Te(res_dict)

        # Recombination and Ionization
        if line_int_part_bal:
            recover_line_int_particle_bal(self, res_dict, sion_H_transition=sion_H_transition,
                                           srec_H_transition=srec_H_transition, ne_scal=1.0,
                                           cherab_ne_Te_KT3_resfile=self.cherab_ne_Te_KT3_resfile)

        # delL * neutral density assuming excitation dominated
        if delL_atomden:
            recover_delL_atomden_product(self, res_dict, sion_H_transition=sion_H_transition)

        
    
    
if __name__=='__main__':
    print('To run PESDT, use the "PESDT_run.py" script in the root of PESDT')
