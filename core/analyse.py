
import numpy as np
# import scipy.io as io
import os, errno
import json
# http://lmfit.github.io/lmfit-py/parameters.html
from lmfit import minimize, Parameters, fit_report


from .process import ProcessEdgeSim
from pyADASread import adas_adf11_read, adas_adf15_read, continuo_read
from .utils import get_ADAS_dict

import logging
logger = logging.getLogger(__name__)

class AnalyseSynthDiag(ProcessEdgeSim):
    """
        Inherits from ProcessEdgeSim and adds methods for analysis of synthetic spectra
    """
    def __init__(self, input_dict):
        self.input_dict = input_dict

        tmpstr = input_dict['edge_code']['sim_path'].replace('/','_')
        logger.info(f"{input_dict['edge_code']['sim_path']}")
        if tmpstr[:3] == '_u_':
            tmpstr = tmpstr[3:]
        elif tmpstr[:6] == '_work_':
            tmpstr = tmpstr[6:]
        else:
            tmpstr = tmpstr[1:]

        self.savedir = input_dict['save_dir'] + tmpstr + '/'

        # Create dir from tran file, if it does not exist
        try:
            os.makedirs(self.savedir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.data2d_save_file = self.savedir +'PESDT.2ddata.pkl'
        self.synth_diag_save_file = self.savedir + 'PESDT.synth_diag.json'
        self.spec_line_dict = input_dict['spec_line_dict']

        # Option to run cherab
        self.run_cherab = input_dict.get('run_cherab', False)

        # Option to use cherab ne and Te fits rather than pyproc's. Use case - Lyman opacity adas data is not suppored
        # by cherab-bridge, so import cherab plasma parameter profiles with reflections impact here instead and apply
        # to Siz and Srec estimates with modified Ly-trapping adas data
        self.cherab_ne_Te_KT3_resfile = None
        if 'use_cherab_resfile_for_KT3_ne_Te_fits' in input_dict['run_options']:
            self.cherab_ne_Te_KT3_resfile = self.savedir + input_dict['run_options']['use_cherab_resfile_for_KT3_ne_Te_fits']

        # Location of adf15 and adf11 ADAS data modified for Ly-series opacity with escape factor method
        if 'read_ADAS_lytrap' in input_dict:
            self.adas_lytrap = input_dict['read_ADAS_lytrap']
            self.spec_line_dict_lytrap = self.adas_lytrap['spec_line_dict']
            self.ADAS_dict_lytrap = get_ADAS_dict(self.savedir,
                                                  self.spec_line_dict_lytrap,
                                                  restore=not input_dict['read_ADAS_lytrap']['read'],
                                                  adf11_year = self.adas_lytrap['adf11_year'],
                                                  lytrap_adf11_dir=self.adas_lytrap['adf11_dir'],
                                                  lytrap_pec_file=self.adas_lytrap['pec_file'])
        else:
            self.ADAS_dict_lytrap = None
            self.spec_line_dict_lytrap = None

        # Also get standard ADAS data
        self.ADAS_dict = get_ADAS_dict(input_dict['save_dir'],
                                       self.spec_line_dict, adf11_year=12, restore=not input_dict['read_ADAS'])

        # Check for any outlier cells to be 
        outlier_cell_dict = input_dict.get("interp_outlier_cell", None)

        logger.info(f"diag_list: {input_dict['diag_list']}")
        super().__init__(self.ADAS_dict, 
                         input_dict['edge_code'], 
                         ADAS_dict_lytrap = self.ADAS_dict_lytrap, 
                         machine=input_dict['machine'],
                         pulse=input_dict['pulse'], 
                         outlier_cell_dict=outlier_cell_dict,
                         interactive_plots = input_dict['interactive_plots'],
                         spec_line_dict=self.spec_line_dict,
                         spec_line_dict_lytrap = self.spec_line_dict_lytrap,
                         diag_list=input_dict['diag_list'],
                         calc_synth_spec_features=input_dict['run_options']['calc_synth_spec_features'],
                         save_synth_diag=True,
                         synth_diag_save_file=self.synth_diag_save_file,
                         data2d_save_file=self.data2d_save_file,
                         data_source = input_dict['run_options'].get('data_source', "AMJUEL"),
                         AMJUEL_date = input_dict['run_options'].get('AMJUEL_date', 2016), # Default to <2017 (no H3+)
                         recalc_h2_pos = input_dict['run_options'].get('recalc_h2_pos', True),# When set to true H2+ density is recalculated with AMJUEL H.12 2.0c
                         run_cherab = self.run_cherab, 
                         input_dict = self.input_dict)   

        if self.input_dict['run_options']['analyse_synth_spec_features']:
            # Read synth diag saved data
            try:
                with open(self.synth_diag_save_file, 'r') as f:
                    res_dict = json.load(f)
                self.analyse_synth_spectra(res_dict)
            except IOError as e:
                raise
        
        if self.run_cherab:
            if input_dict['cherab_options'].get('analyse_synth_spec_features', False):
                try:
                    self.recover_line_int_Stark_ne(self.outdict)
                    if input_dict['cherab_options'].get('ff_fb_emission', False):
                        self.recover_line_int_ff_fb_Te(self.outdict)
                except:
                    # SafeGuard for possible issues, so that not all comp. time is lost 
                    logger.info('Something went wrong with AnalyseSynthDiag')
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


    # Analyse synthetic spectra
    def analyse_synth_spectra(self, res_dict, stark_ne = True, cont_Te = True, line_int_part_bal = False, delL_atomden = False):

        sion_H_transition = self.input_dict['run_options'].get('Sion_H_transition',[[2,1],[3,2]])
        srec_H_transition = self.input_dict['run_options'].get('Srec_H_transition',[[5,2]])

        # Estimate parameters and update res_dict. Call order matters since ne and Te
        # are needed as constraints
        #
        # Electron density estimate from Stark broadening of H6-2 line
        if stark_ne:
            self.recover_line_int_Stark_ne(res_dict)

        # Electron temperature estimate from ff+fb continuum
        if cont_Te:
            self.recover_line_int_ff_fb_Te(res_dict)

        # Recombination and Ionization
        if line_int_part_bal:
            self.recover_line_int_particle_bal(res_dict, sion_H_transition=sion_H_transition,
                                           srec_H_transition=srec_H_transition, ne_scal=1.0,
                                           cherab_ne_Te_KT3_resfile=self.cherab_ne_Te_KT3_resfile)

        # delL * neutral density assuming excitation dominated
        if delL_atomden:
            self.recover_delL_atomden_product(res_dict, sion_H_transition=sion_H_transition)

        with open(self.synth_diag_save_file, mode='w', encoding='utf-8') as f:
                json.dump(res_dict, f, indent=2)

    @staticmethod
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    @staticmethod
    def residual_lorentz_52(params, x, data=None, eps_data=None):

        cwl = params['cwl'].value
        area = params['area'].value
        stark_fwhm = params['stark_fwhm'].value

        model = 1. / (np.power(np.abs(x - cwl), 5. / 2.) + np.power(stark_fwhm / 2.0, 5. / 2.))

        model_area = np.trapz(model, x=x)
        amp_scal = area / model_area
        model *= amp_scal

        if data is None:
            return model
        if eps_data is None:
            return (model - data)
        return (model - data) / eps_data

    @staticmethod
    def residual_continuo(params, wave, data=None, eps_data=None):
        delL = params['delL'].value
        te = params['te_360_400'].value
        ne = params['ne'].value

        model_ff, model_ff_fb = continuo_read.adas_continuo_py(wave, te, 1, 1)
        model_ff = model_ff * ne * ne * delL
        model_ff_fb = model_ff_fb * ne * ne * delL

        if data is None:
            return model_ff_fb
        if eps_data is None:
            return (model_ff_fb - data)
        return (model_ff_fb - data) / eps_data

    
    @staticmethod
    def recover_line_int_ff_fb_Te(res_dict):
        """
        Recover line-averaged electron temperature from FF-FB continuum spectra,
        using Balmer continuum intensity ratios. 
        """
        # Reference FF+FB ratio functions
        cont_ratio_360_400 = continuo_read.get_fffb_intensity_ratio_fn_T(360.0, 400.0, 1.0, save_output=True, restore=False)
        cont_ratio_300_360 = continuo_read.get_fffb_intensity_ratio_fn_T(300.0, 360.0, 1.0, save_output=True, restore=False)
        cont_ratio_400_500 = continuo_read.get_fffb_intensity_ratio_fn_T(400.0, 500.0, 1.0, save_output=True, restore=False)

        for diag_key, diag_data in res_dict.items():
            for i, chord_data in enumerate(diag_data["chord"]):
                logger.info(f"Fitting ff+fb continuum spectra, LOS id = {diag_key}, chord index = {i}")

                wave_fffb = np.asarray(chord_data.get("wave", []))
                synth_data_fffb = np.asarray(chord_data.get("intensity", []))
                if len(wave_fffb) == 0 or len(synth_data_fffb) == 0:
                    logger.warning(f"No spectrum found for {diag_key} chord {i}, skipping.")
                    continue

                try:
                    idx_300, _ = AnalyseSynthDiag.find_nearest(wave_fffb, 300.0)
                    idx_360, _ = AnalyseSynthDiag.find_nearest(wave_fffb, 360.0)
                    idx_400, _ = AnalyseSynthDiag.find_nearest(wave_fffb, 400.0)
                    idx_500, _ = AnalyseSynthDiag.find_nearest(wave_fffb, 500.0)

                    ratio_360_400 = synth_data_fffb[idx_360] / synth_data_fffb[idx_400]
                    ratio_300_360 = synth_data_fffb[idx_300] / synth_data_fffb[idx_360]
                    ratio_400_500 = synth_data_fffb[idx_400] / synth_data_fffb[idx_500]

                    # Match to precomputed tables
                    i360_400, _ = AnalyseSynthDiag.find_nearest(cont_ratio_360_400[:, 1], ratio_360_400)
                    i300_360, _ = AnalyseSynthDiag.find_nearest(cont_ratio_300_360[:, 1], ratio_300_360)
                    i400_500, _ = AnalyseSynthDiag.find_nearest(cont_ratio_400_500[:, 1], ratio_400_500)

                    fit_te_360_400 = cont_ratio_360_400[i360_400, 0]
                    fit_te_300_360 = cont_ratio_300_360[i300_360, 0]
                    fit_te_400_500 = cont_ratio_400_500[i400_500, 0]

                    # Store results
                    chord_data.setdefault("fit", {})
                    chord_data["fit"]["te_fffb"] = {
                        "fit_te_360_400": fit_te_360_400,
                        "fit_te_300_360": fit_te_300_360,
                        "fit_te_400_500": fit_te_400_500,
                        "units": "eV"
                    }

                    # Attempt delL fitting if ne is available
                    stark_fit = chord_data.get("los_1d", {}).get("stark", {}).get("fit", {})
                    if "ne" in stark_fit:
                        fit_ne = stark_fit["ne"]

                        params = Parameters()
                        params.add("delL", value=0.5, min=0.0001, max=10.0)
                        params.add("te_360_400", value=fit_te_360_400, vary=False)
                        params.add("ne", value=fit_ne, vary=False)

                        fit_result = minimize(
                            AnalyseSynthDiag.residual_continuo,
                            params,
                            args=(wave_fffb, synth_data_fffb),
                            method='leastsq'
                        )

                        fit_report(fit_result)
                        vals = fit_result.params.valuesdict()

                        chord_data["fit"]["te_fffb"]["delL_360_400"] = vals["delL"]

                except Exception as e:
                    logger.warning(f"Error processing {diag_key} chord {i}: {e}")
                    continue
    
    @staticmethod
    def recover_line_int_Stark_ne(res_dict, data_source="AMJUEL"):
        """
        Recover line-averaged electron density from H6-2 Stark broadened spectra.
        
        """
        mmm_coeff = {'6t2': {'C': 3.954E-16, 'a': 0.7149, 'b': 0.028}}
        print(res_dict)

        for diag_key, diag_data in res_dict.items():
            for i, chord_data in enumerate(diag_data["chord"]):

                logger.info(f"Fitting Stark broadened H6-2 spectra, LOS id = {diag_key}, chord index = {i}")

                spec_dict = chord_data.get("spec_line_dict", {}).get("1", {}).get("1", {})
                print(spec_dict)
                for H_line_key, val in spec_dict.items():
                    if H_line_key.startswith("6") and H_line_key[1] == "2":

                        try:
                            wave_stark = np.asarray(chord_data["los_1d"]["stark"]["wave"])
                            synth_data_stark = np.asarray(chord_data["los_1d"]["stark"]["intensity"])

                            # Fit parameters
                            params = Parameters()
                            params.add("cwl", value=float(H_line_key) / 10.0)

                            # Determine area based on emission source
                            H_emiss = chord_data.get("los_1d", {}).get("H_emiss", {}).get(H_line_key, {})
                            if data_source == "AMJUEL":
                                area_val = sum([
                                    H_emiss.get("excit", 0.0),
                                    H_emiss.get("recom", 0.0),
                                    H_emiss.get("h2", 0.0),
                                    H_emiss.get("h2+", 0.0),
                                    H_emiss.get("h-", 0.0),
                                ])
                            else:
                                area_val = H_emiss.get("excit", 0.0) + H_emiss.get("recom", 0.0)

                            params.add("area", value=float(area_val))
                            params.add("stark_fwhm", value=0.15, min=0.0001, max=10.0)

                            params["cwl"].vary = True
                            params["area"].vary = True

                            # Perform fit
                            fit_result = minimize(
                                AnalyseSynthDiag.residual_lorentz_52,
                                params,
                                args=(wave_stark,),
                                kws={"data": synth_data_stark},
                                method="leastsq"
                            )

                            fit_report(fit_result)

                            vals = fit_result.params.valuesdict()

                            # Calculate ne assuming Te = 1 eV
                            fit_ne = (vals["stark_fwhm"] / mmm_coeff["6t2"]["C"])**(1.0 / mmm_coeff["6t2"]["a"])

                            # Store results
                            chord_data.setdefault("los_1d", {}).setdefault("stark", {}).setdefault("fit", {})
                            chord_data["los_1d"]["stark"]["fit"]["ne"] = fit_ne
                            chord_data["los_1d"]["stark"]["fit"]["units"] = "m^-3"

                        except Exception as e:
                            logger.warning(f"Stark fit failed for {diag_key} chord {i}, H_line={H_line_key}: {e}")
                            continue       

    def recover_line_int_particle_bal(self, res_dict, sion_H_transition=[[2, 1], [3, 2]],
                                    srec_H_transition=[[7, 2]], ne_scal=1.0,
                                    cherab_ne_Te_KT3_resfile=None):
        """
        Estimate recombination/ionisation rates using ADF11 ACD, SCD coefficients.
        CHERAB-compatible version using res_dict[diag_key]["chord"][i] structure.
        """
        # Choose appropriate ADAS datasets
        if self.ADAS_dict_lytrap:
            ADAS_dict_local = self.ADAS_dict_lytrap
            spec_line_dict_key = "spec_line_dict_lytrap"
        else:
            ADAS_dict_local = self.ADAS_dict
            spec_line_dict_key = "spec_line_dict"

        # Load CHERAB KT3 override results if needed
        if cherab_ne_Te_KT3_resfile:
            with open(cherab_ne_Te_KT3_resfile, "r") as f:
                cherab_res_dict = json.load(f)

        for diag_key, diag_data in res_dict.items():
            for i, chord_data in enumerate(diag_data["chord"]):

                try:
                    # Get Te and ne either from CHERAB override or local results
                    if cherab_ne_Te_KT3_resfile and diag_key == "KT3":
                        cherab_chord_data = cherab_res_dict[diag_key]["chord"][i]
                        fit_ne = ne_scal * cherab_chord_data["los_1d"]["stark"]["fit"]["ne"]
                        fit_Te = cherab_chord_data["los_1d"]["ff_fb_continuum"]["fit"]["fit_te_360_400"]
                    else:
                        fit_ne = ne_scal * chord_data["los_1d"]["stark"]["fit"]["ne"]
                        fit_Te = chord_data["los_1d"]["ff_fb_continuum"]["fit"]["fit_te_360_400"]
                except Exception as e:
                    logger.warning(f"Skipping chord {i} in {diag_key} due to missing ne/Te: {e}")
                    continue

                logger.info(f"Ionization/recombination, LOS id = {diag_key}, chord index = {i}")

                w2unmod = chord_data["geom"]["w2"]
                R = chord_data["geom"]["p2"][0]
                area_cm2 = 1.0e04 * 2.0 * np.pi * w2unmod * R

                idxTe, Te_val = self.find_nearest(self.ADAS_dict['adf11']['1'].Te_arr, fit_Te)
                idxne, ne_val = self.find_nearest(self.ADAS_dict['adf11']['1'].ne_arr, fit_ne * 1e-6)

                # --- RECOMBINATION ---
                for tran in srec_H_transition:
                    for H_line_key, info in chord_data.get("spec_line_dict", {}).get("1", {}).get("1", {}).items():
                        if info[0] == str(tran[0]) and info[1] == str(tran[1]):
                            hij = chord_data["los_1d"]["H_emiss"][H_line_key]["excit"] + \
                                chord_data["los_1d"]["H_emiss"][H_line_key]["recom"]
                            pec = self.ADAS_dict["adf15"]["1"]["1"][H_line_key + "recom"].pec[idxTe, idxne]
                            acd = ADAS_dict_local["adf11"]["1"].acd[idxTe, idxne]
                            srec = 1e-4 * area_cm2 * hij * 4.0 * np.pi * acd / pec

                            tran_str = f"H{tran[0]}{tran[1]}"
                            chord_data.setdefault("los_1d", {}).setdefault("adf11_fit", {})[tran_str] = {
                                "Srec": srec, "units": "s^-1"
                            }

                            # Optional: Ly-trap optically thin comparison
                            if self.ADAS_dict_lytrap:
                                _acd = self.ADAS_dict["adf11"]["1"].acd[idxTe, idxne]
                                _srec = 1e-4 * area_cm2 * hij * 4.0 * np.pi * _acd / pec
                                chord_data.setdefault("los_1d", {}).setdefault("adf11_fit_opt_thin", {})[tran_str] = {
                                    "Srec": _srec, "units": "s^-1"
                                }

                # --- IONIZATION ---
                for tran in sion_H_transition:
                    for H_line_key, info in chord_data.get(spec_line_dict_key, {}).get("1", {}).get("1", {}).items():
                        if info[0] == str(tran[0]) and info[1] == str(tran[1]):
                            h_intensity = chord_data["los_1d"]["H_emiss"][H_line_key]["excit"] + \
                                        chord_data["los_1d"]["H_emiss"][H_line_key]["recom"]
                            pec = ADAS_dict_local["adf15"]["1"]["1"][H_line_key + "excit"].pec[idxTe, idxne]
                            scd = ADAS_dict_local["adf11"]["1"].scd[idxTe, idxne]
                            sion = 1e-4 * area_cm2 * h_intensity * 4.0 * np.pi * scd / pec

                            tran_str = f"H{tran[0]}{tran[1]}"
                            chord_data.setdefault("los_1d", {}).setdefault("adf11_fit", {}).setdefault(tran_str, {})["Sion"] = sion
                            chord_data["los_1d"]["adf11_fit"][tran_str]["units"] = "s^-1"

                            if self.ADAS_dict_lytrap:
                                _scd = self.ADAS_dict["adf11"]["1"].scd[idxTe, idxne]
                                _sion = 1e-4 * area_cm2 * h_intensity * 4.0 * np.pi * _scd / pec
                                chord_data.setdefault("los_1d", {}).setdefault("adf11_fit_opt_thin", {})[tran_str] = {
                                    "Sion": _sion, "units": "s^-1"
                                }

    def recover_delL_atomden_product(self, res_dict, sion_H_transition=[[2, 1], [3, 2]], excit_only=True):
        """
        Estimate delL * atomic density product from H-line intensity assuming excitation-dominated emission.

        Parameters:
        -----------
        sion_H_transition : list
            List of [upper, lower] quantum number transitions (e.g., [[2,1]] for Ly-alpha).
        excit_only : bool
            If True, only excitation contribution is used (for comparison with experimental measurements).
        """

        # Choose appropriate ADAS dataset (Ly-trapped or not)
        ADAS_dict_local = self.ADAS_dict_lytrap if self.ADAS_dict_lytrap else self.ADAS_dict
        spec_line_dict_key = 'spec_line_dict_lytrap' if self.ADAS_dict_lytrap else 'spec_line_dict'

        for diag_key, diag_data in res_dict.items():
            for chord_key, chord_data in diag_data.items():
                los_int = chord_data['los_int']
                try:
                    fit_ne = los_int['stark']['fit']['ne']
                    fit_Te = los_int['ff_fb_continuum']['fit']['fit_te_360_400']
                except KeyError:
                    continue  # Skip if necessary fits are missing

                for tran in sion_H_transition:
                    up, low = map(str, tran)
                    tran_str = f"H{up}{low}"
                    logger.info(f"n0·ΔL from transition {tran_str}, LOS id = {diag_key}, {chord_key}")

                    for H_line_key, (q_up, q_low) in chord_data[spec_line_dict_key]['1']['1'].items():
                        if q_up == up and q_low == low:
                            hij = los_int['H_emiss'][H_line_key]['excit']
                            if not excit_only:
                                hij += los_int['H_emiss'][H_line_key]['recom']

                            # Find nearest ADAS indices
                            pec_data = ADAS_dict_local['adf15']['1']['1'][H_line_key + 'excit']
                            recom_data = ADAS_dict_local['adf15']['1']['1'][H_line_key + 'recom']
                            idxTe, Te_val = self.find_nearest(recom_data.Te_arr, fit_Te)
                            idxne, ne_val = self.find_nearest(recom_data.ne_arr, fit_ne * 1e-6)

                            # Calculate n0·ΔL [m^-2]
                            n0delL = 4. * np.pi * 1e-4 * hij / (pec_data.pec[idxTe, idxne] * ne_val)
                            n0delL *= 1e6 * 1e-2  # convert to m^-2

                            # Store result
                            los_int.setdefault('n0delL_fit', {})[tran_str] = {'n0delL': n0delL, 'units': 'm^-2'}

                            # Also calculate "optically thin" result if Ly-trapped data used
                            if self.ADAS_dict_lytrap:
                                pec_thin = self.ADAS_dict['adf15']['1']['1'][H_line_key + 'excit']
                                recom_thin = self.ADAS_dict['adf15']['1']['1'][H_line_key + 'recom']
                                idxTe, _ = self.find_nearest(recom_thin.Te_arr, fit_Te)
                                idxne, ne_val = self.find_nearest(recom_thin.ne_arr, fit_ne * 1e-6)

                                n0delL_thin = 4. * np.pi * 1e-4 * hij / (pec_thin.pec[idxTe, idxne] * ne_val)
                                n0delL_thin *= 1e6 * 1e-2

                                los_int.setdefault('n0delL_fit_thin', {})[tran_str] = {
                                    'n0delL': n0delL_thin,
                                    'units': 'm^-2'
                                }
    
   