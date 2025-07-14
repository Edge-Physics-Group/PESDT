import numpy as np
import json
# http://lmfit.github.io/lmfit-py/parameters.html
from lmfit import minimize, Parameters, fit_report
from pyADASread import continuo_read
import logging
logger = logging.getLogger(__name__)


"""
Contains analysis functions to estimate plasma parameters from the observed synthetic emission

"""
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


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

def recover_line_int_ff_fb_Te(res_dict):
    """
    Recover line-averaged electron temperature from FF-FB continuum spectra,
    using Balmer continuum intensity ratios. 
    """
    # Reference FF+FB ratio functions
    cont_ratio_360_400 = continuo_read.get_fffb_intensity_ratio_fn_T(360.0, 400.0, 1.0, save_output=True, restore=False)
    cont_ratio_300_360 = continuo_read.get_fffb_intensity_ratio_fn_T(300.0, 360.0, 1.0, save_output=True, restore=False)
    cont_ratio_400_500 = continuo_read.get_fffb_intensity_ratio_fn_T(400.0, 500.0, 1.0, save_output=True, restore=False)
    
    # Iterate over diagnostics, e.g. diag_key == "KT3A"
    for diag_key, diag_data in res_dict.items():
        if diag_key != "description":
            # Initialize result dict
            te_fffb = {"fit_te_360_400": [],
                        "fit_te_300_360": [],
                        "fit_te_400_500": [],
                        "delL_360_400": [],
                        "units": "eV"}
        
            # Repeat for each LOS in diagnostic
            for i, _ in enumerate(diag_data["chord"]):
                logger.info(f"Fitting ff+fb continuum spectra, LOS id = {diag_key}, chord index = {i}")

                wave_fffb = np.asarray(diag_data.get("ff_fb_continuum", {}).get("wave", []))
                synth_data_fffb = np.asarray(diag_data.get("ff_fb_continuum", {}).get("intensity", [])[i])
                if len(wave_fffb) == 0 or len(synth_data_fffb) == 0:
                    logger.warning(f"No spectrum found for {diag_key} chord {i}, skipping.")
                    continue

                try:
                    idx_300, _ = find_nearest(wave_fffb, 300.0)
                    idx_360, _ = find_nearest(wave_fffb, 360.0)
                    idx_400, _ = find_nearest(wave_fffb, 400.0)
                    idx_500, _ = find_nearest(wave_fffb, 500.0)

                    ratio_360_400 = synth_data_fffb[idx_360] / synth_data_fffb[idx_400]
                    ratio_300_360 = synth_data_fffb[idx_300] / synth_data_fffb[idx_360]
                    ratio_400_500 = synth_data_fffb[idx_400] / synth_data_fffb[idx_500]

                    # Match to precomputed tables
                    i360_400, _ = find_nearest(cont_ratio_360_400[:, 1], ratio_360_400)
                    i300_360, _ = find_nearest(cont_ratio_300_360[:, 1], ratio_300_360)
                    i400_500, _ = find_nearest(cont_ratio_400_500[:, 1], ratio_400_500)

                    fit_te_360_400 = cont_ratio_360_400[i360_400, 0]
                    fit_te_300_360 = cont_ratio_300_360[i300_360, 0]
                    fit_te_400_500 = cont_ratio_400_500[i400_500, 0]

                    # Store results in the chord dictionary
                    te_fffb["fit_te_360_400"].append(fit_te_360_400)
                    te_fffb["fit_te_300_360"].append(fit_te_300_360)
                    te_fffb["fit_te_400_500"].append(fit_te_400_500)

                    # Attempt delL fitting if ne is available
                    stark_fit = diag_data.get("stark", {}).get("fit", {})
                    if "ne" in stark_fit:
                        fit_ne = stark_fit["ne"][i]

                        params = Parameters()
                        params.add("delL", value=0.5, min=0.0001, max=10.0)
                        params.add("te_360_400", value=fit_te_360_400, vary=False)
                        params.add("ne", value=fit_ne, vary=False)

                        fit_result = minimize(
                            residual_continuo,
                            params,
                            args=(wave_fffb, synth_data_fffb),
                            method='leastsq'
                        )

                        fit_report(fit_result)
                        vals = fit_result.params.valuesdict()
                        te_fffb["delL_360_400"].append(vals["delL"])

                except Exception as e:
                    logger.warning(f"Error processing {diag_key} chord {i}: {e}")
                    continue
            # Add fit results
            diag_data["ff_fb_continuum"]["fit"] = te_fffb


def recover_line_int_Stark_ne(res_dict, data_source="AMJUEL"):
    """
    Recover line-averaged electron density from H6-2 Stark broadened spectra.
    
    """
    mmm_coeff = {'6t2': {'C': 3.954E-16, 'a': 0.7149, 'b': 0.028}}
    

    for diag_key, diag_data in res_dict.items():
        if diag_key != "description":
            num_chords = len(diag_data["chord"])
            wave_stark = np.asarray(diag_data["stark"]["wave"])
            synth_data_stark = diag_data["stark"]["intensity"]
            wl = diag_data["stark"]["wavelength"]
            cwl = diag_data["stark"]["cwl"]
            # Fit parameters
            params = Parameters()
            params.add("cwl", value=cwl)

            # Determine area based on emission source
            H_emiss = diag_data[wl]
            if data_source == "AMJUEL":
                area_val = np.asarray(H_emiss.get("excit", np.zeros(num_chords)))+ np.asarray(H_emiss.get("recom", np.zeros(num_chords)))+ np.asarray(H_emiss.get("h2", np.zeros(num_chords)))+ np.asarray(H_emiss.get("h2+", np.zeros(num_chords))) + np.asarray(H_emiss.get("h3+", np.zeros(num_chords))) + np.asarray(H_emiss.get("h-", np.zeros(num_chords)))
            else:
                area_val = np.asarray(H_emiss.get("excit", np.zeros(num_chords))) + np.asarray(H_emiss.get("recom", np.zeros(num_chords)))

            # Create results dict
            res = {
                "ne": [],
                "units": "m^-3"
            }
            # calculate for each chord
            for i, _ in enumerate(diag_data["chord"]):

                logger.info(f"Fitting Stark broadened H6-2 spectra, LOS id = {diag_key}, chord index = {i}")
        
                try:
                    
                    params.add("area", value=float(area_val[i]))
                    params.add("stark_fwhm", value=0.15, min=0.0001, max=10.0)

                    params["cwl"].vary = True
                    params["area"].vary = True

                    # Perform fit
                    fit_result = minimize(
                        residual_lorentz_52,
                        params,
                        args=(wave_stark,),
                        kws={"data": np.asarray(synth_data_stark[i])},
                        method="leastsq"
                    )

                    fit_report(fit_result)

                    vals = fit_result.params.valuesdict()

                    # Calculate ne assuming Te = 1 eV
                    fit_ne = (vals["stark_fwhm"] / mmm_coeff["6t2"]["C"])**(1.0 / mmm_coeff["6t2"]["a"])

                    # Store results
                    res["ne"].append(fit_ne)

                except Exception as e:
                    logger.warning(f"Stark fit failed for {diag_key} chord {i}, error: {e}")
                    continue     
            diag_data["stark"]["fit"] = res  

def recover_line_int_particle_bal(sim, res_dict, sion_H_transition=[[2, 1], [3, 2]],
                                    srec_H_transition=[[7, 2]], ne_scal=1.0,
                                    cherab_ne_Te_KT3_resfile=None):
    """
    Estimate recombination/ionisation rates using ADF11 ACD, SCD coefficients.
    DATA STRUCTURE HAS CHANGED; FUNCTION NEEDS TO BE ADJUSTED
    parameter "sim" is the Process edge
    """
    # Choose appropriate ADAS datasets
    if sim.ADAS_dict_lytrap:
        ADAS_dict_local = sim.ADAS_dict_lytrap
        spec_line_dict_key = "spec_line_dict_lytrap"
    else:
        ADAS_dict_local = sim.ADAS_dict
        spec_line_dict_key = "spec_line_dict"

    # Load CHERAB KT3 override results if needed
    if cherab_ne_Te_KT3_resfile:
        with open(cherab_ne_Te_KT3_resfile, "r") as f:
            cherab_res_dict = json.load(f)

    for diag_key, diag_data in res_dict.items():
        if diag_key != "description":
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

                idxTe, Te_val = find_nearest(sim.ADAS_dict['adf11']['1'].Te_arr, fit_Te)
                idxne, ne_val = find_nearest(sim.ADAS_dict['adf11']['1'].ne_arr, fit_ne * 1e-6)

                # --- RECOMBINATION ---
                for tran in srec_H_transition:
                    for H_line_key, info in chord_data.get("spec_line_dict", {}).get("1", {}).get("1", {}).items():
                        if info[0] == str(tran[0]) and info[1] == str(tran[1]):
                            hij = chord_data["los_1d"]["H_emiss"][H_line_key]["excit"] + \
                                chord_data["los_1d"]["H_emiss"][H_line_key]["recom"]
                            pec = sim.ADAS_dict["adf15"]["1"]["1"][H_line_key + "recom"].pec[idxTe, idxne]
                            acd = ADAS_dict_local["adf11"]["1"].acd[idxTe, idxne]
                            srec = 1e-4 * area_cm2 * hij * 4.0 * np.pi * acd / pec

                            tran_str = f"H{tran[0]}{tran[1]}"
                            chord_data.setdefault("los_1d", {}).setdefault("adf11_fit", {})[tran_str] = {
                                "Srec": srec, "units": "s^-1"
                            }

                            # Optional: Ly-trap optically thin comparison
                            if sim.ADAS_dict_lytrap:
                                _acd = sim.ADAS_dict["adf11"]["1"].acd[idxTe, idxne]
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

                            if sim.ADAS_dict_lytrap:
                                _scd = sim.ADAS_dict["adf11"]["1"].scd[idxTe, idxne]
                                _sion = 1e-4 * area_cm2 * h_intensity * 4.0 * np.pi * _scd / pec
                                chord_data.setdefault("los_1d", {}).setdefault("adf11_fit_opt_thin", {})[tran_str] = {
                                    "Sion": _sion, "units": "s^-1"
                                }

def recover_delL_atomden_product(sim, res_dict, sion_H_transition=[[2, 1], [3, 2]], excit_only=True):
    """
    DATA STRUCTURE HAS CHANGED; FUNCTION NEEDS TO BE ADJUSTED
    Estimate delL * atomic density product from H-line intensity assuming excitation-dominated emission.

    Parameters:
    -----------
    sion_H_transition : list
        List of [upper, lower] quantum number transitions (e.g., [[2,1]] for Ly-alpha).
    excit_only : bool
        If True, only excitation contribution is used (for comparison with experimental measurements).
    """

    # Choose appropriate ADAS dataset (Ly-trapped or not)
    ADAS_dict_local = sim.ADAS_dict_lytrap if sim.ADAS_dict_lytrap else sim.ADAS_dict
    spec_line_dict_key = 'spec_line_dict_lytrap' if sim.ADAS_dict_lytrap else 'spec_line_dict'

    for diag_key, diag_data in res_dict.items():
        if diag_key != "description":
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
                            idxTe, Te_val = find_nearest(recom_data.Te_arr, fit_Te)
                            idxne, ne_val = find_nearest(recom_data.ne_arr, fit_ne * 1e-6)

                            # Calculate n0·ΔL [m^-2]
                            n0delL = 4. * np.pi * 1e-4 * hij / (pec_data.pec[idxTe, idxne] * ne_val)
                            n0delL *= 1e6 * 1e-2  # convert to m^-2

                            # Store result
                            los_int.setdefault('n0delL_fit', {})[tran_str] = {'n0delL': n0delL, 'units': 'm^-2'}

                            # Also calculate "optically thin" result if Ly-trapped data used
                            if sim.ADAS_dict_lytrap:
                                pec_thin = sim.ADAS_dict['adf15']['1']['1'][H_line_key + 'excit']
                                recom_thin = sim.ADAS_dict['adf15']['1']['1'][H_line_key + 'recom']
                                idxTe, _ = find_nearest(recom_thin.Te_arr, fit_Te)
                                idxne, ne_val = find_nearest(recom_thin.ne_arr, fit_ne * 1e-6)

                                n0delL_thin = 4. * np.pi * 1e-4 * hij / (pec_thin.pec[idxTe, idxne] * ne_val)
                                n0delL_thin *= 1e6 * 1e-2

                                los_int.setdefault('n0delL_fit_thin', {})[tran_str] = {
                                    'n0delL': n0delL_thin,
                                    'units': 'm^-2'
                                }