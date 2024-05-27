#import pyximport; pyximport.install(pyimport=True)
import argparse
import numpy as np
import json, os, sys
import pickle

from PESDT.cherab_bridge.cherab_plasma import CherabPlasma
from PESDT.atomic import get_ADAS_dict
from PESDT.analyse import AnalyseSynthDiag
'''
Old run script to run cherab separately

requires a separate .json input file, and the output .json from PESDT

'''

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def read_input_dict(input_dict_str):

    with open(input_dict_str, mode='r', encoding='utf-8') as f:
        # Remove comments
        with open("temp.json", 'w') as wf:
            for line in f.readlines():
                if line[0:2] == '//' or line[0:1] == '#':
                    continue
                wf.write(line)

    with open("temp.json", 'r') as f:
        input_dict = json.load(f)

    os.remove('temp.json')

    return input_dict

if __name__=='__main__':

    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Run cherab_bridge')
    parser.add_argument('cherab_bridge_input_dict')
    args = parser.parse_args()

    # Handle the input arguments
    input_dict_file = args.cherab_bridge_input_dict

    if os.path.isfile(input_dict_file):
        print('Found input dictionary: ', input_dict_file)
        input_dict = read_input_dict(input_dict_file)
    else:
        sys.exit(input_dict_file + ' not found')

    # Handle the input arguments
    PESDT_case = input_dict['save_dir']+input_dict['PESDT_case']

    if os.path.isdir(PESDT_case):
        if os.path.isfile(PESDT_case + '/PESDT.2ddata.pkl'):
            print('Found EDGE2D pickled data: ', PESDT_case + '/PESDT.2ddata.pkl')
            infile = open(PESDT_case + '/PESDT.2ddata.pkl', 'rb')
            edge2d_pkl = pickle.load(infile)
        else:
            sys.exit('PESDT EDGE2D pickle file not found.')
        if os.path.isfile(PESDT_case + '/PESDT.synth_diag.json'):
            print('Found synthetic diagnostic data file: ', PESDT_case + '/PESDT.synth_diag.json')
            # Read synth diag saved data
            try:
                with open(PESDT_case +  '/PESDT.synth_diag.json', 'r') as f:
                    synth_diag_dict = json.load(f)
            except IOError as e:
                raise
        else:
            sys.exit('PESDT synthetic diagnostic data file not found.')

    else:
        sys.exit('PESDT case ' + PESDT_case + ' not found.')

    # Inputs from cherab_bridge_input_dict
    import_jet_surfaces = input_dict['cherab_options']['import_jet_surfaces']
    include_reflections = input_dict['cherab_options']['include_reflections']
    spectral_bins = input_dict['cherab_options']['spectral_bins']
    pixel_samples = input_dict['cherab_options']['pixel_samples']
    spec_line_dict = input_dict['spec_line_dict']
    diag_list = input_dict['diag_list']
    read_ADAS = input_dict['read_ADAS']
    use_AMJUEL = input_dict['cherab_options']['use_AMJUEL']
    recalc_h2_pos = input_dict['cherab_options'].get("recalc_h2_pos", True)
    stark_transition = input_dict['cherab_options'].get('stark_transition', False)
    ff_fb = input_dict['cherab_options'].get('ff_fb_emission', False)
    #sion_H_transition = input_dict['cherab_options']['Sion_H_transition']
    #srec_H_transition = input_dict['cherab_options']['Srec_H_transition']

    # Read ADAS data
    ADAS_dict = get_ADAS_dict(input_dict['save_dir'],
                              spec_line_dict, adf11_year=12, restore=not input_dict['read_ADAS'])

    # Generate cherab plasma
    plasma = CherabPlasma(edge2d_pkl, ADAS_dict, include_reflections = include_reflections,
                          import_jet_surfaces = import_jet_surfaces, use_AMJUEL=use_AMJUEL, recalc_h2_pos = recalc_h2_pos)

    # Create output dict
    outdict = {}

    # Loop through diagnostics, their LOS, integrate over Lyman/Balmer
    multi = 1.0
    for diag_key in diag_list:
        for diag_key_PESDT, val in synth_diag_dict.items():
            if diag_key == diag_key_PESDT:
                outdict[diag_key] = {}
                for diag_chord, val in synth_diag_dict[diag_key].items():
                    los_p1 = val['chord']['p1']
                    los_p2 = val['chord']['p2']
                    print(los_p1)
                    print(los_p2)
                    break
                    los_w1 = val['chord']['w1']
                    los_w2 = val['chord']['w2']
                    H_lines = spec_line_dict['1']['1']

                    outdict[diag_key][diag_chord] = {
                        'chord':{'p1':los_p1, 'p2':los_p2, 'w1':los_w1, 'w2':los_w2}
                    }
                    outdict[diag_key][diag_chord]['spec_line_dict'] = spec_line_dict

                    outdict[diag_key][diag_chord]['los_int'] = {'H_emiss': {}}

                    print(diag_key, los_p2)
                    for H_line_key, val in H_lines.items():
                     
                        transition = (int(val[0]), int(val[1]))
                        wavelength = float(H_line_key)/10. #nm
                        min_wavelength = (wavelength)-1.0
                        max_wavelength = (wavelength)+1.0

                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                   include_excitation=True, include_recombination=False, use_AMJUEL=use_AMJUEL)
                        exc_radiance, wave = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, #, exc_spectrum,
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spectral_bins, pixel_samples=pixel_samples)

                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                   include_excitation=False, include_recombination=True, use_AMJUEL=use_AMJUEL)

                        rec_radiance, wave = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, # rec_spectrum,
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spectral_bins, pixel_samples=pixel_samples)

                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                   include_excitation=False, include_H2=True, use_AMJUEL=use_AMJUEL)

                        h2_radiance, wave = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, # H2_spectrum,
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spectral_bins, pixel_samples=pixel_samples)

                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                   include_excitation=False, include_H2_pos= True, use_AMJUEL=use_AMJUEL)

                        h2_pos_radiance, wave = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, # H2+_spectrum,
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spectral_bins, pixel_samples=pixel_samples)
                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                   include_excitation=False, include_H_neg=True, use_AMJUEL=use_AMJUEL)

                        h_neg_radiance, wave = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, # H-_spectrum,
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spectral_bins, pixel_samples=pixel_samples)
                                                            
                        outdict[diag_key][diag_chord]['los_int']['H_emiss'][H_line_key] = {
                            'excit':(np.array(exc_radiance)*multi).tolist(),
                            'recom':(np.array(rec_radiance)*multi).tolist(),
                            'h2': (np.array(h2_radiance)*multi).tolist(),
                            'h2+': (np.array(h2_pos_radiance)*multi).tolist(),
                            'h-': (np.array(h_neg_radiance)*multi).tolist(),
                            'units':'ph.s^-1.m^-2.sr^-1'
                        }

                        if stark_transition:
                            if transition == tuple(stark_transition):
                                print('Stark transition')
                                plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                           include_excitation=True, include_recombination=True, 
                                                           include_H2_pos= True, include_H2=True, include_H_neg=True, use_AMJUEL=use_AMJUEL,
                                                           include_stark=True)
                                spec_bins = 50
                                radiance,  wave_arr = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, #spectrum,
                                                                                    min_wavelength, max_wavelength,
                                                                                    spectral_bins=spec_bins,
                                                                                    pixel_samples=pixel_samples,
                                                                                    display_progress=False,no_avg = True)

                                outdict[diag_key][diag_chord]['los_int']['stark']={'cwl': wavelength, 'wave': (np.array(wave_arr)).tolist(),
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

                        outdict[diag_key][diag_chord]['los_int']['ff_fb_continuum'] = {
                                'wave': (np.array(wave_arr)).tolist(),
                                'intensity': (np.array(radiance)).tolist(),
                                'units': 'nm, ph s^-1 m^-2 sr^-1 nm^-1'}

    if input_dict['cherab_options'].get('analyse_synth_spec_features', False):
        try:
            AnalyseSynthDiag.recover_line_int_Stark_ne(outdict)
            if ff_fb:
                AnalyseSynthDiag.recover_line_int_ff_fb_Te(outdict)
        except:
            # SafeGuard for possible issues, so that not all comp. time is lost 
            print('Something went wrong with AnalyseSynthDiag')
            pass
    # SAVE IN JSON FORMAT TO ENSURE PYTHON 2/3 COMPATIBILITY
    if include_reflections:
        savefile = PESDT_case + '/cherab_refl.synth_diag.json'
    else:
        savefile = PESDT_case + '/cherab.synth_diag.json'
    with open(savefile, mode='w', encoding='utf-8') as f:
        json.dump(outdict, f, indent=2)

    print('Saving cherab_bridge synthetic diagnostic data to:', savefile)
