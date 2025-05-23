import pickle
import numpy as np
import os
import sys
import contextlib

from pyADASread import adas_adf11_read, adas_adf15_read


@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    https://stackoverflow.com/questions/977840/redirecting-fortran-called-via-f2py-output-in-python

    A context manager to temporarily redirect stdout or stderr.
    ADAS readers flood stdout and stderr with a stream of fortran warnings.

    e.g.:


    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()

def get_ADAS_dict(save_dir, spec_line_dict, num_samples=100, restore=False, lytrap=False,
                  adf11_year=12, lytrap_adf11_dir=False, lytrap_pec_file=False):
    if restore:
        # Try to restore ADAS_dict
        if lytrap:
            try:
                with open(save_dir + 'ADAS_dict_lytrap.pkl', 'rb') as f:
                    ADAS_dict = pickle.load(f)
            except IOError as e:
                print('ADAS dictionary not found. Set [read_ADAS_lytrap] to True.')
                raise
        else:
            try:
                with open(save_dir + 'ADAS_dict.pkl', 'rb') as f:
                    ADAS_dict = pickle.load(f)
            except IOError as e:
                print('ADAS dictionary not found. Set [read_ADAS] to True.')
                raise

        # Does the restored ADAS_dict contain all of the requested lines?
        for atnum, atnumdict in spec_line_dict.items():
            for ionstage, stagedict in atnumdict.items():
                for line, val in stagedict.items():
                    found_line = False
                    for adas_atnum, adas_atnumdict in ADAS_dict['adf15'].items():
                        for adas_ionstage, adas_stagedict in adas_atnumdict.items():
                            if atnum == adas_atnum and ionstage == adas_ionstage:
                                for adas_line, val in adas_stagedict.items():
                                    if line == adas_line[:-5]:  # strip 'recom', 'excit'
                                        found_line = True
                    if not found_line:
                        print(atnum, ' ', ionstage, ' ', line,
                              ' not found in restored ADAS_dict. Set [read_ADAS] to True and try again.')
                        return
        if lytrap:
            print('ADAS Ly trapping dictionary restored.')
        else:
            print('ADAS dictionary restored.')
    else:
        with stdchannel_redirected(sys.stderr, os.devnull):
            with stdchannel_redirected(sys.stdout, os.devnull):
                # Read all necessary ADAS data here and store in dict
                ADAS_dict = {}
                Te_rnge = [0.05, 5000]
                ne_rnge = [1.0e10, 1.0e16]
                num_samples = 100
                line_blocks =adas_adf15_read.get_adf15_input_dict(os.path.expanduser('~') + '/PESDT/input/adas_adf15_input.json')
                ADAS_dict['adf15'] = adas_adf15_read.get_adas_imp_PECs_interp(spec_line_dict, line_blocks, Te_rnge,
                                                                              ne_rnge, npts=num_samples,
                                                                              npts_interp=1000,
                                                                              lytrap_pec_file=lytrap_pec_file)
                # Also get adf11 for the ionisation balance fractional abundance. No Te_arr, ne_arr interpolation
                # available in the adf11 reader at the moment, so generate more coarse array (sloppy!)
                # TODO: add interpolation capability to the adf11 reader so that adf15 and adf11 are on the same Te, ne grid
                Te_arr_adf11 = np.logspace(np.log10(Te_rnge[0]), np.log10(Te_rnge[1]), 500)
                ne_arr_adf11 = np.logspace(np.log10(ne_rnge[0]), np.log10(ne_rnge[1]), 30)
                adf11_files_dict = adas_adf11_read.get_adf11_input_dict(os.path.expanduser('~') + '/PESDT/input/adas_adf11_input.json')
                ADAS_dict['adf11'] = {}
                for atnum in spec_line_dict:
                    if int(atnum) > 1:
                        # if tint=None, run_adas405, if tint specified, run_adas406 time dependent (ne*tau transport physics)
                        # e.g. to tau=0.1 ms tint=0.0001
                        ADAS_dict['adf11'][atnum] = adas_adf11_read.get_adas_imp_adf11(adf11_files_dict, int(atnum), Te_arr_adf11,
                                                                                       ne_arr_adf11, tint=None)
                    elif int(atnum) == 1:
                        ADAS_dict['adf11'][atnum] = adas_adf11_read.get_adas_H_adf11_interp(Te_rnge, ne_rnge,
                                                                                            npts=num_samples,
                                                                                            npts_interp=1000,
                                                                                            pwr=True,
                                                                                            year=adf11_year,
                                                                                            custom_dir=lytrap_adf11_dir)
                # Pickle ADAS dictionary to save_dir
                if lytrap_adf11_dir:
                    output = open(save_dir + 'ADAS_dict_lytrap.pkl', 'wb')
                else:
                    output = open(save_dir + 'ADAS_dict.pkl', 'wb')
                pickle.dump(ADAS_dict, output)
                output.close()

    return ADAS_dict
