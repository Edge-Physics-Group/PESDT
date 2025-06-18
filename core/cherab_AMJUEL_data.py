

import numpy as np

from cherab.core.atomic import AtomicData
from cherab.PESDT_addon.RateFunctions import RateFunction, NullRateFunction
from .utils import photon_rate_coeffs
from .utils import wavelength as wl


class AMJUEL_Data(AtomicData):
    '''
    Class for passing AMJUEL derived emissivites to Cherab compatible Line<effect>_AM classes

    '''

    def __init__(self, AMJUEL_dir = ''):
        # Populate data dict with AMJUEL reactions as defined in amread.py

        atomic_data_dict = {2: photon_rate_coeffs(2),
                            3: photon_rate_coeffs(3),
                            4: photon_rate_coeffs(4),
                            5: photon_rate_coeffs(5),
                            6: photon_rate_coeffs(6)
        }
        self.atomic_data_dict = atomic_data_dict
        #print(atomic_data_dict)


    def wavelength(self, discard_1, discard_2,  transition):
        '''
        The cdef of Atomic data requires 4 input arguments, so use placeholders
        '''
        #transition = self.transition_check(transition)
        return wl(transition)

    def H_excit(self, transition):
        #transition = self.transition_check(transition)
        MARc = self.atomic_data_dict[transition[0]][0]

        return RateFunction(MARc, transition)

    def H_rec(self, transition):
        #transition = self.transition_check(transition)
        MARc = self.atomic_data_dict[transition[0]][1]

        return RateFunction(MARc, transition)

    def H2_diss(self, transition):
        #transition = self.transition_check(transition)
        MARc = self.atomic_data_dict[transition[0]][2]

        return RateFunction(MARc, transition)

    def H2_pos_diss(self, transition):
        #transition = self.transition_check(transition)
        MARc = self.atomic_data_dict[transition[0]][3]

        return RateFunction(MARc, transition)

    def H3_pos_diss(self, transition):

        return NullRateFunction()

    def H_neg(self, transition):
        #transition = self.transition_check(transition)
        MARc = self.atomic_data_dict[transition[0]][5]

        return RateFunction(MARc, transition)