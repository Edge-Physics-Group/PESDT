
from cherab.core.atomic import AtomicData

# Dummy class for atomic data
class PESDT_Data(AtomicData):

    def __init__(self, atomic_data_dict):

        # Invert dictionary to format transition : wave
        self.atomic_data_dict = {trans: float(wl) for wl, trans in atomic_data_dict.items()}

    def wavelength(self, ion, ion_stage, transition):


        # convert from A to nm!
        return self.atomic_data_dict[transition]*0.1

class PESDT_Power_Data(AtomicData):

    def __init__(self):

        pass

    def wavelength(self, ion, ion_stage, transition):
        pass 


