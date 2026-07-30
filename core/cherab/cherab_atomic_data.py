
from cherab.core.atomic import AtomicData

# Dummy class for atomic data
class PESDT_Data(AtomicData):

    def __init__(self, atomic_data_dict):

        # Invert dictionary to format transition : wave
        self.atomic_data_dict = {trans: float(wl) for wl, trans in atomic_data_dict.items()}

    def wavelength(self, ion, ion_stage, transition):


        # convert from A to nm!
        return self.atomic_data_dict[transition]*0.1

# Dummy class for power. Fill atomic data dict with keys like ["plt", "prb", "fffb"]
class PESDT_Power_Data(AtomicData):

    def __init__(self, atomic_data_dict):

        self.atomic_data_dict = atomic_data_dict

    def wavelength(self, ion, ion_stage, transition):
        self.atomic_data_dict[transition] 


