import json, os


class spectroscopic_lines_db:
    '''
    Database convinience class for loading stored spectroscopic lines from JSON files
    '''
    def __init__(self):
        pesdt_home = os.environ.get('PESDT_HOME', os.path.expanduser('~') + "PESDT/")
        self.H_lines    = self._load(os.path.join(pesdt_home, "core/database/H.json"))
        self.D_lines    = self._load(os.path.join(pesdt_home, "core/database/H.json"))
        self.T_lines    = self._load(os.path.join(pesdt_home, "core/database/H.json"))
        self.He_lines   = self._load(os.path.join(pesdt_home, "core/database/He.json"))
        self.C_lines    = self._load(os.path.join(pesdt_home, "core/database/C.json"))
        self.Be_lines   = self._load(os.path.join(pesdt_home, "core/database/Be.json"))
        self.N_lines    = self._load(os.path.join(pesdt_home, "core/database/N.json"))
        self.W_lines    = self._load(os.path.join(pesdt_home, "core/database/W.json"))
        self.bolo_wl    = self._load(os.path.join(pesdt_home, "core/database/bolo.json"))
        self.data_full = {"H": self.H_lines, "D": self.H_lines,"T": self.H_lines,"He": self.He_lines, "C": self.C_lines, "Be": self.Be_lines, "N": self.N_lines, "W": self.W_lines}
        self.data = {"H": self.H_lines, "D": self.H_lines,"T": self.H_lines,"He": self.He_lines, "C": self.C_lines, "Be": self.Be_lines, "N": self.N_lines, "W": self.W_lines}
        # flatten data
        for species, data in self.data.items():
            flat_data = {}
            for key, item in data.items():
                if key == "ATOM_NUM": continue
                for k, i in item.items():
                    flat_data[k] = tuple(i) # Add items, all transitions are unique
            self.data[species] = flat_data

    def _load(self, path):
        with open(path, "r") as f:
            return json.load(f)
