import json, os


class spectroscopic_lines_db:
    '''
    Database convinience class for loading stored spectroscopic lines from JSON files
    '''
    def __init__(self):
        
        self.H_lines    = self._load(os.path.join(os.path.expanduser("~"), "PESDT/core/databese/H.json"))
        self.He_lines   = self._load(os.path.join(os.path.expanduser("~"), "PESDT/core/databese/He.json"))
        self.C_lines    = self._load(os.path.join(os.path.expanduser("~"), "PESDT/core/databese/C.json"))
        self.Be_lines   = self._load(os.path.join(os.path.expanduser("~"), "PESDT/core/databese/Be.json"))
        self.N_lines    = self._load(os.path.join(os.path.expanduser("~"), "PESDT/core/databese/N.json"))
        self.W_lines    = self._load(os.path.join(os.path.expanduser("~"), "PESDT/core/databese/W.json"))

    def _load(self, path):
        with open(path, "r") as f:
            return json.load(f)