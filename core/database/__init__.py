import json, os


class spectroscopic_lines_db:
    '''
    Database convinience class for loading stored spectroscopic lines from JSON files
    '''
    def __init__(self):
        pesdt_home = os.environ.get('PESDT_HOME', os.path.expanduser('~') + "PESDT/")
        self.H_lines    = self._load(os.path.join(pesdt_home, "core/database/H.json"))
        self.He_lines   = self._load(os.path.join(pesdt_home, "core/database/He.json"))
        self.C_lines    = self._load(os.path.join(pesdt_home, "core/database/C.json"))
        self.Be_lines   = self._load(os.path.join(pesdt_home, "core/database/Be.json"))
        self.N_lines    = self._load(os.path.join(pesdt_home, "core/database/N.json"))
        self.W_lines    = self._load(os.path.join(pesdt_home, "core/database/W.json"))

    def _load(self, path):
        with open(path, "r") as f:
            return json.load(f)