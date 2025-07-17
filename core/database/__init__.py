import json, os


class spectroscopic_lines_db:
    '''
    Database convinience class for loading stored spectroscopic lines from JSON files
    '''
    def __init__(self):
        self.H_lines = self._load( "H.json")
        self.He_lines = self._load("He.json")
        self.C_lines = self._load("C.json")
        self.Be_lines = self._load("Be.json")
        self.N_lines = self._load("N.json")
        self.W_lines = self._load("W.json")

    def _load(self, path):
        with open(path, "r") as f:
            return json.load(f)