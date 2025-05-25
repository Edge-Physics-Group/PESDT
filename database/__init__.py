import json

class spectroscopic_lines_db():
    '''
    Database convinience class for loading stored spectroscopic lines from JSON files
    '''
    def __init__(self):
        self.H_lines = json.load("H.json")
        self.He_lines = json.load("He.json")
        self.C_lines = json.load("C.json")
        self.Be_lines = json.load("Be.json")
        self.N_lines = json.load("N.json")
        self.W_lines = json.load("W.json")