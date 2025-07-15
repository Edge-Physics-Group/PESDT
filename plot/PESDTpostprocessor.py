######################################################################################################
### PESDT POST-PROCESSOR --- PLOT EMISSION ALONG LOS, DERIVED QUANTITIES, AND BG PLASMA PARAMETERS ###
######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import json, pickle, os
from core.edge_code_formats import Edge2D, SOLPS, OEDGE, EIRENE





class PESDTPostProcessor:
    colors = ["black", "red", "blue"]
    colors2 = ["black", "red", "blue", "green", "orange", "purple"]
    EDGE_CODES_DICT = {"edge2d": Edge2D, "solps": SOLPS, "oedge": OEDGE, "eirene": EIRENE}
    linestyles = ["solid", "dashed", "dashdot"]
    markers = ["x", "o", ".", "^", "v"]
    cmap = "bwr"

    def __init__(self, result_path: str = None, edge_code: str = None, edge_code_path: str = None):
        # Load PESDT results
        if result_path is not None:
            res_dir = os.path.join(os.path.expanduser("~"), result_path)
            print(f"Reading results from: {res_dir}")
            with open(res_dir, "r") as f:
                self.PESDTresult = json.load(f)
        else :
            self.PESDTresult = None
        pass
        # Load edge code background plasma
        if edge_code is not None and edge_code_path is not None:
            self.edge_code = edge_code
            self.bg_data_path = os.path.join(os.path.expanduser("~"), edge_code_path)
            print(f"Loading edge code '{self.edge_code}' data from: {self.bg_data_path}")
            self.data = self.EDGE_CODES_DICT[self.edge_code](self.bg_data_path)
        else: 
            self.edge_code = None
            self.bg_data_path = None
            self.data = None

    def plot_target_data(self, ax = None):

        return
    
    def plot_row(self, ax = None):


        return
    
    def plot_ring(self, ax = None, colors = None, linestyle = None):

        return
    
    def plot_2D(self, ax = None, cmap = None, ):
        return
    
    def plot_synth_instrument(self, ax = None):
        return
    
    def parse_PESDT_output(self):
        return
    
if __name__ == "__main__":
    print("This is the PESDT post-processing tool. It is recommended that you import the 'PESDTPostProcessor' class in your own .py script file, and use it there")