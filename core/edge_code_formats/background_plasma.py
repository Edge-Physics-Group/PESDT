'''
Loads data from an edge code result to a generic PESDT background plasma
PESDT requires the following plasma parameters, and grid parameters, to be available:
plasma:
    te
    ti
    ne
    ni (main fuel ion)
    n0 (fuel neutral atom)

grid:
    
'''
from edge2d_format import Edge2D
from solps_format import SOLPS

class BackgroundPlasma():

    def __init__(self, edge_code, sim_path):
        '''
        Interface class for loading background plasma data from different edge plasma codes
        '''

        self.edge_data = None
        self.plasma_data = None
        self.grid_data = None
        self.sim_path = sim_path
        self.edge_code = edge_code
        if edge_code == "edge2d":
            self.edge_data = Edge2D(self.sim_path)
        elif edge_code == "solps":
            self.edge_data = SOLPS(self.sim_path)
        elif edge_code == "oedge":
            '''
            TODO
            '''
        else:
            raise Exception("Edge code not supported")
        
        
            

        
    