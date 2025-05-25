


class BackgroundPlasma():   
    '''
        Interface base class for loading background plasma data from different edge plasma codes
        
        The background plasma does not need any mehtods, only some data fields, which are listed here.
        Additional fields can be loaded, but are not required for running PESDT
    '''

    def __init__(self):
        self.edge_code = None
        self.sim_path = None
        
        self.te = None
        self.ti = None
        self.ne = None
        self.ni = None

        self.n0 = None
        self.n2 = None
        self.n2p = None

        self.cells = None 
            

        
    