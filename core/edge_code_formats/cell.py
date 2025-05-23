
class Cell:

    def __init__(self, R=None, Z=None, row=None, ring=None,
                 poly=None, te=None, ti=None, ne=None, ni=None, n0=None,  n2 = None, n2p = None,
                 Srec=None, Sion=None):

        self.R = R # m
        self.Z = Z # m
        self.row = row
        self.ring = ring
        self.poly = poly
        self.te = te # eV
        self.ti = ti # eV
        self.ni = ni # m^-3
        self.ne = ne # m^-3
        self.n0 = n0 # m^-3
        self.n2 = n2
        self.n2p = n2p
        self.H_emiss = {}
        self.ff_fb_filtered_emiss = None
        self.ff_radpwr = None
        self.ff_radpwr_perm3 = None
        self.ff_fb_radpwr = None
        self.ff_fb_radpwr_perm3 = None
        self.H_radpwr = None
        self.H_radpwr_perm3 = None  # W m^-3
        self.H_radpwr_Lytrap = None
        self.H_radpwr_Lytrap_perm3 = None  # W m^-3
        self.imp_emiss = {}
        self.Srec = Srec # m^-3 s^-1
        self.Sion = Sion # m^-3 s^-1

        # LOS ORTHOGONAL POLYGON PROPERTIES
        self.dist_to_los_v1 = None
        self.los_ortho_width = None
        self.los_ortho_delL = None