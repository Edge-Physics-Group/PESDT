
from matplotlib import patches
import numpy as np
from shapely.geometry import Polygon
from core.utils import floatToBits
from .cell import Cell
from .background_plasma import BackgroundPlasma
from .readtran import Tran


import logging
logger = logging.getLogger(__name__)

class Edge2D(BackgroundPlasma):
    '''
    A class to handle reading EDGE2D-EIRENE simulation results. To initialize, give
    "tranfile", i.e. the directory path to the result tran file, as a string (str)
    
    __init__: 
    initializes the class, and automatically loads in the fluid variables
    currently read_eirene_side is not implemented
    
    read_edge2d_fluid_side_data_using_eproc:
    loads in a specified set of plasma parameters, coordinates, target profiles etc.
    
    get_eproc_param:
    a function which implements Python eproc for reading row, rings or single points
    from the tran file
    '''

    def __init__(self, tranfile):
        super().__init__()
        self.tranfile = tranfile
        
        self.quad_cells = [] # EDGE2D mesh uses quadralaterals
        self.tran = Tran(self.tranfile)
        self.read_edge2d_data()

    def read_edge2d_data(self):
        '''
        Reads Te, Ti, ne, ni, S_iz, S_rec, n_D^0, n_D^0_2, R, Z + impurities 
        from an EDGE2D-EIRENE tran file
        '''

        logger.info(f"Getting data from  {self.tranfile}" )
        
        # Read in R,Z center, corner coordiantes
        self.rmesh = self.tran.load_data2d('RMESH')
        self.zmesh = self.tran.load_data2d( 'ZMESH')
        self.rvertp = self.tran.load_data2d('RVERTP')
        self.zvertp = self.tran.load_data2d('ZVERTP')
        # Read in Te, Ti, ni, ne
        self.teve = self.tran.load_data2d( 'TEVE')
        self.tev = self.tran.load_data2d( 'TEV')
        self.den = self.tran.load_data2d( 'DEN')
        self.denel = self.tran.load_data2d( 'DENEL')
        # Read in na and nm (atomic, molecular)
        self.da = self.tran.load_data2d( 'DA')
        self.dm = self.tran.load_data2d( 'DM')
        #self.di = self.tran.load_data2d( 'DI')
        # Cell indices
        self.korpg = self.tran.load_data2d( 'KORPG')
        # Read in recombination and ionization sources
        self.sirec = self.tran.load_data2d( 'SIREC')
        self.soun = self.tran.load_data2d( 'SOUN')
            
        # GET INNER AND OUTER TARGET DATA
        self.psi_OT = self.tran.load_data1d('PSI', ot=True, )
        self.psi_IT = self.tran.load_data1d('PSI', it=True)
        self.qeflxd_OT = self.tran.load_data1d('QEFLXD', ot=True)
        self.qeflxd_IT = self.tran.load_data1d('QEFLXD', it=True)
        self.qiflxd_OT = self.tran.load_data1d('QIFLXD', ot=True)
        self.qiflxd_IT = self.tran.load_data1d('QIFLXD', it=True)
        self.pflxd_OT = self.tran.load_data1d('PFLXD', ot=True)
        self.pflxd_IT = self.tran.load_data1d('PFLXD', it=True)
        self.teve_OT = self.tran.load_data1d('TEVE', ot=True)
        self.teve_IT = self.tran.load_data1d('TEVE', it=True)
        self.tev_OT = self.tran.load_data1d('TEV', ot=True)
        self.tev_IT = self.tran.load_data1d('TEV', it=True)
        self.denel_OT = self.tran.load_data1d('DENEL', ot=True)
        self.denel_IT = self.tran.load_data1d('DENEL', it=True)
        self.da_OT = self.tran.load_data1d('DA', ot=True)
        self.da_IT = self.tran.load_data1d('DA', it=True)
        self.dm_OT = self.tran.load_data1d('DM', ot=True)
        self.dm_IT = self.tran.load_data1d('DM', it=True)
        #self.di_OT = self.tran.load_data1d('DI', ot=True)
        #self.di_IT = self.tran.load_data1d('DI', it=True)

        # GET GEOM INFO
        self.geom = {'rpx':self.tran.rpx,'zpx':self.tran.zpx}
        self.geom['zpx']*=-1.0
        

        # GET MID-PLANE PROFILE
        self.ne_OMP = self.tran.load_data1d('DENEL', omp = True)
        self.te_OMP = self.tran.load_data1d('TEVE', omp = True)
        self.ni_OMP = self.tran.load_data1d('DEN', omp = True)
        self.ti_OMP = self.tran.load_data1d('TEV', omp = True)
        self.da_OMP = self.tran.load_data1d('DA', omp = True)
        self.dm_OMP = self.tran.load_data1d('DM', omp = True)
        #self.di_OMP = self.tran.load_data1d('DI', omp = True)
        self.psi_OMP = self.tran.load_data1d('PSI', omp = True)

        # GET POWER CROSSING THE SEPARATRIX
        self.powsol = self.tran.powsol

        # GET IMPURITY DATA
        # TODO
        
        # Z data is upside down
        self.zvertp *=-1.0
        self.zmesh *=-1.0

        self.cells = []
        self.patches = []

        self.row = np.zeros((self.tran.np+1), dtype=int)
        self.ring = np.zeros((self.tran.np+1), dtype=int)
        self.rv = np.zeros((self.tran.np+1, 5))
        self.zv = np.zeros((self.tran.np+1, 5))
        self.te = np.zeros((self.tran.np+1))
        self.ti = np.zeros((self.tran.np+1))
        self.ne = np.zeros((self.tran.np+1))
        self.ni = np.zeros((self.tran.np+1))
        self.n0 = np.zeros((self.tran.np+1))
        self.n2 = np.zeros((self.tran.np+1))
        self.n2p = np.zeros((self.tran.np+1))
        self.srec = np.zeros((self.tran.np+1))
        self.sion = np.zeros((self.tran.np+1))
       
    
        k = 0
        for i in range(self.tran.np):
            j = int(self.korpg[i] - 1) # gotcha: convert from fortran indexing to idl/python
            if j >= 0:
                j*=5
                self.rv[k] = [self.rvertp[j],  self.rvertp[j+1], self.rvertp[j+2], self.rvertp[j+3], self.rvertp[j]]
                self.zv[k] = [self.zvertp[j],  self.zvertp[j+1], self.zvertp[j+2], self.zvertp[j+3], self.zvertp[j]]
                self.te[k] = self.teve[i]
                self.ti[k] = self.tev[i]
                self.ni[k] = self.den[i]
                self.ne[k] = self.denel[i]
                self.n0[k] = self.da[i]
                self.n2[k] = self.dm[i]
                self.n2p[k] = 0.0 #self.di[i]
                self.srec[k] = self.sirec[i]
                self.sion[k] = self.soun[i]

               
                poly = patches.Polygon([(self.rv[k,0], self.zv[k,0]), (self.rv[k,1], self.zv[k,1]), (self.rv[k,2], self.zv[k,2]), (self.rv[k,3], self.zv[k,3]), (self.rv[k,4], self.zv[k,4])], closed=True)
                self.patches.append(poly)
                # create Cell object for each polygon containing the relevant field data
                shply_poly = Polygon(poly.get_xy())

                self.cells.append(Cell(self.rmesh[i], self.zmesh[i],
                                       row=self.row[k], ring=self.ring[k],
                                       poly=shply_poly, te=self.te[k],
                                       ne=self.ne[k], ni=self.ni[k],
                                       n0=self.n0[k], n2 = self.n2[k], n2p = self.n2p[k], Srec=self.srec[k], Sion=self.sion[k]))
                k+=1

        ##############################################
        # GET STRIKE POINT COORDS AND SEPARATRIX POLY
        
        self.osp, self.isp = self.tran.sp
        self.sep_poly = patches.Polygon(self.tran.sepx, closed=False, ec='pink', linestyle='dashed', lw=2.0, fc='None', zorder=10)
        self.shply_sep_poly = Polygon(self.sep_poly.get_xy())
        
        ##############################################
        # GET WALL POLYGON
        
        self.wall_poly = self.tran.wall
        self.shply_wall_poly = Polygon(self.wall_poly.get_xy())
        
    
    
