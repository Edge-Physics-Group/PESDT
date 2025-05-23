
from matplotlib import patches
import numpy as np
from shapely.geometry import Polygon
# Only available on the JET data cluster
import eproc as ep

from utils import floatToBits
from cell import Cell
# Duplicate from process.py
# Unsure whether using this is necessary, Henri used it in his
# "fixed" version of pyproc, so I'm using it here


class Edge2D():
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
    
    TODO: implement read_eirene_side function for reading the EIRENE output

    '''

    def __init__(self, tranfile):
        self.tranfile = tranfile
        
        self.quad_cells = [] # EDGE2D mesh uses quadralaterals
        self.read_edge2d_fluid_side_data_using_eproc()

    def read_edge2d_fluid_side_data_using_eproc(self):
        '''
        Reads Te, Ti, ne, ni, S_iz, S_rec, n_D^0, n_D^0_2, R, Z + impurities 
        from an EDGE2D-EIRENE tran file
        '''

        print('Getting data from ' + self.tranfile )
        
        # Read in R,Z center, corner coordiantes
        self.rmesh = self.get_eproc_param("EprocDataRead", 'RMESH')
        self.zmesh = self.get_eproc_param("EprocDataRead", 'ZMESH')
        self.rvertp = self.get_eproc_param("EprocDataRead", 'RVERTP')
        self.zvertp = self.get_eproc_param("EprocDataRead", 'ZVERTP')
        # Read in Te, Ti, ni, ne
        self.teve = self.get_eproc_param("EprocDataRead", 'TEVE')
        self.tev = self.get_eproc_param("EprocDataRead", 'TEV')
        self.den = self.get_eproc_param("EprocDataRead", 'DEN')
        self.denel = self.get_eproc_param("EprocDataRead", 'DENEL')
        # Read in na and nm (atomic, molecular)
        self.da = self.get_eproc_param("EprocDataRead", 'DA')
        self.dm = self.get_eproc_param("EprocDataRead", 'DM')
        self.di = self.get_eproc_param("EprocDataRead", 'DI')
        # ?
        self.korpg = self.get_eproc_param("EprocDataRead", 'KORPG')
        # Read in recombination and ionization (?) sources
        self.sirec = self.get_eproc_param("EprocDataRead", 'SIREC')
        self.soun = self.get_eproc_param("EprocDataRead", 'SOUN')
        
        
        for i in range(self.korpg['npts']):
            self.korpg['data'][i] = floatToBits(self.korpg['data'][i])
            
        # GET INNER AND OUTER TARGET DATA
        self.psi_OT = self.get_eproc_param("EprocRow", 'PSI', 'OT', args='ALL_POINTS=0')
        self.psi_IT = self.get_eproc_param("EprocRow", 'PSI', 'IT', args='ALL_POINTS=0')
        self.qeflxd_OT = self.get_eproc_param("EprocRow", 'QEFLXD', 'OT', args='ALL_POINTS=0')
        self.qeflxd_IT = self.get_eproc_param("EprocRow", 'QEFLXD', 'IT', args='ALL_POINTS=0')
        self.qiflxd_OT = self.get_eproc_param("EprocRow", 'QIFLXD', 'OT', args='ALL_POINTS=0')
        self.qiflxd_IT = self.get_eproc_param("EprocRow", 'QIFLXD', 'IT', args='ALL_POINTS=0')
        self.pflxd_OT = self.get_eproc_param("EprocRow", 'PFLXD', 'OT', args = 'ALL_POINTS=0')
        self.pflxd_IT = self.get_eproc_param("EprocRow", 'PFLXD', 'IT', args = 'ALL_POINTS=0')
        self.teve_OT = self.get_eproc_param("EprocRow", 'TEVE', 'OT', args = 'ALL_POINTS=0')
        self.teve_IT = self.get_eproc_param("EprocRow", 'TEVE', 'IT', args = 'ALL_POINTS=0')
        self.tev_OT = self.get_eproc_param("EprocRow", 'TEV', 'OT', args = 'ALL_POINTS=0')
        self.tev_IT = self.get_eproc_param("EprocRow", 'TEV', 'IT', args = 'ALL_POINTS=0')
        self.denel_OT = self.get_eproc_param("EprocRow", 'DENEL', 'OT', args = 'ALL_POINTS=0')
        self.denel_IT = self.get_eproc_param("EprocRow", 'DENEL', 'IT', args = 'ALL_POINTS=0')
        self.da_OT = self.get_eproc_param("EprocRow", 'DA', 'OT', args = 'ALL_POINTS=0')
        self.da_IT = self.get_eproc_param("EprocRow", 'DA', 'IT', args = 'ALL_POINTS=0')
        self.dm_OT = self.get_eproc_param("EprocRow", 'DM', 'OT', args='ALL_POINTS=0')
        self.dm_IT = self.get_eproc_param("EprocRow", 'DM', 'IT', args='ALL_POINTS=0')
        self.di_OT = self.get_eproc_param("EprocRow", 'DI', 'OT', args='ALL_POINTS=0')
        self.di_IT = self.get_eproc_param("EprocRow", 'DI', 'IT', args='ALL_POINTS=0')

        # GET GEOM INFO
        self.geom = {'rpx':0,'zpx':0}
        self.geom['zpx']*=-1.0
        

        # GET MID-PLANE PROFILE
        self.ne_OMP = self.get_eproc_param("EprocRow", 'DENEL', 'OMP', args = 'ALL_POINTS=0')
        self.te_OMP = self.get_eproc_param("EprocRow", 'TEVE', 'OMP', args = 'ALL_POINTS=0')
        self.ni_OMP = self.get_eproc_param("EprocRow", 'DEN', 'OMP', args = 'ALL_POINTS=0')
        self.ti_OMP = self.get_eproc_param("EprocRow", 'TEV', 'OMP', args = 'ALL_POINTS=0')
        self.da_OMP = self.get_eproc_param("EprocRow", 'DA', 'OMP', args = 'ALL_POINTS=0')
        self.dm_OMP = self.get_eproc_param("EprocRow", 'DM', 'OMP', args = 'ALL_POINTS=0')
        self.di_OMP = self.get_eproc_param("EprocRow", 'DI', 'OMP', args = 'ALL_POINTS=0')
        self.psi_OMP = self.get_eproc_param("EprocRow", 'PSI', 'OMP', args = 'ALL_POINTS=0')

        # GET POWER CROSSING THE SEPARATRIX
        self.powsol = self.get_eproc_param("EprocDataRead", 'POWSOL')

        # GET IMPURITY DATA
        self.imp1_atom_num = None
        self.imp2_atom_num = None
        self.imp2_bundle_num = None
        self.imp1_chrg_idx = []
        self.imp2_chrg_idx = []
        self.imp1_denz = []
        self.imp2_denz = []
        self.imp2_chargez = []
        _imp1_denz = []
        _imp2_denz = []
        _imp1_chrg_idx = []
        _imp2_chrg_idx = []
        
        self.zch = self.get_eproc_param("EprocDataRead", 'ZCH') # impurity atomic number (max 2 impurities)
        self.nz = self.get_eproc_param("EprocDataRead", 'NZ') #  number of impurity charge states (max 2 impurities)
        # Convert zch, nz to int 
        zch = []
        nz = []
        for i in self.zch['data']:
            zch.append(int(floatToBits(i)))
        for i in self.nz['data']:
            nz.append(int(floatToBits(i)))
        self.zch['data'] = np.array(zch)
        self.nz['data'] = np.array(nz)

        if self.zch['data'][0] > 0.0:
            self.imp1_atom_num = self.zch['data'][0]
            # First append neutral density, then each charge state density
            _imp1_denz.append(self.get_eproc_param("EprocDataRead", 'DZ_1'))
            for i in range(self.zch['data'][0]):
                if i < 9 :
                    stridx = '0' + str(i+1)
                    _imp1_chrg_idx.append(stridx)
                else:
                    stridx = str(i+1)
                    _imp1_chrg_idx.append(stridx)
                _imp1_denz.append(self.get_eproc_param("EprocDataRead", 'DENZ'+stridx))

            if self.zch['data'][1] > 0.0:
                self.imp2_atom_num = self.zch['data'][1]
                self.imp2_bundle_num = self.nz['data'][1]
                # First append neutral density, then each charge state density
                _imp2_denz.append(self.get_eproc_param_temp("EprocDataRead", 'DZ_2'))
                self.imp2_chargez.append({'data':[0]*[0]})
                for i in range(self.nz['data'][1]):
                    if i + self.zch['data'][0] < 9:
                        stridx = '0' + str(i + 1 + self.zch['data'][0])
                        _imp2_chrg_idx.append(stridx)
                    else:
                        stridx = str(i + 1 + self.zch['data'][0])
                        _imp2_chrg_idx.append(stridx)
                    _imp2_denz.append(self.get_eproc_param("EprocDataRead", 'DENZ'+stridx))
                    self.imp2_chargez.append(self.get_eproc_param("EprocDataRead", 'ZI'+stridx))
        else:
            # If no impurities found reset
            self.zch = None

        
        rtmp = self.get_eproc_param("EprocRing", 'RMESH', 'S01', args = 'parallel=1')
        ztmp = self.get_eproc_param("EprocRing", 'ZMESH', 'S01', args = 'parallel=1')
        rtmp1 = self.get_eproc_param("EprocDataRead", 'RVESM1')
        ztmp1 = self.get_eproc_param("EprocDataRead", 'ZVESM1')
        rtmp2 = self.get_eproc_param("EprocDataRead", 'RVESM2')
        ztmp2 = self.get_eproc_param("EprocDataRead", 'ZVESM2')
        # Z data is upside down
        self.zvertp['data'] *=-1.0
        self.zmesh['data'] *=-1.0

        self.NE2Ddata = 0
        for i in range(self.korpg['npts']):
            j = self.korpg['data'][i]
            if j != 0: self.NE2Ddata = self.NE2Ddata+1

        self.cells = []
        self.patches = []

        self.row = np.zeros((self.NE2Ddata), dtype=int)
        self.ring = np.zeros((self.NE2Ddata), dtype=int)
        self.rv = np.zeros((self.NE2Ddata, 5))
        self.zv = np.zeros((self.NE2Ddata, 5))
        self.te = np.zeros((self.NE2Ddata))
        self.ti = np.zeros((self.NE2Ddata))
        self.ne = np.zeros((self.NE2Ddata))
        self.ni = np.zeros((self.NE2Ddata))
        self.n0 = np.zeros((self.NE2Ddata))
        self.n2 = np.zeros((self.NE2Ddata))
        self.n2p = np.zeros((self.NE2Ddata))
        self.srec = np.zeros((self.NE2Ddata))
        self.sion = np.zeros((self.NE2Ddata))
        self.imp1_den = np.zeros((len(_imp1_chrg_idx)+1,self.NE2Ddata))
        self.imp2_den = np.zeros((len(_imp2_chrg_idx)+1,self.NE2Ddata))
        self.imp2_charge = np.zeros((len(self.imp2_chrg_idx)+1,self.NE2Ddata))
        ringct = 0        
        k = 0
        for i in range(self.korpg['npts']):
            j = int(self.korpg['data'][i] - 1) # gotcha: convert from fortran indexing to idl/python
            if j >= 0:
                j*=5
                self.rv[k] = [self.rvertp['data'][j],  self.rvertp['data'][j+1], self.rvertp['data'][j+2], self.rvertp['data'][j+3], self.rvertp['data'][j]]
                self.zv[k] = [self.zvertp['data'][j],  self.zvertp['data'][j+1], self.zvertp['data'][j+2], self.zvertp['data'][j+3], self.zvertp['data'][j]]
                self.te[k] = self.teve['data'][i]
                self.ti[k] = self.tev['data'][i]
                self.ni[k] = self.den['data'][i]
                self.ne[k] = self.denel['data'][i]
                self.n0[k] = self.da['data'][i]
                self.n2[k] = self.dm['data'][i]
                self.n2p[k] = self.di['data'][i]
                self.srec[k] = self.sirec['data'][i]
                self.sion[k] = self.soun['data'][i]

                if self.imp1_atom_num:
                    for ichrg in range(len(_imp1_chrg_idx)+1):# +1 to include neutral density
                        self.imp1_den[ichrg,k] = _imp1_denz[ichrg]['data'][i]
                if self.imp2_atom_num:
                    for ichrg in range(len(_imp2_chrg_idx)+1):# +1 to include neutral density
                        self.imp2_den[ichrg,k] = _imp2_denz[ichrg]['data'][i]
                        
                poly = patches.Polygon([(self.rv[k,0], self.zv[k,0]), (self.rv[k,1], self.zv[k,1]), (self.rv[k,2], self.zv[k,2]), (self.rv[k,3], self.zv[k,3]), (self.rv[k,4], self.zv[k,4])], closed=True)
                self.patches.append(poly)
                # create Cell object for each polygon containing the relevant field data
                shply_poly = Polygon(poly.get_xy())

                self.cells.append(Cell(self.rmesh['data'][i], self.zmesh['data'][i],
                                       row=self.row[k], ring=self.ring[k],
                                       poly=shply_poly, te=self.te[k],
                                       ne=self.ne[k], ni=self.ni[k],
                                       n0=self.n0[k], n2 = self.n2[k], n2p = self.n2p[k], Srec=self.srec[k], Sion=self.sion[k]))
                k+=1

        ##############################################
        # GET STRIKE POINT COORDS AND SEPARATRIX POLY
        ztmp['ydata'] = -1.0*np.array(ztmp['ydata'])
        self.osp = [rtmp['ydata'][0], ztmp['ydata'][0]]
        self.isp = [rtmp['ydata'][rtmp['npts']-1], ztmp['ydata'][ztmp['npts']-1]]

        # find start and end index
        istart = -1
        iend = -1
        for i in range(self.korpg['npts']):
            j = int(self.korpg['data'][i] - 1) # gotcha: convert from fortran indexing to idl/python
            j*=5
            if j >= 0:
                if (self.rvertp['data'][j+4] == rtmp['ydata'][0]) and (self.zvertp['data'][j+4] == ztmp['ydata'][0]):
                    istart = i
                if (self.rvertp['data'][j+4] == rtmp['ydata'][rtmp['npts']-1]) and (self.zvertp['data'][j+4] == ztmp['ydata'][ztmp['npts']-1]):
                    iend = i

        sep_points = []
        sep_points_below_xpt = []
        k = 0
        for i in range(istart, iend+1):
            j = int(self.korpg['data'][i] - 1)
            j*=5
            sep_points.append((self.rvertp['data'][j+0],self.zvertp['data'][j+0]))
            if self.zvertp['data'][j+0] <= self.geom['zpx']:
                sep_points_below_xpt.append((self.rvertp['data'][j+0],self.zvertp['data'][j+0]))
            # sep_points.append((self.rvertp['data'][j+0],self.zvertp['data'][j+0]))
            k+=1
        sep_points_below_xpt.append((self.geom['rpx'], self.geom['zpx']))

        self.sep_poly = patches.Polygon(sep_points, closed=False, ec='pink', linestyle='dashed', lw=2.0, fc='None', zorder=10)
        self.shply_sep_poly = Polygon(self.sep_poly.get_xy())
        self.sep_poly_below_xpt = patches.Polygon(sep_points_below_xpt, closed=False, ec='k', fc='None', zorder=10)
        self.shply_sep_poly_below_xpt = Polygon(self.sep_poly_below_xpt.get_xy())

        ##############################################
        # GET WALL POLYGON
        ztmp1['data'] = -1.0*ztmp1['data']
        ztmp2['data'] = -1.0*ztmp2['data']
        # remove non connected segments
        useSegment = np.zeros((rtmp1['npts']))
        nsegs = 0
        for i in range(rtmp1['npts']):
            check = 0
            if i != 0:
                if ((rtmp1['data'][i] == rtmp2['data'][i-1]) and (ztmp1['data'][i] == ztmp2['data'][i-1])) or \
                    ((rtmp2['data'][i] == rtmp1['data'][i-1]) and (ztmp2['data'][i] == ztmp1['data'][i-1])):
                    check = 1
            if i != (rtmp1['npts']):
                if ((rtmp1['data'][i] == rtmp2['data'][i+1]) and (ztmp1['data'][i] == ztmp2['data'][i+1])) or \
                    ((rtmp2['data'][i] == rtmp1['data'][i+1]) and (ztmp2['data'][i] == ztmp1['data'][i+1])):
                    check = 1
            if check:
                useSegment[i] = 1
                nsegs += 1

        wall_poly_pts = []
        for i in range(rtmp1['npts']):
            if useSegment[i]:
                wall_poly_pts.append((rtmp1['data'][i],ztmp1['data'][i]))
                wall_poly_pts.append((rtmp2['data'][i],ztmp2['data'][i]))
        wall_poly_pts.append(wall_poly_pts[0]) # connect last point to first to complete wall polygon

        self.wall_poly = patches.Polygon(wall_poly_pts, closed=False, ec='k', lw=2.0, fc='None', zorder=10)
        self.shply_wall_poly = Polygon(self.wall_poly.get_xy())
        
    def get_eproc_param(self, funstr, parstr, par2str=None, args=None):
        '''
        Uses Python eproc to load data from a tran file
        funstr: eproc function name
        parstr: tran file variable name (e.g. "DENEL" or "TEV")
        par2str: identifier for location ('OT', 'IT', 'OMP', 'IMP') or location
                 as an integer index 
        args: seemingly unused, execpt for accesing eproc ring/row
        
        '''
        if par2str and args:
            if type(par2str) == int:
                cmd = 'ret=' + funstr +'(tranfile,'+ """'""" + parstr + """'""" + ',' + str(par2str) + ',' + args + ')'
            elif type(par2str) == str:
                cmd = 'ret=' + funstr +'(tranfile,'+ """'""" + parstr + """'""" + ','  + """'""" + par2str + """'""" + ',' + args + ')'

            print(cmd)
            if funstr == "EprocRow":
                epdata = ep.row(self.tranfile,parstr,par2str)
            else:
                epdata = ep.ring(self.tranfile,parstr,par2str)
            return {'xdata': epdata.xData, 'ydata': epdata.yData, 'npts': epdata.nPts}

        else:
            cmd = 'ret=' + funstr +'(tranfile,'+ """'""" + parstr + """'""" + ')'
            epdata = ep.data(self.tranfile,parstr)
            print(parstr, epdata.nPts)
            return {'data': epdata.data, 'npts': epdata.nPts}
    
    # def read_eirene_data(self):

