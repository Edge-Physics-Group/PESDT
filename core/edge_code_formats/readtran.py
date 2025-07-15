
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm, ListedColormap
import matplotlib.cm as cm

class Tran :
    def __init__(self,tranfile='tran'):

        # Dictionary storing the data by key values
        self.tran={}
        self.readTran(tranfile)
        if 'ITAG' not in self.tran :
            print('Tran file version not compatible. Use eproc instead')
            raise Exception("Tran file version not compatible. Use eproc instead")

        # Copy each field in an attribute for direct use
        for f in self.tran :
            setattr(self, f.lower(), self.tran[f]['data'] if type(self.tran[f]) is dict else self.tran[f])
        
        # Fill some derived signals
        self.nzs = 0
        self.denztot = []
        idz = 0
        for nz in self.nz:
            self.denztot.append(np.zeros(self.np))
            for iz in range(nz):
                idz = idz+1
                self.denztot[-1] = self.denztot[-1] + self.tran['DENZ{:02d}'.format(idz)]['data']
        for izs in range(len(self.nz)):
            fieldname = 'DENZTOT{:d}'.format(izs+1)
            self.tran[fieldname] = {
                     'type':'2D' ,
                     'staggered':False,
                     'data':self.denztot[izs],
                     'desc':"Tot. imp density",
                     'unit':'M-3'}
            setattr(self, fieldname.lower(), self.tran[fieldname]['data'])

        self.nj2d = self.nrow-1
        self.ilc = self.ni2d - (self.nxw - self.iopen + 1)
        self.itag = self.itag.reshape((-1,5))
        self.korxy = self.korxy.reshape((-1,self.ni2d)).T
        self.jfp = self.itag[0,1]

        # Find the first index or the main plasma in the 2D grid
        self.ifmp = 0
        for ii in range(self.ni2d) :
            k = self.korxy[ii,self.jfp-1]
            if k == 0 : continue
            self.ifmp = ii
            break

        # Find targets index
        self.jjtarget = []
        for jj in range(self.nj2d) :
            k = self.korxy[self.ni2d-2,jj] - 1
            if(k>self.np) : continue
            if self.itag[k,3]>0 : self.jjtarget.append(jj)
        self.nxpnt = int(len(self.jjtarget)/2)

        # Build correspondency arrays to access the data by rings
        self.kory = np.zeros((self.nc,self.nj2d+1), dtype=int)
        self.nj   = np.zeros(self.nc, dtype=int)
        i = 0
        for k in range(self.np):
            if self.itag[k,0]!=i :
                i = self.itag[k,0]
                j = 1
            self.kory[i-1,j-1] = k
            self.nj[i-1] = self.nj[i-1]+1
            j = j+1

        # Build correspondency arrays to access the data by 2D map
        self.ikor = np.zeros(self.np, dtype=int)
        self.jkor = np.zeros(self.np, dtype=int)
        for jj in range(self.nj2d) :
            for ii in range(self.ni2d) :
                k = self.korxy[ii,jj]
                if k ==0 : continue
                if k > self.np : continue
                self.ikor[k-1] = ii
                self.jkor[k-1] = jj

        # Build regions
        self.regkor = np.zeros((self.np+self.nc), dtype=int)
        region = -1
        nj = 0
        for k in range(self.np):
            i = self.itag[k,0] - 1
            # Usually, regions are found with different row sizes
            # but careful in case consecutive SOL/PFR have the same
            # Hence one more control on first cell of the ring
            k1 = self.kory[i,1] - 1
            if i > 1 and k == k1 :
                if (self.ikor[k+1] != self.ikor[self.kory[i-1,1]+1]+1 or 
                   self.jkor[self.kory[i,2]]!=self.jkor[self.kory[i-1,2]]) :
                    nj = self.nj[i]
                    region = region + 1
            elif nj != self.nj[i] :
                nj = self.nj[i]
                region = region + 1
            self.regkor[k] = region
        for i in range(self.nc) :
            self.regkor[self.np+i] = self.regkor[self.kory[i,0]]

    def readString(self,f):
       ndata = np.fromfile(f, dtype=np.int32, count = 1)
       data  = np.fromfile(f, dtype=np.byte, count = ndata[0])
       string = ''.join([chr(x) for x in data])  
       dum = np.fromfile(f, dtype=np.int32, count = 1)
       return string

    def readInts(self,f):
        ndata = int(np.fromfile(f, dtype=np.int32, count = 1)[0]/4) # 4 = int size
        data  = np.fromfile(f, dtype=np.int32, count = ndata)
        dum = np.fromfile(f, dtype=np.int32, count = 1)
        return data

    def readFloats(self,f):
        ndata = int(np.fromfile(f, dtype=np.int32, count = 1)[0]/4) # 4 = int size
        data  = np.fromfile(f, dtype=np.single, count = ndata)
        dum = np.fromfile(f, dtype=np.int32, count = 1)
        return data

    def readTran(self,filename):
        self.offsets = {}
        with open(filename,'rb') as f:
            while(True):
                string = self.readString(f)
                if string[:9]=='*GEOMETRY':
                    gridfile = self.readString(f)
                    self.tran['gridfile'] = gridfile
                    string = self.readString(f)
                    geom = self.readInts(f)
                    i=0
                    for param in string.split(','):
                        self.tran[param.strip()]=geom[i]
                        i=i+1
                    fieldType = '0D'
                elif string[:12]=='*TIME TRACES' :
                    fieldType = '1D'

                elif string[0]=='#' :
                    offset=f.tell()
                    staggered = string[59]=='S'
                    if(string.split()[1]=='I'): data = self.readInts(f)
                    else : data = self.readFloats(f)
                    if len(data)==self.tran['NP'] : fieldType = '2D'
                    fieldname = string[1:].split()[0]
                    self.offsets[fieldname]=offset
                    self.tran[fieldname] = {
                        'type':fieldType ,
                        'staggered':staggered,
                        'data':data,
                        'desc':string[18:50].strip(),
                        'unit':string[50:59].strip()}

                elif string[0]=='%':
                    fortranFormat = string.split()[0][1:]
                    if fortranFormat[-1]=='I':
                        data = self.readInts(f)
                    elif fortranFormat[-1]=='R':
                        data = self.readFloats(f)

                elif string[0] in ['*',' ','-'] :
                    if string[1:4]=='EOF':
                        break
                    if string[1:8]=='CATALOG':
                        self.tran['CATID'] = '/'.join(self.readString(f).split())
                    if string[1:15]=='TIME STEP DATA':
                        tmp = self.readString(f).split()
                        self.tran['TIME'] = float(tmp[2])
                        self.tran['STEP'] = int(tmp[5])
                        try :
                            self.tran['TIMJET'] = float(tmp[9])
                        except :
                            pass
                else :
                    print('format not found',string)

    def getNames(self):
        return [field for field in self.tran]

    def getField(self,field):
        return self.tran[field]

    def fastRead(self,filename, fieldname):
        with open(filename,'rb') as f:
            f.seek(self.offsets[fieldname])
            data = self.readFloats(f)
            return data
      
    def load_data1d(self, field: str, row_or_ring: bool = True, index = 12,**kwargs):
        ot = kwargs.get("ot", False)
        it = kwargs.get("it", False)
        omp = kwargs.get("omp", False)
        imp = kwargs.get("imp", False)
        if ot: index = self.jjtarget[0] +1
        if it: index = self.jjtarget[-1] +1
        if omp: index = self.find_midplane_index()
        if imp: index = self.find_midplane_index(inner=True)
        

        if field in ["JSAT", "jsat"]:
            x0, r0, ne = self.load_data_1d(self, field= "DENEL", row_or_ring= row_or_ring, ot = ot, index= index)
            x0, r0, te = self.load_data_1d(self, field= "TEVE", row_or_ring= row_or_ring, ot = ot, index= index)
            x0, r0, ti = self.load_data_1d(self, field= "TEV", row_or_ring= row_or_ring, ot = ot, index= index)
            jsat = kwargs.get("jsat_factor", 1.0)*ne*np.sqrt(1.6e-19*(te+ti)/(2.014*1.67e-27))*1.6e-19
            return x0, r0, jsat
        cells = []
        if row_or_ring :
            columns = range(self.ni2d)
            for ii in columns :
                k = self.korxy[ii,index-1]
                if k == 0 : continue
                if k > self.np  : continue
                if self.itag[k-1,3] >= 0 : 
                    cells.append(k-1)

        if not row_or_ring :
            for j in range(self.nj[index-1]):
                k = self.kory[index-1,j]
                cells.append(k)

        data = [k+1 for k in cells]
        data = getattr(self,field.lower())[cells]
        
        if row_or_ring :
            x=[];y=[]; r=[]
            isep = 0
            rsep = 0
            zsep = 0
            for ii in range(len(data)):
                k = cells[ii]
                x.append(self.rmesh[k])
                r.append(self.rmesh[k])
                y.append(self.zmesh[k])
                if ii > 0 :
                    k1 = cells[ii-1]
                    if self.ikor[k] == self.ilc :
                        isep = ii
                        rsep = 0.5*(self.rmesh[k] + self.rmesh[k1])
                        zsep = 0.5*(self.zmesh[k] + self.zmesh[k1])
            
            x = [-np.sqrt((x[i]-rsep)**2+(y[i]-zsep)**2) if i<isep else np.sqrt((x[i]-rsep)**2+(y[i]-zsep)**2) for i in range(len(x))]
            r = [-np.sqrt((r[i]-rsep)**2) if i<isep else np.sqrt((r[i]-rsep)**2) for i in range(len(r))]
        elif not row_or_ring :
            # TODO
            # Read/calculate distance along the field line
            x=range(len(cells))
            r = x
        return x, r, data

    def load_data2d(self, param: str):
        data = getattr(self,param.lower())
        field=np.zeros(len(self.nvertp))
        for i in range(len(data)):
            if i >= len(self.korpg) : continue
            if self.korpg[i] > 0:
                field[self.korpg[i]-1]=data[i]
        return field

    def plot1D(self, field, x: np.ndarray= None, row_or_ring: bool = True, index = 12, ax = None, **kwargs): 
        
        if isinstance(field, str):
            x, r, data = self.load_data1d(field, row_or_ring=row_or_ring, index=index, **kwargs)
            
        else:
            # Given user supplied field the user must supply the x vector
            data = field
            x = x

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(x,data,label=self.catid,color='r',marker='+')
        return x, data, ax

    def plot2D(self, param, ax = None, **kwargs):
        """
        Create a 2D plot of the given plasma parameter
        if param is of type str, look for the param in 
        the tran file, else assume that it is a valid
        2D map on the grid.
        """
        if isinstance(param, str):
        
            data = getattr(self,param.lower())
            field=np.zeros(len(self.nvertp))
            for i in range(len(data)):
                if i >= len(self.korpg) : continue
                if self.korpg[i] > 0:
                    field[self.korpg[i]-1]=data[i]
        else:
            field = param

        if ax is None:
            fig, ax = plt.subplots(figsize=(4.5,7.5))
        mesh, pol2k = self.mesh
    
        cmap = plt.get_cmap('plasma', 100)
        if kwargs.get("log", False) :
            if kwargs.get("vmin", False):
                p = PatchCollection(mesh,cmap=cmap,norm=SymLogNorm(linthresh=float(kwargs["log"]), vmin = kwargs["vmin"]))
            else:
                p = PatchCollection(mesh,cmap=cmap,norm=SymLogNorm(linthresh=float(kwargs["log"])))
        else :
            p = PatchCollection(mesh,cmap=cmap)
        p.set_edgecolor('face')
        p.set_array(field)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(p, cax=cax)

        ax.add_collection(p)
        wall = self.wall
        wall.set(color = "black", fill = False)
        ax.add_patch(wall)

        sep = self.sepx
        for i in range(len(sep)):
            ax.plot(sep[i][0],sep[i][1],'black', linewidth = 0.75)
        
        lims=ax.axis('image')
        
        return ax
    
    def find_midplane_index(self, inner = False, midplane_Z = 0.0):
        """
        Finds the midplane index
        inner = False --> outer

        midplane_Z: the z coordinate, which defines the midplane location
        """
        _, _, ztmp =  self.load_data1d('ZMESH', row_or_ring=False, index=self.iopen) 
        midpoint = len(ztmp)//2
        if inner:
            
            ztmp = ztmp[midpoint:-1]

            index = np.argmin(ztmp**2)
            return index + midpoint
        else:
            ztmp = ztmp[:midpoint]

            index = np.argmin(ztmp**2)
            return index 
    
    @property
    def mesh(self):
        if not hasattr(self, '_mesh'):
            self.mesh = None  # Trigger the setter
        return self._mesh, self._pol2k
    
    @mesh.setter
    def mesh(self, value):
        mesh=[]
        pol2k={}
        
        for k in range(self.np):
            if self.korpg[k] != 0 :
                i = self.korpg[k] - 1
                if i not in pol2k : 
                    pol2k[i] = [k]
                else : 
                    pol2k[i].append(k)
                if i > k - self.nj[0] +1 : continue
                points = np.transpose(np.concatenate(([self.rvertp[i*5:i*5+4]], [-self.zvertp[i*5:i*5+4]]), axis=0))
                polygon = Polygon(points, closed=True)
                mesh.append(polygon)
                
        self._mesh = mesh
        self._pol2k = pol2k

    @property
    def wall(self):
        """
        Returns the wall polygon
        """
        if not hasattr(self, '_wall'):
            self.wall = None  # Trigger the setter
        return self._wall
    
    @wall.setter
    def wall(self, value):
        rtmp1=self.rvesm1
        ztmp1=-self.zvesm1
        rtmp2=self.rvesm2
        ztmp2=-self.zvesm2

        # remove non connected segments
        useSegment = np.zeros(len(rtmp1))
        nsegs = 0
        for i in range(len(rtmp1)):
            check = 0
            if i != 0:
                if ((rtmp1[i] == rtmp2[i-1]) and (ztmp1[i] == ztmp2[i-1])) or \
                    ((rtmp2[i] == rtmp1[i-1]) and (ztmp2[i] == ztmp1[i-1])):
                    check = 1
            if i != len(rtmp1)-1:
                if ((rtmp1[i] == rtmp2[i+1]) and (ztmp1[i] == ztmp2[i+1])) or \
                    ((rtmp2[i] == rtmp1[i+1]) and (ztmp2[i] == ztmp1[i+1])):
                    check = 1
            if check:
                useSegment[i] = 1
                nsegs += 1

        wall_poly_pts = []
        for i in range(len(rtmp1)):
            if useSegment[i]:
                wall_poly_pts.append((rtmp1[i],ztmp1[i]))
                wall_poly_pts.append((rtmp2[i],ztmp2[i]))
        wall_poly_pts.append(wall_poly_pts[0]) # connect last point to first to complete wall polygon
        self._wall = Polygon(wall_poly_pts, closed=False, ec='k', lw=2.0, fc='None', zorder=10)
        
    @property
    def sepx(self):
        """
        Returns the separatrix
        """
        if not hasattr(self, '_sepx'):
            self.sepx = None  # Trigger the setter
        return self._sepx

    @sepx.setter
    def sepx(self, value):
        rsepx = self.rsepx
        zsepx = self.zsepx
        points = []
        for i in range(len(rsepx)):
            points.append([rsepx[i], -zsepx[i]])

        self._sepx = np.array(points)
    
    @property
    def sp(self):
        """
        Returns the outer and inner strikepoint coords
        """
        if not hasattr(self, '_sp'):
            self.sp = None  # Trigger the setter
        return self._sp

    @sp.setter
    def sp(self, value):
        _, _, rtmp = self.load_data1d('RMESH', row_or_ring=False, index=self.iopen)
        _, _, ztmp =  self.load_data1d('ZMESH', row_or_ring=False, index=self.iopen) 
        ztmp = -1.0*ztmp
        osp = [rtmp[0], ztmp[0]]
        isp = [rtmp[-1], ztmp[-1]]
        self._sp = (osp, isp)
