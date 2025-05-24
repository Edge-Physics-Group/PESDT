# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import numpy as np
import sys, os

from shapely.geometry import Polygon

from cell import Cell


SIM_INFO_DATA = 0
MESH_DATA = 1
Q = 1.602E-19

# Functions adapted from on cherab-solps, based on a script by Felix Reimold (2016)


INFINITY = 1E99


class SOLPSMesh:
    """
    SOLPSMesh geometry object.

    The SOLPS mesh is rectangular. Each mesh cell is denoted by four vertices with one centre point. Vertices
    may be shared with neighbouring cells. The centre points should be unique.

    Raysect's mesh interpolator uses a different mesh scheme. Mesh cells are triangles and data values are stored at the
    triangle vertices. Therefore, each SOLPS rectangular cell is split into two triangular cells. The data points are
    later interpolated onto the vertex points.

    :param ndarray cr_r: Array of cell vertex r coordinates, must be 3 dimensional. Example shape is (98 x 32 x 4).
    :param ndarray cr_z: Array of cell vertex z coordinates, must be 3 dimensional. Example shape is (98 x 32 x 4).
    :param ndarray vol: Array of cell volumes. Example shape is (98 x 32).
    """

    def __init__(self, cr_r, cr_z, vol):

        self._cr = None
        self._cz = None
        self._poloidal_grid_basis = None

        nx = cr_r.shape[0]
        ny = cr_r.shape[1]
        self._nx = nx
        self._ny = ny

        self._r = cr_r
        self._z = cr_z

        self._vol = vol

        # Iterate through the arrays from MDS plus to pull out unique vertices
        unique_vertices = {}
        vertex_id = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(4):
                    vertex = (cr_r[i, j, k], cr_z[i, j, k])
                    try:
                        unique_vertices[vertex]
                    except KeyError:
                        unique_vertices[vertex] = vertex_id
                        vertex_id += 1

        # Load these unique vertices into a numpy array for later use in Raysect's mesh interpolator object.
        self.num_vertices = len(unique_vertices)
        self.vertex_coords = np.zeros((self.num_vertices, 2), dtype=np.float64)
        for vertex, vertex_id in unique_vertices.items():
            self.vertex_coords[vertex_id, :] = vertex

        # Work out the extent of the mesh.
        rmin = cr_r.flatten().min()
        rmax = cr_r.flatten().max()
        zmin = cr_z.flatten().min()
        zmax = cr_z.flatten().max()
        self.mesh_extent = {"minr": rmin, "maxr": rmax, "minz": zmin, "maxz": zmax}

        # Number of triangles must be equal to number of rectangle centre points times 2.
        self.num_tris = nx * ny * 2
        self.triangles = np.zeros((self.num_tris, 3), dtype=np.int32)

        self._triangle_to_grid_map = np.zeros((nx*ny*2, 2), dtype=np.int32)
        tri_index = 0
        for i in range(nx):
            for j in range(ny):
                # Pull out the index number for each unique vertex in this rectangular cell.
                # Unusual vertex indexing is based on SOLPS output, see Matlab code extract from David Moulton.
                # cell_r = [r(i,j,1),r(i,j,3),r(i,j,4),r(i,j,2)];
                v1_id = unique_vertices[(cr_r[i, j, 0], cr_z[i, j, 0])]
                v2_id = unique_vertices[(cr_r[i, j, 2], cr_z[i, j, 2])]
                v3_id = unique_vertices[(cr_r[i, j, 3], cr_z[i, j, 3])]
                v4_id = unique_vertices[(cr_r[i, j, 1], cr_z[i, j, 1])]

                # Split the quad cell into two triangular cells.
                # Each triangle cell is mapped to the tuple ID (ix, iy) of its parent mesh cell.
                self.triangles[tri_index, :] = (v1_id, v2_id, v3_id)
                self._triangle_to_grid_map[tri_index, :] = (i, j)
                tri_index += 1
                self.triangles[tri_index, :] = (v3_id, v4_id, v1_id)
                self._triangle_to_grid_map[tri_index, :] = (i, j)
                tri_index += 1

        tri_indices = np.arange(self.num_tris, dtype=np.int32)
        # self._tri_index_loopup = Discrete2DMesh(self.vertex_coords, self.triangles, tri_indices)

    @property
    def nx(self):
        """Number of grid cells in the x direction."""
        return self._nx

    @property
    def ny(self):
        """Number of grid cells in the y direction."""
        return self._ny

    @property
    def cr(self):
        """R-coordinate of the cell centres."""
        return self._cr

    @property
    def cz(self):
        """Z-coordinate of the cell centres."""
        return self._cz

    @property
    def vol(self):
        """Volume/area of each grid cell."""
        return self._vol

    @property
    def poloidal_grid_basis(self):
        """
        Array of 2D basis vectors for grid cells.

        For each cell there is a parallel and radial basis vector.

        Any vector on the poloidal grid can be converted to cartesian with the following transformation.
        bx = (p_x  r_x) ( b_p )
        by   (p_y  r_y) ( b_r )

        :return: ndarray with shape (nx, ny, 2) where the two basis vectors are [parallel, radial] respectively.
        """
        return self._poloidal_grid_basis

    @property
    def triangle_to_grid_map(self):
        """
        Array mapping every triangle index to a tuple grid cell ID, i.e. (ix, iy).

        :return: ndarray with shape (nx*ny*2, 2)
        """
        return self._triangle_to_grid_map

    # @property
    # def triangle_index_lookup(self):
    #     """
    #     Discrete2DMesh instance that looks up a triangle index at any 2D point.
    #
    #     Useful for mapping from a 2D point -> triangle cell -> parent SOLPS mesh cell
    #
    #     :return: Discrete2DMesh instance
    #     """
    #     return self._tri_index_loopup

    def __getstate__(self):
        state = {
            'cr_r': self._r,
            'cr_z': self._z,
            'vol': self._vol,
        }
        return state

    def plot_mesh(self):
        """
        Plot the mesh grid geometry to a matplotlib figure.
        """
        fig, ax = plt.subplots()
        patches = []
        for triangle in self.triangles:
            vertices = self.vertex_coords[triangle]
            patches.append(patches.Polygon(vertices, closed=True))
        p = PatchCollection(patches, facecolors='none', edgecolors='b')
        ax.add_collection(p)
        ax.axis('equal')
        return ax

        # Code for plotting vessel geometry if available
        # for i in range(vessel.shape[0]):
        #     plt.plot([vessel[i, 0], vessel[i, 2]], [vessel[i, 1], vessel[i, 3]], 'k')
        # for i in range(vessel.shape[0]):
        #     plt.plot([vessel[i, 0], vessel[i, 2]], [vessel[i, 1], vessel[i, 3]], 'or')

class SOLPS():

    def __init__(self, sim_path, read_fluid_side=True, read_eirene_side=False):

        self.sim_path = sim_path

        self.tri_cells = []
        self.tri_poly = [] 
        
        # Kludge warning! These conform with existing edge2d pyproc data structure.
        # TODO: reorganize this to be edge code agnostic
        self.geom = {}
        self.zch = {} # impurity atomic number list used in pyproc
        
        # Legacy impurity data format used by edge2d - needs to be generalized
        self.imp1_atom_num = None
        self.imp2_atom_num = None
        
        _patches = [] 
        
        self.load_solps_from_raw_output(debug=True)
            
            # Tri centroids
        for i, tri in enumerate(self.mesh.triangles):
            _vert_idx = [tri[0], tri[1], tri[2], tri[0]]
            _vert = self.mesh.vertex_coords[_vert_idx]
            poly = patches.Polygon([(_vert[0]), (_vert[1]), (_vert[2]), (_vert[3])], closed=True)
            _patches.append(poly)
            # create Cell object for each polygon containing the relevant field data
            shply_poly = Polygon(poly.get_xy())
            
            self.tri_poly.append(Polygon(poly.get_xy()))
            
            _idx_grid_map = self.mesh.triangle_to_grid_map[i]
            
            # eV and m^-3 units
            
            _te = self.mesh_data_dict['te'][_idx_grid_map[0], _idx_grid_map[1]]/Q
            _ti = self.mesh_data_dict['ti'][_idx_grid_map[0], _idx_grid_map[1]]/Q     
            _ne = self.mesh_data_dict['ne'][_idx_grid_map[0], _idx_grid_map[1]]       
        
            # Plasma species (fuel + impurity)
            # TODO: Generalize to multi-species - also needs mods on PESDT process side, 
            # which is edge2d-centric with the legacy imp1 and imp2 structure
            # Neutral species densities must be read from fort.44 file, dab2. This is the 
            # EIRENE output on the B2 quad mesh. fort.46 pdena holds the same info on the tri
            # EIRENE mesh.
#                _n0 = self.mesh_data_dict['na'][_idx_grid_map[0], _idx_grid_map[1], 0] # fluid neutral den  
            _n0 = self.fort44_data_dict['dab2'][_idx_grid_map[0], _idx_grid_map[1]] # kinetic atom den  
            _n2 = self.fort44_data_dict['dmb2'][_idx_grid_map[0], _idx_grid_map[1]] # kinetic mol. den 
            _n2p = self.fort44_data_dict['dib2'][_idx_grid_map[0], _idx_grid_map[1]] # kinetic mol. ion den
            _ni = self.mesh_data_dict['na'][_idx_grid_map[0], _idx_grid_map[1], 1] # fuel ion den      
            _imp1_den = self.mesh_data_dict['na'][_idx_grid_map[0], _idx_grid_map[1], 2:] # impurity den by ion stage
            self.imp1_atom_num = None
            
#                for i in range(len(sim_info_dict['zn'])):
#                    zn = int(sim_info_dict['zn'][i])  # Nuclear charge
#                    am = float(sim_info_dict['am'][i])  # Atomic mass
#                    charge = int(sim_info_dict['zamax'][i])  # Ionisation/charge
#                    species = _popular_species[(zn, am)]
#                    sim.species_list.append(species.symbol + str(charge))
            
            
            # TODO: read-in equlibirium and populate x-pt, osp, isp, wall_poly, sep_poly
            self.geom['rpx'] = 2.563
            self.geom['zpx'] = -1.449
            self.zch = {}
                                    
            self.tri_cells.append(Cell(shply_poly.centroid.x, shply_poly.centroid.y,
                                        row=_idx_grid_map[0], ring=_idx_grid_map[1],                                       
                                        poly=shply_poly, te=_te, ti = _ti,
                                        ne=_ne, ni=_ni,
                                        n0=_n0, n2=_n2, n2p=_n2p, Srec=0, Sion=0))

        # Conform to PESDT data format 
        self.cells = self.tri_cells
        self.ne = []
        self.n0 = []
        self.n2 = []
        self.n2p = []
        self.ni = []
        self.te = []
        for cell in self.cells:
            self.ne.append(cell.ne)   
            self.n0.append(cell.n0)
            self.n2.append(cell.n2)
            self.n2p.append(cell.n2p)
            self.ni.append(cell.ni)   
            self.te.append(max(cell.te,0.1))  
        
            print()
            
    # Code based on script by Felix Reimold (2016)
    def load_solps_from_raw_output(self, debug=False):
        """
        Load a SOLPS simulation from raw SOLPS output files.
    
        Required files include:
        * mesh description file (b2fgmtry)
        * B2 plasma state (b2fstate)
        * Eirene output file on B2 triangular grid (fort.44)
    
        :param str simulation_path: String path to simulation directory.
        :rtype: SOLPSSimulation
        """
    
        if not os.path.isdir(self.sim_path):
            RuntimeError("simulation_path must be a valid directory")
    
        mesh_file_path = os.path.join(self.sim_path, 'b2fgmtry')
        b2_state_file = os.path.join(self.sim_path, 'b2fstate')
        eirene_fort44_file = os.path.join(self.sim_path, "fort.44")
    
        if not os.path.isfile(mesh_file_path):
            raise RuntimeError("No B2 b2fgmtry file found in SOLPS output directory")
    
        if not(os.path.isfile(b2_state_file)):
            RuntimeError("No B2 b2fstate file found in SOLPS output directory")
    
        if not(os.path.isfile(eirene_fort44_file)):
            RuntimeError("No EIRENE fort.44 file found in SOLPS output directory")
    
        # Load SOLPS mesh geometry
        self.mesh = self.load_mesh_from_files(mesh_file_path=mesh_file_path, debug=debug)
#        self.mesh.plot_mesh()

        self.fort44_header_dict, self.fort44_info_dict, self.fort44_data_dict = self.load_fort44_file(eirene_fort44_file, debug=debug)
        self.header_dict, self.sim_info_dict, self.mesh_data_dict = self.load_b2f_file(b2_state_file, debug=debug)
    
    

    def load_b2f_file(self, filepath, debug=False):
        """
        File for parsing the 'b2fstate' B2 Eirene output file.
    
        :param str filepath: full path for file to load and parse
        :param bool debug: flag for displaying textual debugging information.
        :return: tuple of dictionaries. First is the header information such as the version, label, grid size, etc. Second
        dictionary has a SOLPSData object for each piece of data found in the file.
        """
    
        # Inline function for mapping str data to floats, reshaping arrays, and loading into SOLPSData object.
        def _make_solps_data_object(_data):
    
            # Convert list of strings to list of floats
            for idx, item in enumerate(_data):
                _data[idx] = float(item)
    
            # Multiple 2D data field (e.g. na)
            if number > nxyg:
                _data = np.array(_data).reshape((nxg, nyg, int(number / nxyg)), order='F')
                if debug:
                    print('Mesh data field {} with dimensions:  {:d} x {:d} x {:d}'.format(name, nxg, nyg, int(number/nxyg)))
                return MESH_DATA, _data
    
            # 2D data field (e.g. ne)
            elif number == nxyg:
                _data = np.array(_data).reshape((nxg, nyg), order='F')
                if debug:
                    print('Mesh data field {} with dimensions:  {:d} x {:d}'.format(name, nxg, nyg))
                return MESH_DATA, _data
    
            # Additional information field (e.g. zamin)
            else:
                _data = np.array(_data)
                if debug:
                    print('Sim info field {} with length:     {} '.format(name, _data.shape[0]))
                return SIM_INFO_DATA, _data
    
        if not(os.path.isfile(filepath)):
            raise IOError('File {} not found.'.format(filepath))
    
        # Open SOLPS geometry file for reading
        fh = open(filepath, 'r')
    
        # Version header
        version = fh.readline()
    
        # Read mesh size
        fh.readline()
        line = fh.readline().split()
        nx = int(line[0])
        ny = int(line[1])
    
        # Calculate guard cells
        nxg = nx + 2
        nyg = ny + 2
    
        # Flat vector size
        nxy = nx * ny
        nxyg = nxg * nyg
    
        # Read Label
        fh.readline()
        label = fh.readline()
    
        # Save header
        header_dict = {'version': version, 'nx': nx, 'ny': ny, 'nxg': nxg, 'nyg': nyg, 'nxy': nxy, 'nxyg': nxyg, 'label': label}
    
        # variables for file data
        name = ''
        number = 0
        data = []
        sim_info_dict = {}
        mesh_data_dict = {}
    
        # Repeat reading data blocks till EOF
        while True:
            # Read line
            line = fh.readline().split()
    
            # EOF --OR--  New block of similar data (vector qty, e.g. bb)
            if len(line) == 0:
                # Check if last line
                line = fh.readline().split()
                if len(line) == 0:
                    break
    
            # New block found
            if line[0] == '*cf:':
    
                # If previous block read --> Convert data to float, reshape and save to Object
                if name != '':
                    flag, shaped_data = _make_solps_data_object(data)
                    if flag == SIM_INFO_DATA:
                        sim_info_dict[name] = shaped_data
                    elif flag == MESH_DATA:
                        mesh_data_dict[name] = shaped_data
    
                # Read next field paramters
                data_type = str(line[1].strip())
                number = int(line[2])
                name = str(line[3].strip())
                data = []
    
            # Append line to vector of data
            else:
                data.extend(line)
    
        if name != '':
            flag, shaped_data = _make_solps_data_object(data)
            if flag == SIM_INFO_DATA:
                sim_info_dict[name] = shaped_data
            elif flag == MESH_DATA:
                mesh_data_dict[name] = shaped_data
    
        return header_dict, sim_info_dict, mesh_data_dict



    def load_fort44_file(self, filepath, debug=False):
        """
        File for parsing the 'fort.44' B2 Eirene output file.
    
        :param str filepath: full path for file to load and parse
        :param bool debug: flag for displaying textual debugging information.
        :return: tuple of dictionaries. First is the header information such as the version, label, grid size, etc. Second
        dictionary has a SOLPSData object for each piece of data found in the file.
        """
    
        # Inline function for mapping str data to floats, reshaping arrays, and loading into SOLPSData object.
        def _make_solps_data_object(_data): 

            # Convert list of strings to list of floats
            for idx, item in enumerate(_data):
                _data[idx] = float(item)

            # Multiple 2D data field (e.g. dab2)
            if number > nxyg:
                _data = np.array(_data).reshape((nxg, nyg, int(number / nxyg)), order='F')
                if debug:
                    print('Mesh data field {} with dimensions:  {:d} x {:d} x {:d}'.format(name, nxg, nyg, int(number/nxyg)))
                return MESH_DATA, _data
    
            # 2D data field (e.g. ne)
            elif number == nxyg:
                _data = np.array(_data).reshape((nxg, nyg), order='F')
                if debug:
                    print('Mesh data field {} with dimensions:  {:d} x {:d}'.format(name, nxg, nyg))
                return MESH_DATA, _data
    
            # Additional information field (e.g. zamin)
            else:
                _data = np.array(_data)
                if debug:
                    print('Sim info field {} with length:     {} '.format(name, _data.shape[0]))
                return SIM_INFO_DATA, _data
    
        if not(os.path.isfile(filepath)):
            raise IOError('File {} not found.'.format(filepath))
    
        # Open fort.44 file for reading
        fh = open(filepath, 'r')
    
        # Version header
        version = fh.readline()
    
        # Read mesh size
        line = version.split()
        nx = int(line[0])
        ny = int(line[1])
        date = int(line[2])
        version = line[3]
    
        # Calculate guard cells
        nxg = nx + 2
        nyg = ny + 2
    
        # Flat vector size
        nxy = nx * ny
        nxyg = nxg * nyg
    
        # Read num species
        line = fh.readline().split()
        num_atomic_species = int(line[0])
        num_mol_species = int(line[1])
        num_testion_species = int(line[2])
        
        atomic_species_label = []
        for _i in range(num_atomic_species):
            atomic_species_label.append(fh.readline())
            
        mol_species_label = []
        for _i in range(num_mol_species):
            mol_species_label.append(fh.readline())
            
        testion_species_label = []
        for _i in range(num_testion_species):
            testion_species_label.append(fh.readline())

    
        # Save header
        fort44_header_dict = {'version': version, 'nx': nx, 'ny': ny, 
                       'nxg': nxg, 'nyg': nyg, 'nxy': nxy, 'nxyg': nxyg, 
                       'num_atomic_species': num_atomic_species,
                       'num_mol_species': num_mol_species,
                       'num_testion_species': num_testion_species,
                       'atomic_species_label': atomic_species_label,
                       'mol_species_label': mol_species_label,
                       'testion_species_label': testion_species_label
                       }
    
        # variables for file data
        name = ''
        number = 0
        data = []
        fort44_data_dict = {}
        fort44_info_dict = {}
    
        # Repeat reading data blocks till EOF
        while True:
            # Read line
            line = fh.readline().split()
    
            # EOF 
            if len(line) == 0:
                # Check if last line
                line = fh.readline().split()
                if len(line) == 0:
                    break
    
            # New block found
            if line[0] == '*eirene':
                # If previous block read --> Convert data to float, reshape and save to Object
                if name != '':
                    data.extend((nxg+1)*[0.0]) # Add last row of guard cells
                    flag, shaped_data = _make_solps_data_object(data)
                    if flag == SIM_INFO_DATA:
                        fort44_info_dict[name] = shaped_data                    
                    elif flag == MESH_DATA:
                        fort44_data_dict[name] = shaped_data
    
                # Read next field paramters
                data_type = 'real'
                number = int(line[6]) 
                if number >= nxy: # adding guard cells
                    number = number + (2*nxg+2*ny)*(number/nxy)           
                name = str(line[3].strip())
                if name == 'edissml': # you can stop reading here, because other variables are not needed
                    break
                data = (nxg+1)*['0.0']
                ndata = 0
                nrow = 1
    
            # Append line to vector of data
            else:
                for i in range(len(line)):
                    ndata = ndata+1
                    if ndata > nrow*nx: # add guard cells
                       nrow = nrow+1
                       data.extend(['0.0'])
                       data.extend(['0.0'])
                    data.extend([line[i]])
    
        if name != '' and name != 'edissml':
            data.extend((nxg+1)*[0.0]) # Add last row of guard cells
            flag, shaped_data = _make_solps_data_object(data)
            if flag == SIM_INFO_DATA:
                fort44_info_dict[name] = shaped_data
            elif flag == MESH_DATA:
                fort44_data_dict[name] = shaped_data
    
        return fort44_header_dict, fort44_info_dict, fort44_data_dict
    
    
    def load_mesh_from_files(self,mesh_file_path, debug=False):
        """
        Load SOLPS grid description from B2 Eirene output file.
    
        :param str filepath: full path for B2 eirene mesh description file
        :param bool debug: flag for displaying textual debugging information.
        :return: tuple of dictionaries. First is the header information such as the version, label, grid size, etc.
          Second dictionary has a ndarray for each piece of data found in the file.
        """
        _, _, geom_data_dict = self.load_b2f_file(mesh_file_path, debug=debug)
    
        cr_x = geom_data_dict['crx']
        cr_z = geom_data_dict['cry']
        vol = geom_data_dict['vol']
    
        # build mesh object
        return SOLPSMesh(cr_x, cr_z, vol)


# Test case
if __name__ == '__main__':

    simulation_path = "/home/5bl/PESDT/examples/mastu_solps_dmoulton/puff=40.0e21_pump=0.001_drifts_divchiconstfix_bcmom2_visper1_rxf0p1_parmvsa2_pin2.0_redoutpfrtrans"

    # p = load_b2f_file(simulation_path + "/b2fstate", debug=True)

    SOLPS(simulation_path)

    print()

    plt.show()
