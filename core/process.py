
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import pickle
import json
import struct

import eproc as ep
from matplotlib.collections import PatchCollection
from matplotlib import patches
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, LineString
from PESDT.machine_defs import get_DIIIDdefs, get_JETdefs
from PESDT.pyADASread import adas_adf11_read, adas_adf15_read, continuo_read
from PESDT.edge_code_formats.edge2d_format import Edge2D, Cell
from PESDT.edge_code_formats.solps_format import SOLPS
from PESDT.cherab_bridge.cherab_plasma import CherabPlasma

import amread


at_sym = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg',
          'Al','SI','P','S','CL','AR','K','CA','SC','TI','V','CR',
          'MN','FE','CO','NI','CU','ZN','GA','GE','AS','SE','BR',
          'KR','RB','SR','Y','ZR','NB','MO','TC','RU','RH','PD',
          'AG','CD','IN','SN','SB','TE','I','XE','CS','BA','LA',
          'CE','PR','ND','PM','SM','EU','GD','TB','DY','HO','ER',
          'TM','YB','LU','HF','TA','W','RE','OS','IR','PT','AU',
          'HG','TL','PB','BI','PO','AT','RN','FR','RA','AC','TH','P']

roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

def floatToBits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>l', s)[0]

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def gaussian(cwl, wv, area, fwhm):
    sigma = fwhm / (2. * np.sqrt(2 * np.log(2.)) )
    g = area * (1./(sigma*np.sqrt(2.*np.pi))) * np.exp(-1.*((wv - cwl)**2) / (2*(sigma**2)) )
    return g

def interp_nearest_neighb(point, neighbs, neighbs_param_pervol):
    """
        Nearest neigbours weighted average of given parameter with per unit volume units
        point = [r,z]
        neighbs = [[r1,z1,[rn,zn]]
        neighbs_param_pervol = [val1, ...valn]

        returns point_param_pervol, the weighted nearest neighbours average
    """
    distances = []
    for neighb in neighbs:
        distances.append(np.sqrt((point[0]-neighb[0])**2 + (point[1]-neighb[1])**2))

    numerator = 0
    denominator = 0
    for i in range(len(neighbs_param_pervol)):
        numerator += distances[i]*neighbs_param_pervol[i]
        denominator += distances[i]

    point_param_pervol = numerator / denominator

    return point_param_pervol

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class Region:

    def __init__(self, name, Rmin, Rmax, Zmin, Zmax, include_confined=True, include_SOL=True, include_PFR=True):
        self.name = name
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.include_confined = include_confined
        self.include_SOL = include_SOL
        self.include_PFR= include_PFR

        # container of cell objects belonging to the region
        self.cells = []

        # Total radiated power - main ions
        self.Prad_H = 0.0
        self.Prad_H_Lytrap = 0.0
        self.Prad_units = 'W'

        # Main ion ionization and recombination [units: s^-1]
        self.Sion = 0.0
        self.Srec = 0.0

    def cell_in_region(self, cell, shply_sep_poly):

        if cell.Z >= self.Zmin and cell.Z <= self.Zmax and cell.R >= self.Rmin and cell.R <= self.Rmax:
            if self.include_confined and shply_sep_poly.contains(cell.poly):
                return True
            if self.include_SOL and not shply_sep_poly.contains(cell.poly):
                return True
            if self.include_PFR and shply_sep_poly.contains(cell.poly):
                return True

        return False
    
class ProcessEdgeSim:
    '''
    Class to read and store EDGE2D-EIRENE results

    First all properties to be loaded are initialized as "None"
    Currently the code is very edge2d spesific, i.e. reading SOLPS does not actually work

    '''
    def __init__(self, ADAS_dict, edge_code_defs, use_AMJUEL = False, AMJUEL_date = 2016, ADAS_dict_lytrap=None,
                 machine='JET', pulse=90531, spec_line_dict=None, spec_line_dict_lytrap = None, 
                 diag_list=None, calc_synth_spec_features=None, save_synth_diag=False,
                 synth_diag_save_file=None, data2d_save_file=None, outlier_cell_dict=None, recalc_h2_pos=True, 
                 run_cherab = False, input_dict = None, **kwargs):

        self.ADAS_dict = ADAS_dict
        self.AMJUEL_date = AMJUEL_date
        self.use_AMJUEL = use_AMJUEL
        self.ADAS_dict_lytrap = ADAS_dict_lytrap
        self.spec_line_dict = spec_line_dict
        self.spec_line_dict_lytrap = spec_line_dict_lytrap
        self.edge_code = edge_code_defs['code']
        self.sim_path = edge_code_defs['sim_path']
        self.machine = machine
        self.pulse = pulse
        self.recalc_h2_pos = recalc_h2_pos
        self.input_dict = input_dict
        self.regions = {}

        self.cells = None
        # cells dataframe for easy filtering by edge_codes row/ring
        self.cells_df = None

        #Flags

        self.geom = None
        self.teve = None
        self.den = None
        self.denel = None
        self.da = None
        self.korpg = None
        self.rmesh = None
        self.rvertp = None
        self.zmesh = None
        self.zvertp = None
        self.NE2Ddata = None
        self.patches = None
        self.H_adf11 = None

        self.sep_poly = None
        self.shply_sep_poly = None
        self.sep_poly_below_xpt = None
        self.shply_sep_poly_below_xpt = None

        self.wall_poly = None
        self.shply_wall_poly = None

        self.rv = None
        self.zv = None

        # variables mapped onto edge_codes grid
        self.te = None
        self.ne = None
        self.ni = None
        self.n0 = None
        self.n2 = None
        self.n2p = None

        # dictionary of outlier cells and their specified neighbours for interpolation
        self.outlier_cell_dict = outlier_cell_dict
        # Dictionary for storing synthetic diagnostic objects
        self.synth_diag = {}
        
        # New (04-Mar-2021) way of reading edge code data
        if self.edge_code == 'edge2d':
            data = Edge2D(self.sim_path)
            #self.cells = self.data.quad_cells
            # For compatability reasons, copy everything over manually.
            # If we want to continue to use separate "<sol_code>_format.py" files
            # to read results, the best way moving forward would be to just use self.data.
            # This, however, would require changes to multiple places in the code, which 
            # I don't have the time for right now. - V.-P Rikala
            self.geom = data.geom
            self.teve = data.teve
            self.den = data.den
            self.denel = data.denel
            self.da = data.da
            self.korpg = data.korpg
            self.rmesh = data.rmesh
            self.rvertp = data.rvertp
            self.zmesh = data.zmesh
            self.zvertp = data.zvertp
            self.NE2Ddata = data.NE2Ddata
            self.patches = data.patches
            #self.H_adf11 = self.data.H_adf11

            self.sep_poly = data.sep_poly
            self.shply_sep_poly = data.shply_sep_poly
            self.sep_poly_below_xpt = data.sep_poly_below_xpt
            self.shply_sep_poly_below_xpt = data.shply_sep_poly_below_xpt

            self.wall_poly = data.wall_poly
            self.shply_wall_poly = data.shply_wall_poly

            self.rv = data.rv
            self.zv = data.zv
            
            self.osp = data.osp
            self.isp = data.isp

            # variables mapped onto edge_codes grid
            self.te = data.te
            self.ne = data.ne
            self.ni = data.ni
            self.n0 = data.n0
            self.n2 = data.n2
            self.n2p = data.n2p

            self.cells = data.cells

            self.data = data      
        elif self.edge_code == 'solps':
            self.data = SOLPS(self.sim_path)
            self.cells = self.data.tri_cells     

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
        
        # Interpolate any outlier cells using specified neighbours R,Z coords. Interpolated cell's plasma properties
        # are averages of its neighbours weighted by the distance between centroids.
        # self.interp_outlier_cells()

        # Get machine definitions
        if self.machine == 'JET':
            self.defs = get_JETdefs(pulse_ref=self.pulse)
            if self.edge_code == 'edge2d':
                # Define regions for JET. Need to get separatrix polygon from solps.
                self.Zdiv = -1.2
                self.regions['vessel'] = Region('vessel', Rmin=0, Rmax=10, Zmin=-10, Zmax=10,
                                        include_confined=True, include_SOL=True, include_PFR=True)
    
                self.regions['hfs_sol'] = Region('hfs_sol', Rmin=0, Rmax=self.data.geom['rpx'], Zmin=self.Zdiv, Zmax=10,
                                        include_confined=False, include_SOL=True, include_PFR=False)
                self.regions['lfs_sol'] = Region('lfs_sol', Rmin=self.data.geom['rpx'], Rmax=10, Zmin=self.Zdiv, Zmax=10,
                                        include_confined=False, include_SOL=True, include_PFR=False)
                self.regions['hfs_div'] = Region('hfs_div', Rmin=0, Rmax=self.data.geom['rpx'], Zmin=-10, Zmax=self.Zdiv,
                                        include_confined=False, include_SOL=True, include_PFR=False)
                self.regions['lfs_div'] = Region('lfs_div', Rmin=self.data.geom['rpx'], Rmax=10, Zmin=-10, Zmax=self.Zdiv,
                                        include_confined=False, include_SOL=True, include_PFR=False)
                self.regions['xpt_conreg'] = Region('xpt_conreg', Rmin=2.3, Rmax=2.95, Zmin=self.data.geom['zpx'], Zmax=-0.80,
                                        include_confined=True, include_SOL=False, include_PFR=False)
                self.regions['hfs_lower'] = Region('hfs_lower', Rmin=0, Rmax=self.data.geom['rpx'], Zmin=-10, Zmax=self.Zdiv,
                                        include_confined=True, include_SOL=True, include_PFR=True)
                self.regions['lfs_lower'] = Region('lfs_lower', Rmin=self.data.geom['rpx'], Rmax=10, Zmin=-10, Zmax=self.Zdiv,
                                        include_confined=True, include_SOL=True, include_PFR=True)
                self.regions['rhon_09_10'] = Region('rhon_09_10', Rmin=0, Rmax=10, Zmin=self.data.geom['zpx'], Zmax=10,
                                        include_confined=True, include_SOL=False, include_PFR=False)
        elif self.machine == 'DIIID' and self.edge_code == 'edge2d':
            self.defs = get_DIIIDdefs()

        # TODO: Add opacity calcs
        if run_cherab:
            # Currently the run cherab function uses the synth_diag to get the instrument and LOS details, so that needs to be generated
            self.run_cherab_bridge()
        else:
            
            self.calc_H_emiss()
            self.calc_H_rad_power()
            self.calc_ff_fb_emiss()

            # Calc ff+fb emissivity.
            # Use KL11 cam e WI filter.
            # TODO: generalise to accept other filters
            #filter_file = '/home/bloman/python_bal/TomI/kl11_filters/WI40096_kl11_filter_peak_norm.txt'
            #filter_curve = np.genfromtxt(filter_file, delimiter=',', skip_header=4)
            #print('FF+FB emission filter file: ', filter_file)
            #self.calc_ff_fb_filtered_emiss(filter_curve[:,0], filter_curve[:,1])


            # CALULCATE PRAD IN DEFINED MACRO REGIONS
            # TODO: Add opacity calcs
            #if self.regions:
            #    self.calc_region_aggregates()

            # CALCULATE POWER FLOW INTO INNER AND OUTER DIVERTOR AT Z=-1.2
            #if self.edge_code == 'edge2d':
            #    self.calc_qpol_div()

            if diag_list:
                print('diag_list', diag_list)
                for key in diag_list:
                    if key in self.defs.diag_dict.keys():
                        self.synth_diag[key] = SynthDiag(self.defs, diag=key,
                                                        spec_line_dict = self.spec_line_dict,
                                                        spec_line_dict_lytrap=self.spec_line_dict_lytrap, 
                                                        use_AMJUEL = self.use_AMJUEL)
                        for chord in self.synth_diag[key].chords:
                            # Basic LOS implementation using 2D polygons - no reflections
                            self.los_intersect(chord)
                            chord.orthogonal_polys()
                            if self.use_AMJUEL:
                                chord.calc_int_and_1d_los_quantities_AMJUEL_1()
                            else:
                                chord.calc_int_and_1d_los_quantities_1()
                            if calc_synth_spec_features:
                                # Derived ne, te require information along LOS, calc emission again using _2 functions
                                if self.use_AMJUEL:
                                    chord.calc_int_and_1d_los_quantities_AMJUEL_2()
                                else:
                                    chord.calc_int_and_1d_los_quantities_2()
                                print('Calculating synthetic spectra for diag: ', key)
                                chord.calc_int_and_1d_los_synth_spectra()

            if save_synth_diag:
                if self.synth_diag:
                    self.save_synth_diag_data(savefile=synth_diag_save_file)

        if data2d_save_file:
            # pickle serialization of e2deirpostproc object
            output = open(data2d_save_file, 'wb')
            pickle.dump(self, output)
            output.close()

    def run_cherab_bridge(self): 
        # Inputs from cherab_bridge_input_dict
        import_jet_surfaces = self.input_dict['cherab_options']['import_jet_surfaces']
        include_reflections = self.input_dict['cherab_options']['include_reflections']
        spectral_bins = self.input_dict['cherab_options']['spectral_bins']
        pixel_samples = self.input_dict['cherab_options']['pixel_samples']
        spec_line_dict = self.input_dict['spec_line_dict']
        diag_list = self.input_dict['diag_list']
        use_AMJUEL = self.input_dict['run_options']['use_AMJUEL']
        recalc_h2_pos = self.input_dict['run_options'].get("recalc_h2_pos", True)
        stark_transition = self.input_dict['cherab_options'].get('stark_transition', False)
        ff_fb = self.input_dict['cherab_options'].get('ff_fb_emission', False)
        #sion_H_transition = input_dict['cherab_options']['Sion_H_transition']
        #srec_H_transition = input_dict['cherab_options']['Srec_H_transition']
        # Generate cherab plasma
        plasma = CherabPlasma(self, self.ADAS_dict, include_reflections = include_reflections,
                            import_jet_surfaces = import_jet_surfaces, use_AMJUEL=use_AMJUEL, recalc_h2_pos = recalc_h2_pos)

        # Create output dict
        self.outdict = {}

        diag_dict = get_JETdefs().diag_dict

        # Loop through diagnostics, their LOS, integrate over Lyman/Balmer
        for diag_key in diag_list:
            self.outdict[diag_key] = {}
            for diag_chord, los_p2 in enumerate(diag_dict[diag_key]['p2']):
                los_p2 = los_p2.tolist()
                los_p1 = diag_dict[diag_key]['p1'][0].tolist()
                #print(los_p1)
                #print(los_p2)
                los_w1 = 0.0 # unused
                los_w2 = diag_dict[diag_key]['w'][0][1]
                #print(los_w2)
                H_lines = spec_line_dict['1']['1']

                self.outdict[diag_key][str(diag_chord)] = {
                    'chord':{'p1':los_p1, 'p2':los_p2, 'w1':los_w1, 'w2':los_w2}
                }
                self.outdict[diag_key][str(diag_chord)]['spec_line_dict'] = spec_line_dict

                self.outdict[diag_key][str(diag_chord)]['los_int'] = {'H_emiss': {}}

                print(diag_key, los_p2)
                for H_line_key, val in H_lines.items():
                
                    transition = (int(val[0]), int(val[1]))
                    wavelength = float(H_line_key)/10. #nm
                    min_wavelength = (wavelength)-1.0
                    max_wavelength = (wavelength)+1.0

                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                            include_excitation=True, include_recombination=False, use_AMJUEL=use_AMJUEL)
                    exc_radiance, wave = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, #, exc_spectrum,
                                                                            min_wavelength, max_wavelength,
                                                                            spectral_bins=spectral_bins, pixel_samples=pixel_samples)

                    plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                            include_excitation=False, include_recombination=True, use_AMJUEL=use_AMJUEL)

                    rec_radiance, wave = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, # rec_spectrum,
                                                                            min_wavelength, max_wavelength,
                                                                            spectral_bins=spectral_bins, pixel_samples=pixel_samples)
                    if use_AMJUEL:
                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                include_excitation=False, include_H2=True, use_AMJUEL=use_AMJUEL)

                        h2_radiance, wave = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, # H2_spectrum,
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spectral_bins, pixel_samples=pixel_samples)

                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                include_excitation=False, include_H2_pos= True, use_AMJUEL=use_AMJUEL)

                        h2_pos_radiance, wave = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, # H2+_spectrum,
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spectral_bins, pixel_samples=pixel_samples)
                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                include_excitation=False, include_H_neg=True, use_AMJUEL=use_AMJUEL)

                        h_neg_radiance, wave = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, # H-_spectrum,
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spectral_bins, pixel_samples=pixel_samples)
                                                            
                        self.outdict[diag_key][str(diag_chord)]['los_int']['H_emiss'][H_line_key] = {
                            'excit':(np.array(exc_radiance)).tolist(),
                            'recom':(np.array(rec_radiance)).tolist(),
                            'h2': (np.array(h2_radiance)).tolist(),
                            'h2+': (np.array(h2_pos_radiance)).tolist(),
                            'h-': (np.array(h_neg_radiance)).tolist(),
                            'units':'ph.s^-1.m^-2.sr^-1'
                        }
                    else:
                        self.outdict[diag_key][str(diag_chord)]['los_int']['H_emiss'][H_line_key] = {
                            'excit':(np.array(exc_radiance)).tolist(),
                            'recom':(np.array(rec_radiance)).tolist(),
                            'units':'ph.s^-1.m^-2.sr^-1'
                        }

                    if stark_transition:
                        if transition == tuple(stark_transition):
                            print('Stark transition')
                            plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                    include_excitation=True, include_recombination=True, 
                                                    include_H2_pos= True, include_H2=True, include_H_neg=True, use_AMJUEL=use_AMJUEL,
                                                    include_stark=True)
                            spec_bins = 50
                            radiance,  wave_arr = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, #spectrum,
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spec_bins,
                                                                                pixel_samples=pixel_samples,
                                                                                display_progress=False,no_avg = True)

                            self.outdict[diag_key][str(diag_chord)]['los_int']['stark']={'cwl': wavelength, 'wave': (np.array(wave_arr)).tolist(),
                                                                            'intensity': (np.array(radiance)).tolist(),
                                                                            'units': 'nm, ph s^-1 m^-2 sr^-1 nm^-1'}

                    # Free-free + free-bound using adaslib/continuo
                if ff_fb:
                    plasma.define_plasma_model(atnum=1, ion_stage=0,
                                                include_excitation=False, include_recombination=False, use_AMJUEL=use_AMJUEL,
                                                include_stark=False, include_ff_fb=True)
                    min_wave = 300
                    max_wave = 500
                    spec_bins = 50
                    radiance,  wave_arr = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2, #spectrum,
                                                                            min_wave, max_wave,
                                                                            spectral_bins=spec_bins,
                                                                            pixel_samples=pixel_samples,
                                                                            display_progress=False, no_avg= True)

                    self.outdict[diag_key][str(diag_chord)]['los_int']['ff_fb_continuum'] = {
                            'wave': (np.array(wave_arr)).tolist(),
                            'intensity': (np.array(radiance)).tolist(),
                            'units': 'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
        #return outdict

    def __getstate__(self):
        """
            For removing the large ADAS_dict from the object for pickling
            See: https://docs/python/org/2/library/pickle.html#example
        """
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['ADAS_dict']
        return odict

    def __setstate__(self, dict):
        # TODO: Read external ADAS_dict object and add to dict for unpickling
        self.__dict__.update(dict)

    def interp_outlier_cells(self):
        """
            Replace outlier cell plasma properties with interpolated values from nearest specified neighbours.
            Uses average of neighbouring cell values weighted by distance to these cells.
        """
        for outlier_cell in self.outlier_cell_dict:
            for cell in self.cells:
                if isclose(cell.R, outlier_cell['outlier_RZ'][0], rel_tol=1e-09, abs_tol=rz_match_tol) and \
                        isclose(cell.Z, outlier_cell['outlier_RZ'][1], rel_tol=1e-09, abs_tol=rz_match_tol):
                    R = cell.R
                    Z = cell.Z
                    neighbs_RZ = []
                    neighb_ne = []
                    neighb_ni = []
                    neighb_n0 = []
                    neighb_n2 = []
                    neighb_Te = []
                    neighb_Ti = []
                    for i, neighb in enumerate(outlier_cell['neighbs_RZ']):
                        for _cell in self.cells:
                            if isclose(_cell.R, neighb[i][0], rel_tol=1e-09, abs_tol=rz_match_tol) and \
                                    isclose(_cell.Z, neighb[i][1], rel_tol=1e-09, abs_tol=rz_match_tol):
                                neighbs_RZ.append([_cell.R, _cell.Z])
                                neighb_ne.append(_cell.ne)
                                neighb_ni.append(_cell.ni)
                                neighb_n0.append(_cell.n0)
                                neighb_n2.append(_cell.n2)
                                neighb_Te.append(_cell.Te)
                                neighb_Ti.append(_cell.Ti)
                                break

                    # now interpolate
                    cell.ne = interp_nearest_neighb([R,Z], neighbs_RZ, neighb_ne)
                    cell.ni = interp_nearest_neighb([R,Z], neighbs_RZ, neighb_ni)
                    cell.n0 = interp_nearest_neighb([R,Z], neighbs_RZ, neighb_n0)
                    cell.n2 = interp_nearest_neighb([R,Z], neighbs_RZ, neighb_n2)
                    cell.Te = interp_nearest_neighb([R,Z], neighbs_RZ, neighb_Te)
                    cell.Ti = interp_nearest_neighb([R,Z], neighbs_RZ, neighb_Ti)
                break
    

    def save_synth_diag_data(self, savefile=None):
        # output = open(savefile, 'wb')
        outdict = {}
        for diag_key in self.synth_diag:
            outdict[diag_key] = {}
            for chord in self.synth_diag[diag_key].chords:
                outdict[diag_key].update({chord.chord_num:{}})
                # outdict[diag_key][chord.chord_num].update({'H_emiss':chord.los_int['H_emiss']})
                outdict[diag_key][chord.chord_num]['spec_line_dict'] = self.spec_line_dict
                outdict[diag_key][chord.chord_num]['spec_line_dict_lytrap'] = self.spec_line_dict_lytrap
                outdict[diag_key][chord.chord_num]['los_1d'] = chord.los_1d
                outdict[diag_key][chord.chord_num]['los_int'] = chord.los_int
                for spectrum in chord.los_int_spectra:
                    outdict[diag_key][chord.chord_num]['los_int'].update({spectrum:chord.los_int_spectra[spectrum]})
                    # outdict[diag_key][chord.chord_num]['los_1d'].update({spectrum: chord.los_1d_spectra[spectrum]})
                # outdict[diag_key][chord.chord_num]['Srec'] = chord.los_int['Srec']
                # outdict[diag_key][chord.chord_num]['Sion'] = chord.los_int['Sion']
                if chord.shply_intersects_w_sep and diag_key=='KT3':
                    outdict[diag_key][chord.chord_num].update({'chord':{'p1':chord.p1, 'p2':chord.p2unmod, 'w1': chord.w1,'w2':chord.w2unmod, 'sep_intersect_below_xpt':[chord.shply_intersects_w_sep.coords.xy[0][0],chord.shply_intersects_w_sep.coords.xy[1][0]]}})
                else:
                    outdict[diag_key][chord.chord_num].update({'chord':{'p1':chord.p1, 'p2':chord.p2unmod, 'w1': chord.w1,'w2':chord.w2unmod, 'sep_intersect_below_xpt':None}})
                if chord.los_angle:
                    outdict[diag_key][chord.chord_num]['chord']['los_angle'] = chord.los_angle

        # SAVE IN JSON FORMAT TO ENSURE PYTHON 2/3 COMPATIBILITY
        with open (savefile, mode='w', encoding='utf-8') as f:
            json.dump(outdict, f, indent=2)

        print('Saving synthetic diagnostic data to:', savefile)

        # pickle.dump(outdict, output)
        #
        # output.close()


    def calc_H_emiss(self):
        '''
        Calculate the hydrogenic emission for spectral lines defined in the input

        If Use_AMJUEL flag set to true, calculate the emission with contributions from molecules and H-
        Otherwise used ADAS rates for contributions from el-impact excitation and recombination.

        TODO: addability to define path to AMJUEL.tex. Currently assume that the file is located in the home dir.

        '''
        print('Calculating H emission...')
        if self.use_AMJUEL:
            print('Using AMJUEL data')
            debug = True
            h3 = False
            if self.AMJUEL_date >= 2017:
                h3 = True
            for line_key in self.spec_line_dict['1']['1']:
                E_excit, E_recom, E_mol, E_h2_pos, E_h3_pos, E_h_neg, E_tot = amread.calc_photon_rate(self.spec_line_dict['1']['1'][line_key], self.te, self.ne, self.n0, mol_n_density=self.n2,molp_n_density=self.n2p,p_density=self.ni, h3=h3, recalc_h2_pos=self.recalc_h2_pos, debug=debug)
                for k in range(len(self.cells)):
                    self.cells[k].H_emiss[line_key] = {'excit': E_excit[k], 'recom': E_recom[k], "h2": E_mol[k], "h2+": E_h2_pos[k], "h3+": E_h3_pos[k], "h-": E_h_neg[k], "tot": E_tot[k], 
                    "units": 'ph.s^-1.m^-3.sr^-1'}
                                      
        else: 
            print("Using Adas")
            for cell in self.cells:
                for line_key in self.spec_line_dict['1']['1']:
                    E_excit, E_recom= adas_adf15_read.get_H_line_emiss(line_key, self.ADAS_dict['adf15']['1']['1'], cell.te, cell.ne*1.0E-06, cell.ni*1.0E-06, cell.n0*1.0E-06)
                    cell.H_emiss[line_key] = {'excit':E_excit, 'recom':E_recom, 'units':'ph.s^-1.m^-3.sr^-1'}

        if self.spec_line_dict_lytrap:
            print('Calculating H emission for Ly trapping...')
            for cell in self.cells:
                for line_key in self.spec_line_dict_lytrap['1']['1']:
                    E_excit, E_recom= adas_adf15_read.get_H_line_emiss(line_key, self.ADAS_dict_lytrap['adf15']['1']['1'], cell.te, cell.ne*1.0E-06, cell.ni*1.0E-06, cell.n0*1.0E-06)
                    cell.H_emiss[line_key] = {'excit':E_excit, 'recom':E_recom, 'units':'ph.s^-1.m^-3.sr^-1'}


    def calc_ff_fb_filtered_emiss(self, filter_wv_nm, filter_tran):
        # TODO: ADD ZEFF CAPABILITY

        wave_nm = np.linspace(filter_wv_nm[0], filter_wv_nm[-1], 10)

        print('Calculating FF+FB filtered emission...')
        for cell in self.cells:
            ff_only, ff_fb_tot = continuo_read.adas_continuo_py(wave_nm, cell.te, 1, 1)
            f = interp1d(wave_nm, ff_fb_tot)
            ff_fb_tot_interp = f(filter_wv_nm)
            # convert to spectral emissivity: ph s-1 m-3 sr-1 nm-1
            ff_fb_tot_interp = ff_fb_tot_interp * cell.ne * cell.ne
            # multiply by filter transmission
            ff_fb_tot_interp *= filter_tran
            # Integrate over wavelength
            ff_fb_tot_emiss = np.trapz(ff_fb_tot_interp, filter_wv_nm)
            cell.ff_fb_filtered_emiss = {'ff_fb':ff_fb_tot_emiss, 'units':'ph.s^-1.m^-3.sr^-1',
                                'filter_wv_nm':filter_wv_nm, 'filter_tran':filter_tran}

    def calc_ff_fb_emiss(self):
        # TODO: ADD ZEFF !

        wave_nm = np.logspace((0.001), np.log10(100000), 500)

        print('Calculating FF+FB emission...')
        sum_ff_radpwr = 0
        for cell in self.cells:
            ff_only, ff_fb = continuo_read.adas_continuo_py(wave_nm, cell.te, 1, 1, output_in_ph_s=False)

            # convert to spectral emissivity (from W m^3 sr^-1 nm^-1 to W m^-3 nm^-1)
            ff_only = ff_only * cell.ne * cell.ne * 4. * np.pi
            ff_fb = ff_fb * cell.ne * cell.ne * 4. * np.pi

            # Integrate over wavelength
            cell.ff_radpwr_perm3 = np.trapz(ff_only, wave_nm) # W m^-3
            cell.ff_fb_radpwr_perm3 = np.trapz(ff_fb, wave_nm) # W m^-3

            cell_vol = cell.poly.area * 2.0 * np.pi * cell.R  # m^3
            cell.ff_radpwr = cell.ff_radpwr_perm3 * cell_vol
            cell.ff_fb_radpwr = cell.ff_fb_radpwr_perm3 * cell_vol

            sum_ff_radpwr += cell.ff_radpwr
        print('Total ff radiated power:', sum_ff_radpwr, ' [W]')

    def calc_H_rad_power(self):
        # Te_rnge = [0.2, 5000]
        # ne_rnge = [1.0e11, 1.0e15]
        # self.H_adf11 = adas_adf11_utils.get_adas_H_adf11_interp(Te_rnge, ne_rnge, npts=self.ADAS_npts, npts_interp=1000, pwr=True)
        print('Calculating H radiated power...')
        sum_pwr = 0
        for cell in self.cells:
            iTe, vTe = find_nearest(self.ADAS_dict['adf11']['1'].Te_arr, cell.te)
            ine, vne = find_nearest(self.ADAS_dict['adf11']['1'].ne_arr, cell.ne*1.0e-06)
            # plt/prb absolute rad pow contr in units W.cm^3
            plt_contr = self.ADAS_dict['adf11']['1'].plt[iTe,ine]*(1.0e-06*cell.n0)*(1.0e-06*cell.ne) #W.cm^-3
            prb_contr = self.ADAS_dict['adf11']['1'].prb[iTe,ine]*(1.0e-06*cell.ni)*(1.0e-06*cell.ne) #W.cm^-3
            cell_vol = cell.poly.area * 2.0 * np.pi * cell.R # m^3
            cell.H_radpwr = (plt_contr+prb_contr) * 1.e06 * cell_vol # Watts
            cell.H_radpwr_perm3 = (plt_contr+prb_contr) * 1.e06 # Watts m^-3

            sum_pwr += np.sum(np.asarray(cell.H_radpwr)) # sanity check. compare to eproc
        self.Prad_H = sum_pwr
        print('Total H radiated power:', sum_pwr, ' [W]')

        if self.spec_line_dict_lytrap:
            print('Calculating H radiated power for Ly trapping...')
            sum_pwr = 0
            for cell in self.cells:
                iTe, vTe = find_nearest(self.ADAS_dict_lytrap['adf11']['1'].Te_arr, cell.te)
                ine, vne = find_nearest(self.ADAS_dict_lytrap['adf11']['1'].ne_arr, cell.ne * 1.0e-06)
                # plt/prb absolute rad pow contr in units W.cm^3
                plt_contr = self.ADAS_dict_lytrap['adf11']['1'].plt[iTe, ine] * (1.0e-06 * cell.n0) * (
                            1.0e-06 * cell.ne)  # W.cm^-3
                prb_contr = self.ADAS_dict_lytrap['adf11']['1'].prb[iTe, ine] * (1.0e-06 * cell.ni) * (
                            1.0e-06 * cell.ne)  # W.cm^-3
                cell_vol = cell.poly.area * 2.0 * np.pi * cell.R  # m^3
                cell.H_radpwr_Lytrap = (plt_contr + prb_contr) * 1.e06 * cell_vol  # Watts
                cell.H_radpwr_Lytrap_perm3 = (plt_contr + prb_contr) * 1.e06  # Watts m^-3
                sum_pwr += np.sum(np.asarray(cell.H_radpwr_Lytrap))  # sanity check. compare to eproc
            self.Prad_H_Lytrap = sum_pwr
            print('Total H radiated power w/ Ly trapping:', sum_pwr, ' [W]')


    def los_intersect(self, los):
        for cell in self.cells:
            # check if cell lies within los.poly
            if los.los_poly.contains(cell.poly):
                los.cells.append(cell)
            # check if cell interstects with los.poly
            elif los.los_poly.intersects(cell.poly):
                clipped_poly = los.los_poly.intersection(cell.poly)
                if clipped_poly.geom_type == 'Polygon':
                    centroid_p = clipped_poly.centroid
                    clipped_cell = Cell(centroid_p.x, centroid_p.y, poly=clipped_poly, te=cell.te, ne=cell.ne, ni=cell.ni, n0=cell.n0, Srec=cell.Srec, Sion=cell.Sion)
                    clipped_cell.H_emiss = cell.H_emiss
                    area_ratio = clipped_cell.poly.area /  cell.poly.area
                    clipped_cell.H_radpwr = cell.H_radpwr * area_ratio
                    clipped_cell.H_radpwr_perm3 = cell.H_radpwr_perm3
                    if self.spec_line_dict_lytrap:
                        clipped_cell.H_radpwr_Lytrap = cell.H_radpwr_Lytrap * area_ratio
                        clipped_cell.H_radpwr_Lytrap_perm3 = cell.H_radpwr_Lytrap_perm3
                    clipped_cell.ff_radpwr = np.asarray(cell.ff_radpwr) * area_ratio
                    clipped_cell.ff_radpwr_perm3 = cell.ff_radpwr_perm3
                    clipped_cell.ff_fb_radpwr = np.asarray(cell.ff_fb_radpwr) * area_ratio
                    clipped_cell.ff_fb_radpwr_perm3 = cell.ff_fb_radpwr_perm3
                    los.cells.append(clipped_cell)

        # Intersection of los centerline with separatrix: returns points where los crosses sep
        # generate los centerline shapely object
#        los.shply_cenline = LineString([(los.p1),(los.p2)])
#        if los.los_poly.intersects(self.shply_sep_poly_below_xpt):
#            los.shply_intersects_w_sep = None#los.shply_cenline.intersection(self.shply_sep_poly_below_xpt)


    def calc_qpol_div(self):

        """ TODO: CHECK THIS PROCEDURE WITH DEREK/JAMES"""

        # LFS
        endidx = self.data.qpartot_LFS['npts']
        xidx, = np.where(np.array(self.data.qpartot_LFS['xdata'][0:endidx]) > 0.0)
        # CONVERT QPAR TO QPOL (QPOL = QPAR * Btheta/Btot)
        qpol_LFS = np.array(self.data.qpartot_LFS['ydata'])[xidx] * np.array(self.data.bpol_btot_LFS['ydata'])[xidx]
        # calculate DR from neighbours
        dR_LFS = np.zeros((len(xidx)))
        for idx, val in enumerate(xidx):
            left_neighb = np.sqrt(((self.data.qpartot_LFS_rmesh['ydata'][val]-
                                    self.data.qpartot_LFS_rmesh['ydata'][val-1])**2)+
                                  ((self.data.qpartot_LFS_zmesh['ydata'][val]-
                                    self.data.qpartot_LFS_zmesh['ydata'][val-1])**2))
            if val != xidx[-1]:
                right_neighb = np.sqrt(((self.data.qpartot_LFS_rmesh['ydata'][val+1]-
                                         self.data.qpartot_LFS_rmesh['ydata'][val])**2)+
                                       ((self.data.qpartot_LFS_zmesh['ydata'][val+1]-
                                         self.data.qpartot_LFS_zmesh['ydata'][val])**2))
            else:
                right_neighb = left_neighb
            dR_LFS[idx] = (left_neighb+right_neighb)/2.0
        area = 2. * np.pi * np.array(self.data.qpartot_LFS_rmesh['ydata'])[xidx] * dR_LFS
        self.qpol_div_LFS = np.sum(qpol_LFS*area)

        # HFS
        endidx = self.data.qpartot_HFS['npts']
        xidx, = np.where(np.array(self.data.qpartot_HFS['xdata'])[0:endidx] > 0.0)
        # CONVERT QPAR TO QPOL (QPOL = QPAR * Btheta/Btot)
        qpol_HFS = np.array(self.data.qpartot_HFS['ydata'])[xidx] * np.array(self.data.bpol_btot_HFS['ydata'])[xidx]
        # calculate dR from neighbours
        dR_HFS = np.zeros((len(xidx)))
        for idx, val in enumerate(xidx):
            left_neighb = np.sqrt(((self.data.qpartot_HFS_rmesh['ydata'][val]-
                                    self.data.qpartot_HFS_rmesh['ydata'][val-1])**2)+
                                  ((self.data.qpartot_HFS_zmesh['ydata'][val]-
                                    self.data.qpartot_HFS_zmesh['ydata'][val-1])**2))
            if val != xidx[-1]:
                right_neighb = np.sqrt(((self.data.qpartot_HFS_rmesh['ydata'][val+1]-
                                         self.data.qpartot_HFS_rmesh['ydata'][val])**2)+
                                       ((self.data.qpartot_HFS_zmesh['ydata'][val+1]-
                                         self.data.qpartot_HFS_zmesh['ydata'][val])**2))
            else:
                right_neighb = left_neighb
            dR_HFS[idx] = (left_neighb+right_neighb)/2.0
        area = 2. * np.pi * np.array(self.data.qpartot_HFS_rmesh['ydata'])[xidx] * dR_HFS
        self.qpol_div_HFS = np.sum(qpol_HFS*area)

        print('Pdiv_LFS (MW): ', self.qpol_div_LFS*1.e-06, 'Pdiv_HFS (MW): ', 
              self.qpol_div_HFS*1.e-06, 'POWSOL (MW): ', 
              self.data.powsol['data'][ self.data.powsol['npts']-1]*1.e-06)


    def calc_region_aggregates(self):

        # Calculates for each region:
        #   radiated power
        #   ionisation and recombination
        for regname, region in self.regions.items():
            for cell in self.cells:
                if region.cell_in_region(cell, self.data.shply_sep_poly):
                    region.cells.append(cell)
                    region.Prad_units = 'W'
                    region.Prad_H += cell.H_radpwr
                    if self.spec_line_dict_lytrap:
                        region.Prad_H_Lytrap += cell.H_radpwr_Lytrap
                    # ionization/recombination * cell volume
                    region.Sion += cell.Sion * 2.*np.pi*cell.R * cell.poly.area
                    region.Srec += cell.Srec * 2.*np.pi*cell.R * cell.poly.area

    def plot_region(self, name='LFS_DIV'):

        fig, ax = plt.subplots(ncols=1)
        region_patches = []
        for regname, region in self.regions.items():
            if regname == name:
                for cell in region.cells:
                    region_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=False))

        # region_patches.append(patches.Polygon(self.shply_sep_poly.exterior.coords, closed=False))
        coll = PatchCollection(region_patches)
        ax.add_collection(coll)
        ax.set_xlim(1.8, 4.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_title(name)
        ax.add_patch(self.sep_poly)
        ax.add_patch(self.wall_poly)
        plt.axes().set_aspect('equal')


class LOS:

    def __init__(self, diag, los_poly = None, chord_num=None, p1 = None, w1 = None, p2orig = None, p2 = None,
                 w2orig = None, w2 = None, l12 = None, theta=None, los_angle=None, spec_line_dict=None,
                 spec_line_dict_lytrap=None, use_AMJUEL = False):

        self.diag = diag

        self.los_poly = los_poly
        self.chord_num = chord_num
        self.p1 = p1
        self.w1 = w1
        self.p2unmod = p2orig
        self.p2 = p2
        self.w2unmod = w2orig
        self.w2 = w2
        self.l12 = l12
        self.theta = theta
        self.los_angle = los_angle
        self.cells = []
        self.spec_line_dict = spec_line_dict
        self.spec_line_dict_lytrap = spec_line_dict_lytrap
        self.use_AMJUEL = use_AMJUEL

        # center line shapely object
        self.shply_cenline = None
        # intersection with separatrix below x-point (for identifying PFR-SOL regions)
        self.shply_intersects_w_sep = None

        # LINE INTEGRATED QUANTITIES - DICT
        # {'param':val, 'units':''}
        self.los_int = {}

        # 1D QUANTITIES ALONG LOS - DICT
        # {'param':val, 'units':''}
        self.los_1d = {}

        # SYNTHETIC SPECTRA - DICT
        # e.g. features: 'stark', 'ff_fb_continuum'
        # {'feature':{'wave':1d_arr, 'intensity':1d_arr, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}}
        self.los_int_spectra = {}
        # {'feature':{'wave':2d_arr[dl_idx,arr], 'intensity':2d_arr[dl_idx,arr], 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}}
        self.los_1d_spectra = {}
        # Stark broadening coefficients (Lomanowski et al 2015, NF)
        self.lorentz_5_2_coeffs = {}
        if self.spec_line_dict['1']['1']:
            for key in self.spec_line_dict['1']['1']:
                # H 6-2
                if self.spec_line_dict['1']['1'][key][1] == '2' and self.spec_line_dict['1']['1'][key][0] == '6':
                    self.lorentz_5_2_coeffs[key] = {'C':3.954E-16, 'a':0.7149, 'b':0.028}


    def orthogonal_polys(self):
        for cell in self.cells:
            # FOR EACH CELL IN LOS, COMPUTE EQUIVALENT ORTHOGONAL POLY

            cell_area = cell.poly.area
            cell.dist_to_los_v1 = np.sqrt(((self.p1[0] - cell.R) ** 2) + ((self.p1[1] - cell.Z) ** 2))


            # Handle case of uniform (or close to uniform) LOS beam width (e.g., KG1V)
            if np.abs(self.w2-self.w1) <= 1.0e-03:
                cell.los_ortho_width = self.w2
            # Similar triangles method
            else:
                dw = (cell.dist_to_los_v1 / self.l12 ) * ((self.w2 - self.w1) / 2.0)
                cell.los_ortho_width = self.w1 + 2.0 * dw

            cell.los_ortho_delL = cell_area / cell.los_ortho_width

    def calc_int_and_1d_los_quantities_1(self):

        # SUM H EMISSION METHOD 1: TRAVERSE CELLS AND SUM EMISSIVITIES*ORTHO_DELL
        self.los_int['H_emiss'] = {}
        for key in self.spec_line_dict['1']['1']:
            sum_excit = 0
            sum_recom = 0
            for cell in self.cells:
                sum_excit += (cell.H_emiss[key]['excit'] * cell.los_ortho_delL)
                sum_recom += (cell.H_emiss[key]['recom'] * cell.los_ortho_delL)
            self.los_int['H_emiss'].update({key:{'excit':sum_excit, 'recom':sum_recom, 'units':'ph.s^-1.m^-2.sr^-1'}})

        # Same for Ly-opacity
        if self.spec_line_dict_lytrap:
            for key in self.spec_line_dict_lytrap['1']['1']:
                sum_excit = 0
                sum_recom = 0
                for cell in self.cells:
                    sum_excit += (cell.H_emiss[key]['excit'] * cell.los_ortho_delL)
                    sum_recom += (cell.H_emiss[key]['recom'] * cell.los_ortho_delL)
                self.los_int['H_emiss'].update({key:{'excit':sum_excit, 'recom':sum_recom, 'units':'ph.s^-1.m^-2.sr^-1'}})

        # SUM TOTAL RECOMBINATION/IONISATION
        self.los_int['Srec'] = {}
        self.los_int['Sion'] = {}
        sum_Srec = 0
        sum_Sion = 0
        for cell in self.cells:
            sum_Srec += (cell.Srec * cell.poly.area * 2.*np.pi*cell.R)
            sum_Sion += (cell.Sion * cell.poly.area * 2.*np.pi*cell.R)
        self.los_int['Srec'].update({'val':sum_Srec, 'units':'s^-1'})
        self.los_int['Sion'].update({'val':sum_Sion, 'units':'s^-1'})

        # SUM RADIATED POWER (total and per m^3)
        self.los_int['Prad'] = {}
        self.los_int['Prad_perm2'] = {}

        if self.spec_line_dict_lytrap:
            self.los_int['Prad_Lytrap'] = {}
            self.los_int['Prad_Lytrap_perm2'] = {}
            sum_Prad_H_Lytrap = 0
            sum_Prad_H_Lytrap_perm2 = 0

        sum_Prad_H = 0
        sum_Prad_H_perm2 = 0
        sum_Prad_ff = 0
        sum_Prad_ff_perm2 = 0
        for cell in self.cells:
            sum_Prad_H += (cell.H_radpwr)
            sum_Prad_H_perm2 += (cell.H_radpwr_perm3*cell.los_ortho_delL)

            if self.spec_line_dict_lytrap:
                sum_Prad_H_Lytrap += (cell.H_radpwr_Lytrap)
                sum_Prad_H_Lytrap_perm2 += (cell.H_radpwr_Lytrap_perm3 * cell.los_ortho_delL)

            sum_Prad_ff += (cell.ff_radpwr)
            sum_Prad_ff_perm2 += (cell.ff_radpwr_perm3*cell.los_ortho_delL)
        self.los_int['Prad'].update({'H':sum_Prad_H,  'units':'W'})
        self.los_int['Prad_perm2'].update({'H':sum_Prad_H_perm2,  'units':'W m^-2'})
        self.los_int['Prad'].update({'ff':sum_Prad_ff,  'units':'W'})
        self.los_int['Prad_perm2'].update({'ff':sum_Prad_ff_perm2,  'units':'W m^-2'})

        if self.spec_line_dict_lytrap:
            self.los_int['Prad_Lytrap'].update({'H': sum_Prad_H_Lytrap, 'units': 'W'})
            self.los_int['Prad_Lytrap_perm2'].update({'H': sum_Prad_H_Lytrap_perm2, 'units': 'W m^-2'})

    def calc_int_and_1d_los_quantities_2(self):
        ####################################################
        # COMPUTE AVERAGED QUANTITIES ALONG LOS
        ####################################################
        self.los_1d['dl'] = 0.01 # m
        self.los_1d['l'] = np.arange(0, self.l12, self.los_1d['dl'])
        for item in ['ne', 'n0', 'te', 'ortho_delL']:
            self.los_1d[item] = np.zeros((len(self.los_1d['l'])))

        # H EMISS LOS 1D METHOD 2: SIMILAR TO METHOD 1 ABOVE, EXCEPT 1D PROFILE INFO IS ALSO STORED ALONG LOS
        recom = {}
        excit = {}
        recom_pervol = {} # emissivity per m^-3
        excit_pervol = {} # emissivity per m^-3

        for key in self.spec_line_dict['1']['1']:
            recom[key] = np.zeros((len(self.los_1d['l'])))
            excit[key] = np.zeros((len(self.los_1d['l'])))
            recom_pervol[key] = np.zeros((len(self.los_1d['l'])))
            excit_pervol[key] = np.zeros((len(self.los_1d['l'])))

        

        for dl_idx, dl_val in enumerate(self.los_1d['l']):
            # temp storage
            ne_tmp = []
            n0_tmp = []
            te_tmp = []
            delL_tmp = []
            H_emiss_excit_tmp = {}
            H_emiss_recom_tmp = {}
            H_emiss_pervol_excit_tmp = {}
            H_emiss_pervol_recom_tmp = {}
            for key in self.spec_line_dict['1']['1']:
                H_emiss_excit_tmp[key] = []
                H_emiss_recom_tmp[key] = []
                H_emiss_pervol_excit_tmp[key] = []
                H_emiss_pervol_recom_tmp[key] = []
                            
            for cell in self.cells:
                if cell.dist_to_los_v1 >= dl_val and cell.dist_to_los_v1 < dl_val + self.los_1d['dl']:
                    ne_tmp.append(cell.ne)
                    n0_tmp.append(cell.n0)
                    te_tmp.append(cell.te)
                    delL_tmp.append(cell.los_ortho_delL)
                    for key in self.spec_line_dict['1']['1']:
                        H_emiss_excit_tmp[key].append(cell.H_emiss[key]['excit']*cell.los_ortho_delL)
                        H_emiss_recom_tmp[key].append(cell.H_emiss[key]['recom']*cell.los_ortho_delL)
                        H_emiss_pervol_excit_tmp[key].append(cell.H_emiss[key]['excit'])
                        H_emiss_pervol_recom_tmp[key].append(cell.H_emiss[key]['recom'])
            if ne_tmp:
                self.los_1d['ortho_delL'][dl_idx] = np.sum(np.asarray(delL_tmp))
                delL_norm = np.asarray(delL_tmp) / self.los_1d['ortho_delL'][dl_idx]
                self.los_1d['ne'][dl_idx] = np.average(np.asarray(ne_tmp),weights = delL_norm)
                self.los_1d['n0'][dl_idx] = np.average(np.asarray(n0_tmp),weights = delL_norm)
                self.los_1d['te'][dl_idx] = np.average(np.asarray(te_tmp),weights = delL_norm)
                for key in self.spec_line_dict['1']['1']:
                    excit[key][dl_idx] = np.sum(np.asarray(H_emiss_excit_tmp[key]))
                    recom[key][dl_idx] = np.sum(np.asarray(H_emiss_recom_tmp[key]))
                    excit_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_excit_tmp[key]), weights = delL_norm)
                    recom_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_recom_tmp[key]), weights = delL_norm )
                
        # REMOVE ZEROS FROM COMPUTED LOS ARRAYS
        nonzero_idx = np.nonzero(self.los_1d['ne'])
        for item in ['l', 'ne', 'n0', 'te', 'ortho_delL']:
            self.los_1d[item] = self.los_1d[item][nonzero_idx]
            # convert numpy arrays to lists for JSON serialization
            self.los_1d[item] = list(self.los_1d[item])

        self.los_1d['H_emiss'] = {}
        self.los_1d['H_emiss_per_vol'] = {}
        for key in self.spec_line_dict['1']['1']:
            excit[key] = excit[key][nonzero_idx]
            recom[key] = recom[key][nonzero_idx]
            excit_pervol[key] = excit_pervol[key][nonzero_idx]
            recom_pervol[key] = recom_pervol[key][nonzero_idx]
            self.los_1d['H_emiss'].update({key:{'excit':excit[key], 'recom':recom[key], 'units':'ph s^-1 m^-2 sr^-1'}})
            self.los_1d['H_emiss_per_vol'].update({key:{'excit':excit_pervol[key], 'recom':recom_pervol[key], 'units':'ph s^-1 m^-3 sr^-1'}})
            # convert numpy arrays to lists for JSON serialization
            self.los_1d['H_emiss'][key]['excit'] = list(self.los_1d['H_emiss'][key]['excit'])
            self.los_1d['H_emiss'][key]['recom'] = list(self.los_1d['H_emiss'][key]['recom'])
            self.los_1d['H_emiss_per_vol'][key]['excit'] = list(self.los_1d['H_emiss_per_vol'][key]['excit'])
            self.los_1d['H_emiss_per_vol'][key]['recom'] = list(self.los_1d['H_emiss_per_vol'][key]['recom'])

    def calc_int_and_1d_los_quantities_AMJUEL_1(self):
        # SUM H EMISSION METHOD 1: TRAVERSE CELLS AND SUM EMISSIVITIES*ORTHO_DELL
        self.los_int['H_emiss'] = {}
        for key in self.spec_line_dict['1']['1']:
            sum_exc = 0
            sum_rec = 0
            sum_h2 = 0
            sum_h2_pos = 0
            sum_h_neg = 0
            sum_tot = 0
            for cell in self.cells:
                sum_tot += (cell.H_emiss[key]['tot'] * cell.los_ortho_delL)
                sum_exc += (cell.H_emiss[key]['excit'] * cell.los_ortho_delL)
                sum_rec += (cell.H_emiss[key]['recom'] * cell.los_ortho_delL)
                sum_h2  += (cell.H_emiss[key]['h2'] * cell.los_ortho_delL)
                sum_h2_pos += (cell.H_emiss[key]['h2+'] * cell.los_ortho_delL)
                sum_h_neg += (cell.H_emiss[key]['h-'] * cell.los_ortho_delL)
            self.los_int['H_emiss'].update({key:{'tot':sum_tot, 'excit': sum_exc, 'recom': sum_rec,
                                            'h2': sum_h2, 'h2+': sum_h2_pos, 'h-': sum_h_neg, 'units':'ph.s^-1.m^-2.sr^-1'}})

    def calc_int_and_1d_los_quantities_AMJUEL_2(self):
        ####################################################
        # COMPUTE AVERAGED QUANTITIES ALONG LOS
        ####################################################
        self.los_1d['dl'] = 0.01 # m
        self.los_1d['l'] = np.arange(0, self.l12, self.los_1d['dl'])
        for item in ['ne', 'n0', 'te', 'ortho_delL']:
            self.los_1d[item] = np.zeros((len(self.los_1d['l'])))
        # H EMISS LOS 1D METHOD 2: SIMILAR TO METHOD 1 ABOVE, EXCEPT 1D PROFILE INFO IS ALSO STORED ALONG LOS
        # Added for processing emission calculated from AMJUEL, initially only for the total emissivity
        em = {}
        em_pervol = {}
        recom = {}
        excit = {}
        h2 = {}
        h2_pos = {}
        h_neg = {}
        h2_pervol = {}
        h2_pos_pervol = {}
        h_neg_pervol = {}
        recom_pervol = {} # emissivity per m^-3
        excit_pervol = {} # emissivity per m^-3

        for key in self.spec_line_dict['1']['1']:
            em[key] = np.zeros((len(self.los_1d['l'])))
            em_pervol[key] = np.zeros((len(self.los_1d['l'])))
            recom[key] = np.zeros((len(self.los_1d['l'])))
            excit[key] = np.zeros((len(self.los_1d['l'])))
            h2[key] = np.zeros((len(self.los_1d['l'])))
            h2_pos[key] = np.zeros((len(self.los_1d['l'])))
            h_neg[key] = np.zeros((len(self.los_1d['l'])))
            recom_pervol[key] = np.zeros((len(self.los_1d['l'])))
            excit_pervol[key] = np.zeros((len(self.los_1d['l'])))
            h2_pervol[key] = np.zeros((len(self.los_1d['l'])))
            h2_pos_pervol[key] = np.zeros((len(self.los_1d['l'])))
            h_neg_pervol[key] = np.zeros((len(self.los_1d['l'])))

        for dl_idx, dl_val in enumerate(self.los_1d['l']):
            # temp storage
            ne_tmp = []
            n0_tmp = []
            te_tmp = []
            delL_tmp = []
            H_emiss_em_tmp = {}
            H_emiss_em_pervol_tmp = {}
            H_emiss_excit_tmp = {}
            H_emiss_recom_tmp = {}
            H_emiss_h2_tmp = {}
            H_emiss_h2_pos_tmp = {}
            H_emiss_h_neg_tmp = {}
            H_emiss_pervol_excit_tmp = {}
            H_emiss_pervol_recom_tmp = {}
            H_emiss_pervol_h2_tmp = {}
            H_emiss_pervol_h2_pos_tmp = {}
            H_emiss_pervol_h_neg_tmp = {}
                
            for key in self.spec_line_dict['1']['1']:
                H_emiss_em_tmp[key] = []
                H_emiss_em_pervol_tmp[key] = []
                H_emiss_excit_tmp[key] = []
                H_emiss_recom_tmp[key] = []
                H_emiss_h2_tmp[key] = []
                H_emiss_h2_pos_tmp[key] = []
                H_emiss_h_neg_tmp[key] = []
                H_emiss_pervol_excit_tmp[key] = []
                H_emiss_pervol_recom_tmp[key] = []
                H_emiss_pervol_h2_tmp[key] = []
                H_emiss_pervol_h2_pos_tmp[key] = []
                H_emiss_pervol_h_neg_tmp[key] = []
                            
            for cell in self.cells:
                if cell.dist_to_los_v1 >= dl_val and cell.dist_to_los_v1 < dl_val + self.los_1d['dl']:
                    ne_tmp.append(cell.ne)
                    n0_tmp.append(cell.n0)
                    te_tmp.append(cell.te)
                    delL_tmp.append(cell.los_ortho_delL)
                    for key in self.spec_line_dict['1']['1']:
                        H_emiss_em_tmp[key].append(cell.H_emiss[key]['tot']*cell.los_ortho_delL)
                        H_emiss_em_pervol_tmp[key].append(cell.H_emiss[key]['tot'])
                        H_emiss_excit_tmp[key].append(cell.H_emiss[key]['excit']*cell.los_ortho_delL)
                        H_emiss_recom_tmp[key].append(cell.H_emiss[key]['recom']*cell.los_ortho_delL)
                        H_emiss_h2_tmp[key].append(cell.H_emiss[key]['h2']*cell.los_ortho_delL)
                        H_emiss_h2_pos_tmp[key].append(cell.H_emiss[key]['h2+']*cell.los_ortho_delL)
                        H_emiss_h_neg_tmp[key].append(cell.H_emiss[key]['h-']*cell.los_ortho_delL)
                        H_emiss_pervol_excit_tmp[key].append(cell.H_emiss[key]['excit'])
                        H_emiss_pervol_recom_tmp[key].append(cell.H_emiss[key]['recom'])
                        H_emiss_pervol_h2_tmp[key].append(cell.H_emiss[key]['h2'])
                        H_emiss_pervol_h2_pos_tmp[key].append(cell.H_emiss[key]['h2+'])
                        H_emiss_pervol_h_neg_tmp[key].append(cell.H_emiss[key]['h-'])
            if ne_tmp:
                self.los_1d['ortho_delL'][dl_idx] = np.sum(np.asarray(delL_tmp))
                delL_norm = np.asarray(delL_tmp) / self.los_1d['ortho_delL'][dl_idx]
                self.los_1d['ne'][dl_idx] = np.average(np.asarray(ne_tmp),weights = delL_norm)
                self.los_1d['n0'][dl_idx] = np.average(np.asarray(n0_tmp),weights = delL_norm)
                self.los_1d['te'][dl_idx] = np.average(np.asarray(te_tmp),weights = delL_norm)
                for key in self.spec_line_dict['1']['1']:
                    em[key][dl_idx] = np.sum(np.asarray(H_emiss_em_tmp[key]))
                    em_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_em_pervol_tmp[key]), weights = delL_norm )
                    recom[key][dl_idx] = np.sum(np.asarray(H_emiss_recom_tmp[key]))
                    excit[key][dl_idx] = np.sum(np.asarray(H_emiss_excit_tmp[key]))
                    h2[key][dl_idx] = np.sum(np.asarray(H_emiss_h2_tmp[key]))
                    h2_pos[key][dl_idx] = np.sum(np.asarray(H_emiss_h2_pos_tmp[key]))
                    h_neg[key][dl_idx] = np.sum(np.asarray(H_emiss_h_neg_tmp[key]))
                    recom_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_recom_tmp[key]), weights = delL_norm )
                    excit_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_excit_tmp[key]), weights = delL_norm )
                    h2_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_h2_tmp[key]), weights = delL_norm )
                    h2_pos_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_h2_pos_tmp[key]), weights = delL_norm )
                    h_neg_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_h_neg_tmp[key]), weights = delL_norm )
                
        # REMOVE ZEROS FROM COMPUTED LOS ARRAYS
        nonzero_idx = np.nonzero(self.los_1d['ne'])
        for item in ['l', 'ne', 'n0', 'te', 'ortho_delL']:
            self.los_1d[item] = self.los_1d[item][nonzero_idx]
            # convert numpy arrays to lists for JSON serialization
            self.los_1d[item] = list(self.los_1d[item])

        self.los_1d['H_emiss'] = {}
        self.los_1d['H_emiss_per_vol'] = {}
        for key in self.spec_line_dict['1']['1']:
            em[key] = em[key][nonzero_idx]
            em_pervol[key] = em_pervol[key][nonzero_idx]
            excit[key] = excit[key][nonzero_idx]
            excit_pervol[key] = excit_pervol[key][nonzero_idx]
            recom[key] = recom[key][nonzero_idx]
            recom_pervol[key] = recom_pervol[key][nonzero_idx]
            h2[key] = h2[key][nonzero_idx]
            h2_pervol[key] = h2_pervol[key][nonzero_idx]
            h2_pos[key] = h2_pos[key][nonzero_idx]
            h2_pos_pervol[key] = h2_pos_pervol[key][nonzero_idx]
            h_neg[key] = h_neg[key][nonzero_idx]
            h_neg_pervol[key] = h_neg_pervol[key][nonzero_idx]
            self.los_1d['H_emiss'].update({key:{'tot':em[key], 'excit': excit[key], 'recom': recom[key],
                                          'h2': h2[key], 'h2+': h2_pos[key], 'h-': h_neg[key],'units':'ph s^-1 m^-2 sr^-1'}})
            self.los_1d['H_emiss_per_vol'].update({key:{'tot':em_pervol[key], 'excit': excit_pervol[key], 'recom': recom_pervol[key],
                                          'h2': h2_pervol[key], 'h2+': h2_pos_pervol[key], 'h-': h_neg_pervol[key], 'units':'ph s^-1 m^-3 sr^-1'}})
            # convert numpy arrays to lists for JSON serialization
            self.los_1d['H_emiss'][key]['tot'] = list(self.los_1d['H_emiss'][key]['tot'])
            self.los_1d['H_emiss_per_vol'][key]['tot'] = list(self.los_1d['H_emiss_per_vol'][key]['tot'])
            self.los_1d['H_emiss'][key]['excit'] = list(self.los_1d['H_emiss'][key]['excit'])
            self.los_1d['H_emiss_per_vol'][key]['excit'] = list(self.los_1d['H_emiss_per_vol'][key]['excit'])
            self.los_1d['H_emiss'][key]['recom'] = list(self.los_1d['H_emiss'][key]['recom'])
            self.los_1d['H_emiss_per_vol'][key]['recom'] = list(self.los_1d['H_emiss_per_vol'][key]['recom'])
            self.los_1d['H_emiss'][key]['h2'] = list(self.los_1d['H_emiss'][key]['h2'])
            self.los_1d['H_emiss_per_vol'][key]['h2'] = list(self.los_1d['H_emiss_per_vol'][key]['h2'])
            self.los_1d['H_emiss'][key]['h2+'] = list(self.los_1d['H_emiss'][key]['h2+'])
            self.los_1d['H_emiss_per_vol'][key]['h2+'] = list(self.los_1d['H_emiss_per_vol'][key]['h2+'])
            self.los_1d['H_emiss'][key]['h-'] = list(self.los_1d['H_emiss'][key]['h-'])
            self.los_1d['H_emiss_per_vol'][key]['h-'] = list(self.los_1d['H_emiss_per_vol'][key]['h-'])


    def calc_int_and_1d_los_synth_spectra(self):

        ###############################################################
        # FREE-FREE AND FREE-BOUND CONTINUUM
        ###############################################################
        print('Calculating ff fb continuum spectra for chord', self.chord_num)
        wave_nm = np.linspace(300, 500, 50)

        # METHOD 1: SUM CELL-WISE (OK, good agreement with METHOD 2 below, so comment out)
        # sum_ff_fb = np.zeros((len(wave_nm)))
        # for cell in self.cells:
        #     # call adas continuo function (return units: ph s-1 m3 sr-1 nm-1)
        #     ff_only, ff_fb_tot = continuo_read.adas_continuo_py(wave_nm, cell.te, 1, 1)
        #     # convert to spectral radiance: ph s-1 m-2 sr-1 nm-1
        #     sum_ff_fb += ff_fb_tot * cell.ne * cell.ne * cell.los_ortho_delL

        # METHOD 2: SUM BASED ON AVERAGED 1D LOS PARAMS
        # TODO: ADD ZEFF CAPABILITY
        dl_ff_fb_abs = np.zeros((len(self.los_1d['l']), len(wave_nm)))
        for dl_idx, dl_val in enumerate(self.los_1d['l']):
            # call adas continuo function (return units: ph s-1 m3 sr-1 nm-1)
            ff_only, ff_fb_tot = continuo_read.adas_continuo_py(wave_nm, self.los_1d['te'][dl_idx], 1, 1)
            # convert to spectral radiance: ph s-1 m-2 sr-1 nm-1
            dl_ff_fb_abs[dl_idx] = ff_fb_tot * self.los_1d['ne'][dl_idx] * self.los_1d['ne'][dl_idx] * self.los_1d['ortho_delL'][dl_idx]

        # store spectra
        self.los_1d_spectra['ff_fb_continuum'] = {'wave':wave_nm, 'intensity':dl_ff_fb_abs, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
        self.los_int_spectra['ff_fb_continuum'] = {'wave':wave_nm, 'intensity':np.sum(dl_ff_fb_abs, axis=0), 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
        # convert numpy array to list for JSON serialization
        self.los_1d_spectra['ff_fb_continuum']['wave'] = list(self.los_1d_spectra['ff_fb_continuum']['wave'])
        self.los_1d_spectra['ff_fb_continuum']['intensity'] = list(self.los_1d_spectra['ff_fb_continuum']['intensity'])
        self.los_int_spectra['ff_fb_continuum']['wave'] = list(self.los_int_spectra['ff_fb_continuum']['wave'])
        self.los_int_spectra['ff_fb_continuum']['intensity'] = list(self.los_int_spectra['ff_fb_continuum']['intensity'])

        ###############################################################
        # STARK BROADENED H6-2 LINE
        ###############################################################
        print('Calculating Stark H6-2 spectra for chord ', self.chord_num)
        # Generate modified lorentzian profile for each dl position along los (cf Lomanowski et al 2015, NF)
        for key in self.spec_line_dict['1']['1']:
            # H 6-2
            if self.spec_line_dict['1']['1'][key][1] == '2' and self.spec_line_dict['1']['1'][key][0] == '6':
                cwl = float(key) / 10.0 # nm
                # wave_nm = np.linspace(cwl-5, cwl+5, 10000)
                wave_nm = np.linspace(cwl-15, cwl+15, 1000)
                dl_stark = np.zeros((len(self.los_1d['l']), len(wave_nm)))
                for dl_idx, dl_val in enumerate(self.los_1d['l']):
                    stark_fwhm = ( self.lorentz_5_2_coeffs[key]['C']*np.power(self.los_1d['ne'][dl_idx], self.lorentz_5_2_coeffs[key]['a']) /
                                   np.power(self.los_1d['te'][dl_idx], self.lorentz_5_2_coeffs[key]['b']) )
                    dl_stark[dl_idx] = 1. / ( np.power(np.abs(wave_nm-cwl), 5./2.) + np.power(stark_fwhm / 2., 5./2.) )
                    # normalise by emissivity
                    dl_emiss = self.los_1d['H_emiss'][key]['excit'][dl_idx] + self.los_1d['H_emiss'][key]['recom'][dl_idx]
                    if self.use_AMJUEL:
                        dl_emiss += self.los_1d['H_emiss'][key]['h2'][dl_idx]
                        dl_emiss += self.los_1d['H_emiss'][key]['h2+'][dl_idx]
                        dl_emiss += self.los_1d['H_emiss'][key]['h-'][dl_idx]
                    wv_area = np.trapz(dl_stark[dl_idx], x = wave_nm)
                    amp_scal = dl_emiss / wv_area
                    dl_stark[dl_idx] *= amp_scal

                # store spectra
                self.los_1d_spectra['stark'] = {'cwl':cwl, 'wave':wave_nm, 'intensity':dl_stark, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
                self.los_int_spectra['stark'] = {'cwl':cwl, 'wave':wave_nm, 'intensity':np.sum(dl_stark, axis=0), 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
                # convert numpy array to list for JSON serialization
                self.los_1d_spectra['stark']['wave'] = list(self.los_1d_spectra['stark']['wave'])
                self.los_1d_spectra['stark']['intensity'] = list(self.los_1d_spectra['stark']['intensity'])
                self.los_int_spectra['stark']['wave'] = list(self.los_int_spectra['stark']['wave'])
                self.los_int_spectra['stark']['intensity'] = list(self.los_int_spectra['stark']['intensity'])


class SynthDiag:

    def __init__(self, defs, diag, pulse=None, spec_line_dict=None, spec_line_dict_lytrap=None, use_AMJUEL = False):

        self.diag = diag
        self.pulse = pulse
        self.chords = []
        self.spec_line_dict = spec_line_dict
        self.spec_line_dict_lytrap = spec_line_dict_lytrap
        self.use_AMJUEL = use_AMJUEL
        
        self.get_spec_geom(defs)

    def get_spec_geom(self, defs):

        if self.diag in defs.diag_dict.keys():
            p2new = np.zeros((len(defs.diag_dict[self.diag]['p2']), 2))
            for i in range(len(defs.diag_dict[self.diag]['p1'])):
                r1 = defs.diag_dict[self.diag]['p1'][i, 0]
                z1 = defs.diag_dict[self.diag]['p1'][i, 1]
                r2 = defs.diag_dict[self.diag]['p2'][i, 0]
                z2 = defs.diag_dict[self.diag]['p2'][i, 1]
                w1 = defs.diag_dict[self.diag]['w'][i, 0]
                w2 = defs.diag_dict[self.diag]['w'][i, 1]
                if 'angle' in defs.diag_dict[self.diag]:
                    los_angle = defs.diag_dict[self.diag]['angle'][i]
                else:
                    los_angle = None
                theta = np.arctan2((r2 - r1), (z2 - z1))
                # elongate los to ensure defined LOS intersects machine wall (otherwise grid cells may be excluded)
                chord_L = np.sqrt((r1 - r2) ** 2 + (z1 - z2) ** 2)
                p2new[i, 0] = r2 + 1.0 * np.sin(theta)#*np.sign(theta)
                p2new[i, 1] = z2 + 1.0 * np.cos(theta)#*np.sign(theta)
                chord_L_elong = np.sqrt((r1 - p2new[i, 0]) ** 2 + (z1 - p2new[i, 1]) ** 2)
                w2_elong = w2 * chord_L_elong / chord_L
                self.chords.append(LOS(self.diag, los_poly=Polygon([(r1, z1),
                                                         (p2new[i, 0] - 0.5 * w2_elong * np.cos(theta),
                                                          p2new[i, 1] + 0.5 * w2_elong * np.sin(theta)),
                                                         (p2new[i, 0] + 0.5 * w2_elong * np.cos(theta),
                                                          p2new[i, 1] - 0.5 * w2_elong * np.sin(theta)),
                                                         (r1, z1)]),
                                       chord_num=defs.diag_dict[self.diag]['id'][i], p1=[r1, z1], w1=w1, p2orig=[r2, z2],
                                       p2=[p2new[i, 0], p2new[i, 1]], w2orig=w2, w2=w2_elong, l12=chord_L_elong, theta=theta,
                                       los_angle = los_angle, spec_line_dict=self.spec_line_dict,
                                       spec_line_dict_lytrap=self.spec_line_dict_lytrap, 
                                       use_AMJUEL = self.use_AMJUEL))

    def plot_LOS(self, ax, color='w', lw='2.0', Rrng=None):
        for chord in self.chords:
            if Rrng:
                if chord.v2unmod[0] >= Rrng[0] and chord.v2unmod[0] <= Rrng[1]:
                    los_patch = patches.Polygon(chord.los_poly.exterior.coords, closed=False, ec=color, lw=lw, fc='None', zorder=10)
                    ax.add_patch(los_patch)
            else:
                los_patch = patches.Polygon(chord.los_poly.exterior.coords, closed=False, ec=color, lw=lw, fc='None', zorder=10)
                ax.add_patch(los_patch)

    def plot_synth_spec_edge2d_data(self):
        fig, ax1 = plt.subplots(ncols=3, sharex=True, sharey=True)
        ax1[0].set_xlim(1.8, 4.0)
        ax1[0].set_ylim(-2.0, 2.0)
        fig.suptitle(self.diag)
        ax1[0].set_title(r'$\mathrm{T_{e}}$')
        ax1[1].set_title(r'$\mathrm{n_{e}}$')
        ax1[2].set_title(r'$\mathrm{n_{0}}$')

        recon_patches=[]
        recon_grid = []
        los_patches = []
        te=[]
        ne=[]
        n0=[]
        for chord in self.chords:
            los_patches.append(patches.Polygon(chord.los_poly.exterior.coords, closed=False, color='r'))
            for cell in chord.cells:
                recon_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True))
                recon_grid.append(patches.Polygon(cell.poly.exterior.coords, closed=False, color='k', alpha=1.0))
                te.append(cell.te)
                ne.append(cell.ne)
                n0.append(cell.n0)


        coll1 = PatchCollection(recon_patches)
        colors = plt.cm.jet(te/(np.max(te)/10.))
        coll1.set_color(colors)
        ax1[0].add_collection(coll1)

        coll2 = PatchCollection(recon_patches)
        colors = plt.cm.jet(ne/np.max(ne))
        coll2.set_color(colors)
        ax1[1].add_collection(coll2)

        coll3 = PatchCollection(recon_patches)
        colors = plt.cm.jet(n0/np.max(n0))
        coll3.set_color(colors)
        ax1[2].add_collection(coll3)

        split_grid = PatchCollection(recon_grid)
        split_grid.set_facecolor('None')
        split_grid.set_edgecolor('k')
        split_grid.set_linewidth(0.25)
        ax1[0].add_collection(split_grid)

        split_grid2 = PatchCollection(recon_grid)
        split_grid2.set_facecolor('None')
        split_grid2.set_edgecolor('k')
        split_grid2.set_linewidth(0.25)
        ax1[1].add_collection(split_grid2)

        split_grid3 = PatchCollection(recon_grid)
        split_grid3.set_facecolor('None')
        split_grid3.set_edgecolor('k')
        split_grid3.set_linewidth(0.25)
        ax1[2].add_collection(split_grid3)

        # los_coll = PatchCollection(los_patches)
        # los_coll.set_facecolor('None')
        # los_coll.set_edgecolor('r')
        # los_coll.set_linewidth(1.0)
        # ax1[0].add_collection(los_coll)
        #
        # los_coll1 = PatchCollection(los_patches)
        # los_coll1.set_facecolor('None')
        # los_coll1.set_edgecolor('r')
        # los_coll1.set_linewidth(1.0)
        # ax1[1].add_collection(los_coll1)
        #
        # los_coll2 = PatchCollection(los_patches)
        # los_coll2.set_facecolor('None')
        # los_coll2.set_edgecolor('r')
        # los_coll2.set_linewidth(1.0)
        # ax1[2].add_collection(los_coll2)

        # PLOT SEP INTERSECTIONS POINTS ON OUTER DIVERTOR LEG
        for chord in self.chords:
            if chord.shply_intersects_w_sep:
                ax1[0].plot(chord.shply_intersects_w_sep.coords.xy[0][0],chord.shply_intersects_w_sep.coords.xy[1][0], 'rx', ms=8, mew=3.0, zorder=10)

        if self.diag == 'KT3':
            # PLOT PLASMA PROPERTIES ALONG LOS
            fig2, ax2 = plt.subplots(nrows=4, sharex=True)
            ax2[0].set_xlim(0,6)
            for chord in self.chords:
                if chord.v2unmod[0] >=2.74 and chord.v2unmod[0]<=2.76:
                    col = np.random.rand(3,1)
                    ax2[0].plot(chord.los_1d['l'], chord.los_1d['te'], '-', color=col, lw=2.0)
                    ax2[1].plot(chord.los_1d['l'], chord.los_1d['ne'], '-', color=col, lw=2.0)
                    ax2[2].plot(chord.los_1d['l'], chord.los_1d['n0'], '-', color=col, lw=2.0)
                    for key in self.spec_line_dict['1']['1']:
                        ax2[3].plot(chord.los_1d['l'], np.asarray(chord.los_1d['H_emiss_per_vol'][key]['excit'])+np.asarray(chord.los_1d['H_emiss_per_vol'][key]['recom']), '-', color=col)

            ax2[0].set_ylabel(r'$\mathrm{T_{e}}$')
            ax2[1].set_ylabel(r'$\mathrm{n_{e}}$')
            ax2[2].set_ylabel(r'$\mathrm{n_{0}}$')
            ax2[3].set_ylabel(r'$\mathrm{ph\/s^{-1}\/m^{-2}\/sr^{-1}\/nm^{-1}}$')
            ax2[3].set_xlabel('distance along LOS (m)')

            # PLOT H INTEGRATED EMISSION (compare both methods for summing emission - expect identical results)
            fig3, ax3 = plt.subplots(nrows=1)
            # col_dict = {'6561.9':'r','4860.6':'m', '4339.9':'orange', '4101.2':'darkgreen', '3969.5':'b'}
            for key in self.spec_line_dict['1']['1']:
                emiss = []
                coord = []
                for chord in self.chords:
                    emiss.append(chord.los_int['H_emiss'][key]['excit'] + chord.los_int['H_emiss'][key]['recom'])
                    coord.append(chord.v2unmod[0])
                col = np.random.rand(3,1)
                ax3.semilogy(coord, emiss, '-', color=col, lw=3.0)
            ax3.set_xlabel('R tile 5 (m)')
            ax3.set_ylabel(r'$\mathrm{ph\/s^{-1}\/m^{-2}\/sr^{-1}\/nm^{-1}}$')
                # ax3.plot(chord.v2unmod[0], np.sum(chord.los_1d['H_emiss']['6561.9']['recom']), 'rx', markersize=10),

if __name__=='__main__':
    print('')
