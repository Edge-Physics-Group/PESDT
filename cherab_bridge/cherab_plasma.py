
from ast import Raise
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.constants import atomic_mass, electron_mass
from math import sin, cos, pi, atan

from raysect.primitive import Cylinder
from raysect.optical import World, AbsorbingSurface
from raysect.optical.observer import PinholeCamera, FibreOptic, RadiancePipeline0D, SpectralRadiancePipeline0D
from raysect.core.math import Point3D, Vector3D, translate, rotate, rotate_basis#, Discrete2DMesh #Note: this was not being used, and caused a chrash (cannot be found)
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

from cherab.openadas import OpenADAS

from cherab.core import Plasma, Species, Maxwellian, Line, elements
from cherab.core.math.mappers import AxisymmetricMapper
from cherab.core.math import Constant3D, ConstantVector3D
#from cherab.core.model.lineshape import StarkBroadenedLine
from cherab.core.model.plasma.impact_excitation import ExcitationLine
from cherab.core.model.plasma.recombination import RecombinationLine
from cherab.core.utility.conversion import PhotonToJ
from cherab.jet.machine.cad_files import import_jet_mesh


import cherab.PESDT_addon.molecules as molecules
from cherab.PESDT_addon.stark import StarkBroadenedLine
from cherab.PESDT_addon.LineEmitters import LineExcitation_AM, LineRecombination_AM, LineH2_AM, LineH2_pos_AM, LineH3_pos_AM, LineH_neg_AM

#from cherab.amjuel_data.AMJUEL_data import AMJUEL_Data

from cherab_bridge import PESDT_ADAS_Data
from cherab_bridge import  AMJUEL_Data
#from PESDT.cherab_bridge.cherab_AMJUEL_data import LineExcitation_AM, LineRecombination_AM, LineH2_AM, LineH2_pos_AM, LineH3_pos_AM, LineH_neg_AM

#from PESDT.cherab_bridge.continuo import Continuo  
from cherab.PESDT_addon.continuo import Continuo
from cherab_bridge import load_edge2d_from_PESDT
from solps_cherab_bridge import load_solps_from_PESDT
#from PESDT.cherab_bridge import molecules

class CherabPlasma():

    def __init__(self, PESDT_obj, ADAS_dict, include_reflections=False, import_jet_surfaces = False, use_AMJUEL = False, recalc_h2_pos =True, transitions = None):

        self.PESDT_obj = PESDT_obj
        self.include_reflections = include_reflections
        self.import_jet_surfaces = import_jet_surfaces
        self.ADAS_dict = ADAS_dict
        self.use_AMJUEL = use_AMJUEL
        self.recalc_h2_pos = recalc_h2_pos 
        self.transitions = transitions
        self.sim_type = PESDT_obj.edge_code

        # Create CHERAB plasma from PESDT edge_codes object
        self.world = World()
        self.plasma = self.gen_cherab_plasma()

    def gen_cherab_plasma(self):

        # Load PESDT object into cherab_edge2d module, which converts the edge_codes grid to cherab
        # format, and populates cherab plasma parameters
        convert_to_m3 = not (self.use_AMJUEL)
        if (self.sim_type == "edge2d"):
            cherab = load_edge2d_from_PESDT(self.PESDT_obj, convert_denel_to_m3 = convert_to_m3, load_mol_data = self.use_AMJUEL, recalc_h2_pos = self.recalc_h2_pos)
        elif (self.sim_type == "solps"):
            cherab = load_solps_from_PESDT(self.PESDT_obj, convert_denel_to_m3 = convert_to_m3, load_mol_data = self.use_AMJUEL, recalc_h2_pos = self.recalc_h2_pos)
        else:
            Raise("Not a valid edge code!")
        if self.import_jet_surfaces:
            if self.include_reflections:
                import_jet_mesh(self.world)
            else:
                import_jet_mesh(self.world, override_material=AbsorbingSurface())

        # create atomic data source
        plasma = cherab.create_plasma(parent=self.world)
        if self.use_AMJUEL:
            PESDT_AMJUEL_data = AMJUEL_Data()
            print("Using AMJUEL")
            print(plasma)
            plasma.atomic_data = PESDT_AMJUEL_data
        else:
            PESDT_adas = PESDT_ADAS_Data(self.ADAS_dict)
            print(plasma)
            plasma.atomic_data = PESDT_adas

        return plasma


    def define_plasma_model(self, atnum=1, ion_stage=0, transition=(2, 1),
                            include_excitation=True, include_recombination=False,
                            include_H2 = False, include_H2_pos = False, include_H_neg = False,
                            include_H3_pos = False, use_tot = False, use_AMJUEL = False,
                            include_stark=False, include_ff_fb=False):
        # Define one transition at a time and 'observe' total radiance
        # If multiple transitions are fed into the plasma object, the total
        # observed radiance will be the sum of the defined spectral lines.
        
        
        
            
        
        if include_stark:
            lineshape = StarkBroadenedLine
        else:
            lineshape = None

        # Only deuterium supported at the moment
        if atnum == 1:
            
            if use_AMJUEL:
                model_list = []
                if use_tot:
                    model_list.append()
                else:
                    if include_excitation:
                        h_line = Line(elements.deuterium, 0, transition)
                        model_list.append(LineExcitation_AM(h_line, lineshape=lineshape)) #, plasma=self.plasma, atomic_data=self.plasma.atomic_data
                    if include_recombination:
                        h_line = Line(elements.deuterium, 0, transition)
                        model_list.append(LineRecombination_AM(h_line, lineshape=lineshape))
                    if include_H2:
                        h_line = Line(molecules.Deuterium2, 0, transition)
                        model_list.append(LineH2_AM(h_line, lineshape=lineshape))
                    if include_H2_pos:
                        h_line = Line(molecules.Deuterium2, 0, transition) # Increment charge by one 
                        model_list.append(LineH2_pos_AM(h_line, lineshape=lineshape))
                    if include_H_neg:
                        h_line = Line(elements.hydrogen, 0, transition) #Implemented via H proxy
                        model_list.append(LineH_neg_AM(h_line, lineshape=lineshape))
                    if include_ff_fb:
                        h_line = Line(elements.deuterium, 0, transition)
                        model_list.append(Continuo(h_line, lineshape = lineshape))
                self.plasma.models = model_list
                    
            else:
                h_line = Line(elements.deuterium, 0, transition)   
                if include_ff_fb:
                    if include_excitation and include_recombination:
                        self.plasma.models = [ExcitationLine(h_line, lineshape=lineshape),
                                            RecombinationLine(h_line, lineshape=lineshape),
                                            Continuo()]
                    elif include_excitation and not include_recombination:
                        self.plasma.models = [ExcitationLine(h_line, lineshape=lineshape),
                                            Continuo()]
                    elif not include_excitation and include_recombination:
                        self.plasma.models = [RecombinationLine(h_line, lineshape=lineshape),
                                            Continuo()]
                    else:
                        self.plasma.models = [Continuo()]
                else:
                    if include_excitation and include_recombination:
                        self.plasma.models = [ExcitationLine(h_line, lineshape=lineshape),
                                            RecombinationLine(h_line, lineshape=lineshape)]
                    elif include_excitation and not include_recombination:
                        self.plasma.models = [ExcitationLine(h_line, lineshape=lineshape)]
                    elif not include_excitation and include_recombination:
                        self.plasma.models = [RecombinationLine(h_line, lineshape=lineshape)]
                    else:
                        sys.exit('Cherab plasma model must include either (or both) excitation or recombination.')
        else:
            print('Cherab bridge only supports deuterium.')

    def W_str_m2_to_ph_s_str_m2(self, centre_wav_nm, W_str_m2_val):
        h = 6.626E-34 # J.s
        c = 299792458.0 # m/s
        center_wav_m = centre_wav_nm * 1.0e-09
        return W_str_m2_val * center_wav_m / (h*c)

    def integrate_los(self, los_p1, los_p2, los_w1, los_w2,
                      min_wavelength_nm, max_wavelength_nm,
                      spectral_bins=1000, pixel_samples=100,
                      display_progress=False, no_avg = False):
        '''
        los_p1 and los_p2 are R,Z coordinates
        los_w1: origin diameter
        los_w2: diameter at los end point

        The observer needs to be placed into a real viewport, so that it is not obstructedd by the 
        vessel walls. Currently the default is the KT1V viewport, which should be fine for instruments
        located in other top viewports

        '''

        ## Here instead I (bloman) put in the real KS3H/V, KT1H coordinates:
        #
        # Original KS3H LOS:
        if (los_p1 == [6.13, 0.341]) & (los_p2 == [1.768, 0.108]):
            origin = Point3D(3.406, 5.097, 0.341)
            endpoint = Point3D(1.060, 1.415, 0.108)
        # "fake" KS3H limiter edge:
        elif (los_p1 == [6.13, 0.341]) & (los_p2 == [1.768, 0.110]):
            origin = Point3D(3.406, 5.097, 0.341)
            endpoint = Point3D(0.984, 1.473, 0.110)
        # "fake" KS3H limiter center:
        elif (los_p1 == [6.13, 0.341]) & (los_p2 == [1.768, 0.112]):
            origin = Point3D(3.406, 5.097, 0.341)
            endpoint = Point3D(0.855, 1.549, 0.112)
        # KS3H ch12 limiter edge
        elif (los_p1 == [6.13, 0.341]) & (los_p2 == [1.768, 0.196]):
            origin = Point3D(3.406, 5.097, 0.341)
            endpoint = Point3D(0.984, 1.473, 0.196)

        # KS3V real channel
        elif (los_p1 == [3.115, 3.354]) & (los_p2 == [3.251, -1.204]):
            origin = Point3D(0.6077, 3.0551, 3.3543)
            endpoint = Point3D(0.6721, 3.1814, -1.2045)
        # "fake" KS3V channel, looking behind vertical tile:
        elif (los_p1 == [3.115, 3.354]) & (los_p2 == [3.401, -1.204]):
            origin = Point3D(0.6077, 3.0551, 3.3543)
            endpoint = Point3D(0.70298, 3.32756, -1.2045)

        # KT1J, aka "fake" KT1H diag:
        elif (los_p1 == [6.238, -0.74]):
            ## This is the part which toroidally places the diag
            theta = -44.09 / 360 * (2 * pi)
            origin = Point3D(los_p1[0] * cos(theta), los_p1[0] * sin(theta), los_p1[1])
            endpoint = Point3D(los_p2[0] * cos(theta), los_p2[0] * sin(theta), los_p2[1])

        else:
            ## This is the part which toroidally places KT1V for Bart's divertor studies
            ##
            # Not sure what this does exactly. Need to ask Matt.
            # -45.61deg=-0.796rad
            theta = -45.61 / 360 * (2 * pi)
            origin = Point3D(los_p1[0] * cos(theta), los_p1[0] * sin(theta), los_p1[1])
            endpoint = Point3D(los_p2[0] * cos(theta), los_p2[0] * sin(theta), los_p2[1])

        # Calculating direction from origin to endpoint:
        direction = origin.vector_to(endpoint)

        # Determine los cone angle (acceptance_angle)
        # chord_length = ((los_p1[1]-los_p2[1])**2 + (los_p1[0]-los_p2[0])**2)**0.5
        chord_length = origin.distance_to(endpoint) # This is to have the correct acceptance angle after shortening LOSs
        acceptance_angle = 2. * atan( (los_w2/2.0) / chord_length) * 180. / np.pi

        # Define pipelines and observe
        # "total radiance" is not total radiance, its something else. Kept here because the code crahshes without it (perhaps no list when only 1 pipeline is given?)
        #total_radiance = RadiancePipeline0D()
        spectral_radiance = SpectralRadiancePipeline0D(display_progress=display_progress)
        fibre = FibreOptic( [ spectral_radiance], acceptance_angle=acceptance_angle,  #total_radiance,
                           radius=0.01, #width of the pinhole is not recorded anywhere, assume 1cm?
                           spectral_bins=spectral_bins, spectral_rays=1, pixel_samples=pixel_samples,
                           transform = translate(*origin) * rotate_basis(direction, 
                           Vector3D(1, 0, 0)), parent = self.world)
        fibre.min_wavelength = min_wavelength_nm
        fibre.max_wavelength = max_wavelength_nm
        fibre.observe()

        # convert from W/str/m^2 to ph/s/str/m^2
        centre_wav_nm = (max_wavelength_nm - min_wavelength_nm)/2.0
        #total_radiance_ph_s = PhotonToJ.inv(total_radiance.value.mean, centre_wav_nm)
        if no_avg:
            return spectral_radiance.samples.mean, spectral_radiance.wavelengths

        if not (self.use_AMJUEL):
            spectral_radiance_ph_s = PhotonToJ.inv(spectral_radiance.samples.mean, spectral_radiance.wavelengths)
        else:
            spectral_radiance_ph_s = spectral_radiance.samples.mean
        #Average over spectral bins
    
        spectral_radiance_ph_s = np.sum(spectral_radiance_ph_s)*(max_wavelength_nm - min_wavelength_nm)/spectral_bins

        return spectral_radiance_ph_s, spectral_radiance.wavelengths #total_radiance_ph_s, 
