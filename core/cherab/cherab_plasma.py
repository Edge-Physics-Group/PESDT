import numpy as np
import os, sys, pickle, gzip
from math import sin, cos, pi, atan

from raysect.optical import World, AbsorbingSurface
from raysect.optical.observer import PinholeCamera, FibreOptic, RadiancePipeline0D, RadiancePipeline2D,SpectralRadiancePipeline0D
from raysect.core.math import Point3D, Vector3D, translate, rotate, rotate_basis#, Discrete2DMesh #Note: this was not being used, and caused a chrash (cannot be found)
from raysect.core import SerialEngine, MulticoreEngine
from cherab.core import Plasma, Species, Maxwellian, Line, elements
from cherab.core.model.plasma.impact_excitation import ExcitationLine
from cherab.core.model.plasma.recombination import RecombinationLine
from cherab.core.utility.conversion import PhotonToJ
#from cherab.jet.machine.cad_files import import_jet_mesh
from .jet_cad_mesh import import_jet_mesh
from .aug_cad_mesh import import_aug_mesh
from cherab.core.model.lineshape import GaussianLine
from cherab.PESDT_addon.LineShapes import StarkBroadenedLine, DeltaLine
from cherab.PESDT_addon.LineEmitters import DirectEmission, DirectEmissionMol, OpaqueGaussianDirectEmission, OpaqueDeltaDirectEmission, LineExcitation_AM, LineRecombination_AM, LineH2_AM, LineH2_pos_AM, LineH3_pos_AM, LineH_neg_AM
from cherab.PESDT_addon.continuo import Continuo

from cherab.PESDT_addon import PESDTLine, PESDTLineMol
from .cherab_AMJUEL_data import AMJUEL_Data
from .cherab_atomic_data import PESDT_ADAS_Data
from .createCherabPlasma import createHydrogenicCherabPlasma, createHydrogenicCherabPlasmaBolo, createZCherabPlasma, D0, D2, D3, D2vibr
from ..utils.JET_mesh_from_grid import create_toroidal_wall_from_points, modify_wall_polygon_for_observer,plot_wall_modification
from .D3D_mesh import construct_DIIID_mesh
import logging
logger = logging.getLogger(__name__)


class CherabPlasma():

    def __init__(self, PESDT_obj,
                 machine = "JET", 
                 include_reflections: bool = False, 
                 import_surfaces: bool = False, 
                 data_source = "AMJUEL", 
                 recalc_h2_pos: bool = True, 
                 transitions = None,
                 instrument_los_dict: dict = None,
                 bolo_los_dict: dict = None,
                 mol_exc_bands = None,
                 species = ["D"],
                 opaque = False,
                 opaque_mode = 0,):

        self.PESDT_obj = PESDT_obj
        self.machine = machine
        self.include_reflections = include_reflections
        self.import_surfaces = import_surfaces
        self.mesh_from_grid = not import_surfaces
        self.data_source = data_source
        self.ADAS_dict = PESDT_obj.ADAS_dict if self.data_source == "ADAS" else None
        self.recalc_h2_pos = recalc_h2_pos 
        self.transitions = transitions
        self.sim_type = PESDT_obj.edge_code
        self.instrument_los_dict = instrument_los_dict
        self.bolo_los_dict = bolo_los_dict
        self.species_list = species
        self.instrument_fibreoptics = {}
        self.stark_fibreoptics = {}
        self.continuum_fibreoptics = {}
        self.instrument_los_coords = {}
        self.cameras = {}
        self.bolos = {}
        self.mol_exc_bands = mol_exc_bands
        self.opaque = opaque
        self.opaque_mode = opaque_mode
        pesdt_home = os.environ.get('PESDT_HOME', os.path.expanduser('~') + "PESDT/")
        # Create CHERAB plasma from PESDT edge_codes object
        # Try loading for a pickled world definition
        if self.machine in ["JET", "AUG"]:
            if self.import_surfaces:
                logger.info(f"Reading {self.machine} mesh from pickle file")
                if self.include_reflections:
                    try:
                        with gzip.open(pesdt_home +f"PESDTCache/{self.machine}world.pkl.gz", "rb") as f:
                            self.world = pickle.load(f)
                        self.import_surfaces = False
                        logger.info("Mesh read!")
                    except Exception as e:
                        logger.error(f"Exception: {e}")
                        logger.info("Could not read raysect-world object from a pkl, creating a new one.")
                        self.world = World()
                        if self.machine == "JET": import_jet_mesh(self.world)
                        else: import_aug_mesh(self.world)
                        with gzip.open(pesdt_home + f"PESDTCache/{self.machine}world.pkl.gz", "wb") as f:
                             pickle.dump(self.world,f, protocol=pickle.HIGHEST_PROTOCOL)
                else: 
                    try:
                        with gzip.open(pesdt_home +f"PESDTCache/{self.machine}world_no_refl.pkl.gz", "rb") as f:
                            self.world = pickle.load(f)
                        self.import_surfaces = False
                        logger.info("Mesh read!")
                    except Exception as e:
                        logger.error(f"Exception: {e}")
                        logger.info("Could not read raysect-world with no reflections object from a pkl, creating a new one.")
                        self.world = World()
                        if self.machine == "JET": import_jet_mesh(self.world, override_material=AbsorbingSurface())
                        else: import_aug_mesh(self.world, material=AbsorbingSurface())
                        with gzip.open(pesdt_home + f"PESDTCache/{self.machine}world_no_refl.pkl.gz", "wb") as f:
                            pickle.dump(self.world,f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                self.world = World()
                if self.mesh_from_grid:
                    # Find the highest observer
                    observer_coords = []
                    for instrument, los in self.instrument_los_dict.items():
                        observer_coords.append(los[0][0]) # Index 0 is p1
                    observer_pos = observer_coords[0]
                    for obs_pos in observer_coords:
                        if obs_pos[1] > observer_pos[1]:
                            observer_pos = obs_pos
                    #determine safety_distance
                    dist = 0.0
                    dist = np.linalg.norm(np.array(observer_coords) - np.array(observer_pos), axis = 1)
                    max_dist = np.max(dist)
                    safety_distance = 0.3
                    if max_dist > safety_distance: safety_distance = max_dist +0.01
        
                    # Modify the polygons, create mesh
                    # Note that by default, the mesh is composed of W only
                    mod_polygons = modify_wall_polygon_for_observer(self.PESDT_obj.data.wall_poly.get_xy(), observer_pos, safety_distance = safety_distance )
                    #plot_wall_modification(self.PESDT_obj.data.wall_poly.get_xy(), mod_polygons, observer_pos)
                    self.mesh = create_toroidal_wall_from_points(mod_polygons, self.world)
        elif machine == "DIIID":
            self.world = World()
            construct_DIIID_mesh(self.world)
        if (len(self.instrument_los_dict)>0 or len(self.cameras)> 0):
            self.plasma = self.gen_cherab_plasma()
        if len(self.bolo_los_dict)>0: 
            self.bolo_plasma = self.gen_cherab_bolo_plasma()
    def gen_cherab_plasma(self):

        # Load PESDT object into cherab_edge2d module, which converts the edge_codes grid to cherab
        # format, and populates cherab plasma parameters
        convert_to_m3 = not (self.data_source in ["AMJUEL", "YACORA"])
        cherab = createHydrogenicCherabPlasma(self.PESDT_obj,
                                    transitions= self.transitions,
                                    convert_denel_to_m3 = convert_to_m3, 
                                    data_source=self.data_source, 
                                    recalc_h2_pos = self.recalc_h2_pos, 
                                    mol_exc_bands= self.mol_exc_bands,
                                    opaque= self.opaque,
                                    opaque_mode = self.opaque_mode)
        # create atomic data source
        plasma = cherab.create_plasma(parent=self.world, opaque = self.opaque)
        if self.data_source == "AMJUEL":
            PESDT_AMJUEL_data = AMJUEL_Data()
            logger.info("Using AMJUEL")
            
            plasma.atomic_data = PESDT_AMJUEL_data
        elif self.data_source == "YACORA":
            plasma.atomic_data = AMJUEL_Data() # Using AMJUEL data here is fine, because emission is directly passed to cherab; it does not do anything
            logger.info("Using YACORA")
        else:
            #ADAS
            PESDT_adas = PESDT_ADAS_Data(self.ADAS_dict)
            logger.info("Using ADAS")
            plasma.atomic_data = PESDT_adas

        return plasma

    def gen_cherab_bolo_plasma(self):
    
            # Load PESDT object into cherab_edge2d module, which converts the edge_codes grid to cherab
            # format, and populates cherab plasma parameters
            convert_to_m3 = not (self.data_source in ["AMJUEL", "YACORA"])
            cherab = createCherabPlasmaBolo(self.PESDT_obj,
                                        transitions= self.transitions,
                                        convert_denel_to_m3 = convert_to_m3, 
                                        data_source=self.data_source, 
                                        recalc_h2_pos = self.recalc_h2_pos, 
                                        mol_exc_bands= self.mol_exc_bands,
                                        opaque= self.opaque,
                                        opaque_mode = self.opaque_mode)
    
            # create atomic data source
            plasma = cherab.create_plasma(parent=self.world, opaque = self.opaque)
            if self.data_source == "AMJUEL":
                PESDT_AMJUEL_data = AMJUEL_Data()
                logger.info("Using AMJUEL")
                
                plasma.atomic_data = PESDT_AMJUEL_data
            elif self.data_source == "YACORA":
                plasma.atomic_data = AMJUEL_Data() # Using AMJUEL data here is fine, because emission is directly passed to cherab; it does not do anything
                logger.info("Using YACORA")
            else:
                #ADAS
                PESDT_adas = PESDT_ADAS_Data(self.ADAS_dict)
                logger.info("Using ADAS")
                plasma.atomic_data = PESDT_adas
    
            return plasma

    def define_bolometer_plasma_mode(self, ):

        return

    def define_continuum_plasma_model(self):
        h_line = PESDTLine(D0, 0, (4,2))
        self.plasma.models = [Continuo(h_line, lineshape = GaussianLine)]

    def define_stark_plasma_model(self, transition, **kwargs):
        line_emitter = DirectEmission
        lineshape = StarkBroadenedLine
        model_list = []
        if  kwargs.get("include_excitation", True):
            h_line = PESDTLine(D0, 0, transition)
            model_list.append(line_emitter(h_line, lineshape=lineshape))
        if  kwargs.get("include_recombination", True):
            h_line = PESDTLine(D0, 1, transition)
            model_list.append(line_emitter(h_line, lineshape=lineshape))
        if  kwargs.get("include_H2", True):
            h_line = PESDTLine(D2, 0, transition)
            model_list.append(line_emitter(h_line, lineshape=lineshape))
        if  kwargs.get("include_H2_pos", True):
            h_line = PESDTLine(D2, 1, transition) 
            model_list.append(line_emitter(h_line, lineshape=lineshape))
        if  kwargs.get("include_H3_pos", True):
            h_line = PESDTLine(D3, 1, transition) 
            model_list.append(line_emitter(h_line, lineshape=lineshape))
        if  kwargs.get("include_H_neg", False):
            h_line = PESDTLine(D0, -1, transition) 
            model_list.append(line_emitter(h_line, lineshape=lineshape))
        if  kwargs.get("include_ph", False) and self.opaque:
            h_line = PESDTLine(D0, 2, transition)
            model_list.append(line_emitter(h_line, lineshape=lineshape))
        self.plasma.models = model_list

    def define_Hydrogenic_plasma_model(self, transition=(2, 1),
                            include_excitation=False, include_recombination=False,
                            include_H2 = False, include_H2_pos = False, include_H_neg = False,
                            include_H3_pos = False, include_ph = False, data_source = "AMJUEL",
                            include_mol_exc = False):
        # Define one transition at a time and 'observe' total radiance
        # If multiple transitions are fed into the plasma object, the total
        # observed radiance will be the sum of the defined spectral lines.
        line_emitter = DirectEmission
        lineshape = None
        model_list = []
        if data_source in ["AMJUEL", "YACORA"]:
            if include_excitation:
                h_line = PESDTLine(D0, 0, transition)
                model_list.append(line_emitter(h_line, lineshape=lineshape))
            if include_recombination:
                h_line = PESDTLine(D0, 1, transition)
                model_list.append(line_emitter(h_line, lineshape=lineshape))
            if include_H2:
                h_line = PESDTLine(D2, 0, transition)
                model_list.append(line_emitter(h_line, lineshape=lineshape))
            if include_H2_pos:
                h_line = PESDTLine(D2, 1, transition) 
                model_list.append(line_emitter(h_line, lineshape=lineshape))
            if include_H3_pos:
                h_line = PESDTLine(D3, 1, transition) 
                model_list.append(line_emitter(h_line, lineshape=lineshape))
            if include_H_neg:
                h_line = PESDTLine(D0, -1, transition) 
                model_list.append(line_emitter(h_line, lineshape=lineshape))
            if include_ph and self.opaque:
                h_line = PESDTLine(D0, 2, transition)
                model_list.append(line_emitter(h_line, lineshape=lineshape))
            if include_mol_exc and data_source != "YACORA":
                h_line = PESDTLineMol(D2vibr, 0, transition)
                model_list.append(DirectEmissionMol(h_line, lineshape = lineshape))
        else:
            h_line = Line(D0, 0, transition)
            model_list = []
            if include_excitation:
                model_list.append(ExcitationLine(h_line, lineshape=lineshape))
            if include_recombination:
                model_list.append(RecombinationLine(h_line, lineshape=lineshape))
            self.plasma.models = model_list
        self.plasma.models = model_list
        
    def setup_observers(self, pixel_samples = 1000, num_processes = 1):
        """
        Creates fibreOptics, which are used to integrate radiance along a line-of-sight using Raysect's RadiancePipeline0D.
        Parameters:
            los_p1, los_p2: (R, Z) coordinates of the LOS start and end.
            los_w1: LOS entrance diameter (not used directly here but could be logged or used in fiber setup).
            los_w2: LOS exit diameter (used to calculate acceptance angle).
            pixel_samples: Number of samples per pixel (Monte Carlo rays).
        """
        for instrument in self.instrument_los_dict.keys():
            fibreoptics = []
            los_coords = []
            for los_p1, los_p2, los_w1, los_w2 in self.instrument_los_dict[instrument]:
                los_coords.append(los_p2)
                 
                 # Define LOS direction and observer origin using KT1V viewport angle
                theta = -45.61 / 360 * (2 * pi)
                origin = Point3D(los_p1[0] * cos(theta), los_p1[0] * sin(theta), los_p1[1])
                endpoint = Point3D(los_p2[0] * cos(theta), los_p2[0] * sin(theta), los_p2[1])
                direction = origin.vector_to(endpoint)

                # Calculate acceptance angle from los_w2 and LOS length
                chord_length = origin.distance_to(endpoint)
                acceptance_angle = atan((los_w2 / 2.0) / chord_length) * 180. / np.pi

                # Setup radiance pipeline
                pipeline = RadiancePipeline0D(accumulate = False)
                fibre = FibreOptic(
                    pipelines=[pipeline],
                    acceptance_angle=acceptance_angle,
                    radius=0.01,  # Default pinhole size of 1 cm
                    pixel_samples=pixel_samples,
                    min_wavelength = 1.0,
                    max_wavelength = 1e20,
                    spectral_rays=1,  # Not used in RadiancePipeline0D, but required by FibreOptic
                    transform=translate(*origin) * rotate_basis(direction, Vector3D(1, 0, 0)),
                    parent=self.world)
                if num_processes > 1:
                    fibre.render_engine = MulticoreEngine(processes=int(num_processes))
                else:
                    fibre.render_engine = SerialEngine()
                # Create fibre optic observer
                fibreoptics.append((pipeline, fibre))
                
            self.instrument_fibreoptics[instrument] = fibreoptics
            self.instrument_los_coords[instrument] = los_coords
        return
    
    def setup_spectral_observers(self,
                                 min_wavelength_nm, 
                                 max_wavelength_nm, 
                                 destination = "stark",
                                 pixel_samples = 1000, 
                                 spectral_rays = 100, 
                                 spectral_bins = 100,
                                 num_processes = 1):
        """
        Creates fibreOptics, which are used to integrate spectral radiance along a line-of-sight using SpectralRadiancePipeline0D.

        Parameters:
            los_p1, los_p2: (R, Z) coordinates of the LOS start and end.
            los_w1, los_w2: entrance and exit diameters of LOS (used to define acceptance angle).
            min_wavelength_nm, max_wavelength_nm: spectral range in nanometers.
            spectral_bins: number of wavelength bins.
            pixel_samples: number of Monte Carlo samples per pixel.
            spectral_rays: number of rays per wavelength bin.
        """
        for instrument in self.instrument_los_dict.keys():
            fibreoptics = []
            los_coords = []
            for los_p1, los_p2, los_w1, los_w2 in self.instrument_los_dict[instrument]:
                los_coords.append(los_p2)
                # Define LOS direction and observer origin using KT1V viewport angle
                theta = -45.61 / 360 * (2 * pi)
                origin = Point3D(los_p1[0] * cos(theta), los_p1[0] * sin(theta), los_p1[1])
                endpoint = Point3D(los_p2[0] * cos(theta), los_p2[0] * sin(theta), los_p2[1])
                direction = origin.vector_to(endpoint)

                # Calculate acceptance angle from los_w2 and LOS length
                chord_length = origin.distance_to(endpoint)
                acceptance_angle =  atan((los_w2 / 2.0) / chord_length) * 180. / np.pi

                # Setup radiance pipeline
                pipeline = SpectralRadiancePipeline0D(display_progress=False, accumulate = False)

                # Create fibre optic observer
                fibre = FibreOptic(
                    pipelines=[pipeline],
                    acceptance_angle=acceptance_angle,
                    radius=0.01,  # 1 cm pinhole
                    pixel_samples=pixel_samples,
                    spectral_rays=spectral_rays,
                    spectral_bins=spectral_bins,
                    min_wavelength = 1.0,
                    max_wavelength = 1e20,
                    transform=translate(*origin) * rotate_basis(direction, Vector3D(1, 0, 0)),
                    parent=self.world
                )
                # Set wavelength bounds
                fibre.min_wavelength = min_wavelength_nm
                fibre.max_wavelength = max_wavelength_nm
                if num_processes > 1:
                    fibre.render_engine = MulticoreEngine(processes=int(num_processes))
                else:
                    fibre.render_engine = SerialEngine()
                fibreoptics.append((pipeline, fibre))

            if destination == "stark":
                self.stark_fibreoptics[instrument] = fibreoptics
            elif destination == "continuum":
                self.continuum_fibreoptics[instrument] = fibreoptics
            else:
                #Assume the user want's to do spectral bins for regular line-emission
                self.instrument_fibreoptics[instrument] = fibreoptics
            self.instrument_los_coords[instrument] = los_coords
        return
    
    def setup_cameras(self, pixel_samples = 250, num_processes = 8):
        
        for instrument, defs in self.camera_los_dict.items():
            pipeline = RadiancePipeline2D(display_progress=False, accumulate = False)
            los_p1 = defs["p1"]
            los_p2 = defs["p2"]
            theta_p1 = defs["p1"][2]
            theta_p2 = defs["p2"][2]

            origin = Point3D(los_p1[0] * cos(theta_p1), los_p1[0] * sin(theta_p1), los_p1[1])
            endpoint = Point3D(los_p2[0] * cos(theta_p2), los_p2[0] * sin(theta_p2), los_p2[1])
            direction = origin.vector_to(endpoint)
            fov = defs["angle"]

            ccd = PinholeCamera(pixels = defs["pixels"],pipelines=[pipeline],
                fov=fov,
                transform=translate(*origin) * rotate_basis(direction, Vector3D(0, 0, -1)),
                parent=self.world)
            ccd.pixel_samples = pixel_samples
            ccd.min_wavelength = 1.0
            ccd.max_wavelength = 1e20

            if num_processes > 1:
                ccd.render_engine = MulticoreEngine(processes=int(num_processes))
            else:
                ccd.render_engine = SerialEngine()
            self.cameras[instrument] = (pipeline, ccd)

        return

    def setup_bolometers(self, pixel_samples, num_processes = 1):

        return
    def integrate_instrument(self, instrument):
        """
        Iterates over the fibreOptics of an instrument
        
        Returns:
            Radiance in ph/s/(sr*m^2)
        """
        res = []
        los_coords = self.instrument_los_coords[instrument]
        i = 0
        for pipeline, fibre in self.instrument_fibreoptics[instrument]:
            logger.info(f"LOS: {los_coords[i]}")
            # Perform ray tracing
            fibre.observe()
            res.append([pipeline.value.mean, pipeline.value.variance])
            i+=1
        # Return scalar radiance
        return res
    
    def observe_camera(self, instrument):
        pipeline, camera = self.cameras[instrument]
        camera.observe()
        # pipeline.frame is a RaysectImage with per-pixel statistics
        frame = pipeline.frame

        return [frame.mean, frame.variance]

    def integrate_instrument_spectral(self, instrument, destination):
        '''
        Iterates over the fibreOptics of an instrument
        Returns:
            Tuple of (spectrum, wavelengths):
                - spectrum: ndarray of ph / (sr * m² * nm)
                - wavelengths: ndarray in nanometers
        '''
        src = None
        if destination == "stark":
            src = self.stark_fibreoptics[instrument]
        elif destination == "continuum":
            src = self.continuum_fibreoptics[instrument]
        else:
            src = self.instrument_fibreoptics[instrument]
        res_mean = []
        res_wl = []
        los_coords = self.instrument_los_coords[instrument]
        i = 0
        for pipeline, fibre in src:
            # Perform ray tracing
            logger.info(f"LOS: {los_coords[i]}")
            fibre.observe()
            res_mean.append(list(pipeline.samples.mean))
            res_wl.append(list(pipeline.wavelengths))
            i+=1
        # Return scalar radiance
        return res_mean, res_wl
        
    
    def W_str_m2_to_ph_s_str_m2(self, centre_wav_nm, W_str_m2_val):
        h = 6.626E-34 # J.s
        c = 299792458.0 # m/s
        center_wav_m = centre_wav_nm * 1.0e-09
        return W_str_m2_val * center_wav_m / (h*c)