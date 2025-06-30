
import numpy as np
from raysect.core.math.function.float.function2d.interpolate import Discrete2DMesh
from cherab.edge2d.mesh_geometry import Edge2DMesh
from cherab.PESDT_addon import PESDTSimulation, PESDTElement, deuterium

from .utils import read_amjuel_1d,read_amjuel_2d,reactions, calc_cross_sections, calc_photon_rate, H2_wavelength, calc_H2_band_emission, YACORA
import logging
logger = logging.getLogger(__name__)

BaseD = deuterium
D0 = PESDTElement("Deuterium", "D", 1.0, 2.0, BaseD)
D2 = PESDTElement("Deuterium2", "D2", 2.0, 4.0, BaseD)
D2vibr = PESDTElement("Deuterium2vibr", "D2", 2.0, 4.0, BaseD)
D3 = PESDTElement("Deuterium3+", "D3", 3.0, 6.0, BaseD)

def createCherabPlasma(PESDT, transitions: list, 
                       convert_denel_to_m3 = True, 
                       data_source = "AMJUEL", 
                       recalc_h2_pos = True,
                       mol_exc_bands = None):
    '''
    Creates a cherab compatible PLASMA simulation object
    
    convert_denel_to_m3: When using adas data, you need to explicitly convert to the right units
    load_mol_data: Load and pass molecular data based on AMJUEL rates to the object
        recalc_h2_pos: Recalculate the H2+ denisity according to AMJUEL H.12 2.0c. Should only be used, if
                       the molecular ion density is not available in the simualtion output
    quick (BETA): Pre calculate emission, and pass directly to cherab
    **kwargs: used in passing data required for 'quick'
    
    '''

    ########################################################################
    # Start by loading in all the data from the PESDT object #

    num_cells = len(PESDT.cells)

    rv = np.zeros((num_cells, 4))
    zv = np.zeros((num_cells, 4))
    rc = np.zeros(num_cells)
    zc = np.zeros(num_cells)

    te = np.zeros(num_cells)
    ti = np.zeros(num_cells)
    ne = np.zeros(num_cells)
    ni = np.zeros(num_cells)
    n0 = np.zeros(num_cells)
    n2 = np.zeros(num_cells)
    n2p = np.zeros(num_cells)
    multi = 1.0
    if convert_denel_to_m3:
        multi = 1e-6

    for ith_cell, cell in enumerate(PESDT.cells):
        # extract cell centres and vertices
        rc[ith_cell] = cell.R
        zc[ith_cell] = cell.Z

        if PESDT.edge_code == "solps":
            coords = np.array(cell.poly.exterior.coords).transpose()
            rv[ith_cell, :] = coords[0]
            zv[ith_cell, :] = coords[1]
        else:
            rv[ith_cell, :] = PESDT.data.rv[ith_cell, 0:4]
            zv[ith_cell, :] = PESDT.data.zv[ith_cell, 0:4]
        # Pull over plasma values to new CHERAB arrays

        te[ith_cell] = cell.te
        ti[ith_cell] = cell.te
        # Multiply by 1e-6, I think cherab wants densities in cm^-3
        
        ni[ith_cell] = cell.ni*multi
        ne[ith_cell] = cell.ne*multi
        n0[ith_cell] = cell.n0*multi
        n2[ith_cell] = cell.n2*multi
        n2p[ith_cell] = cell.n2p*multi

    #####################################################
    # Now load the simulation object with plasma values #
    rv = np.transpose(rv)
    zv = np.transpose(zv)

    species_list = [(D0, 0), (D0, 1)]
    
    if data_source == "AMJUEL":
        '''
        Calculate the H2+, H3+, and H- densities through AMJUEL rates, and add the molecular density 
        and derived densities to species

        '''

        logger.info("Loading H2, H2+, H3+ and H-")
        num_species = 6
        if mol_exc_bands is not None:
            logger.info(f"Allocating space for molecular band emission, num. bands {len(mol_exc_bands)}")
            #
            num_species += 1
        species_density = np.zeros((num_species, num_cells))
        species_list.append((D2, 0))
        
        reac = reactions(2) # The densities are independent of the hydrogenic excited state
        if recalc_h2_pos:
            MARc_h2_pos_den = read_amjuel_2d(reac["den_H2+"][0],reac["den_H2+"][1])
            h2_pos_den = calc_cross_sections(MARc_h2_pos_den, T = te, n = ne*1e-6)*n2
        else:
            h2_pos_den = n2p[:]
        species_list.append((D2, 1))
        
        MARc_h3_pos_den = read_amjuel_1d(reac["den_H3+"][0],reac["den_H3+"][1])
        h3_pos_den = calc_cross_sections(MARc_h3_pos_den, T = te, n = ne*1e-6)*n2*h2_pos_den/ne
        species_list.append((D3, 1))
        
        MARc_h_neg_den = read_amjuel_1d(reac["den_H-"][0],reac["den_H-"][1])
        h_neg_den = calc_cross_sections(MARc_h_neg_den, T = te, n = ne*1e-6)*n2
        species_list.append((D0, -1)) 

        species_density[2, :] = n2[:]  # Mol. density D2
        species_density[3, :] = h2_pos_den[:]
        species_density[4, :] = h3_pos_den[:]
        species_density[5, :] = h_neg_den[:]
        emission = [{} for _ in range(len(species_density))]
        logger.info("Precalculating emission")    
        for i in range(len(transitions)):
            logger.info(f"   Calculating emission for line: {transitions[i]}")
            em_n_exc, em_n_rec, em_mol, em_h2_pos, em_h3_pos, em_h_neg, _ = calc_photon_rate(transitions[i], te, ne, n0[:], n2[:], h2_pos_den[:], debug = True)
            emission[0][transitions[i]] = em_n_exc
            emission[1][transitions[i]] = em_n_rec
            emission[2][transitions[i]] = em_mol
            emission[3][transitions[i]] = em_h2_pos
            emission[4][transitions[i]] = em_h3_pos
            emission[5][transitions[i]] = em_h_neg
        if mol_exc_bands is not None:
            logger.info("Precalculating molecular band emission")
            species_list.append((D2vibr, 0))
            for band in mol_exc_bands:
                logger.info(f"   Band: {band}")
                emission[6][band], species_density[6, :] = calc_H2_band_emission(te, ne, n2[:], band=band)
    elif data_source == "YACORA":
        yacora = YACORA(PESDT.YACORA_RATES_PATH)
        
        num_species = 3
        species_density = np.zeros((num_species, num_cells))
        species_density[2,:] = n2[:]
        species_list.append((D2, 0))
        logger.info("Precalculating emission")    
        emission = [{} for _ in range(len(species_density))]
        for i in range(len(transitions)):
            logger.info(f"   Calculating emission for line: {transitions[i]}")
            h_emiss, h2_emiss = yacora.calc_photon_rate(transitions[i], te, ne, n0[:], n2[:])
            emission[0][transitions[i]] = h_emiss
            emission[1][transitions[i]] = np.zeros(h_emiss.shape)
            emission[2][transitions[i]] = h2_emiss
    else:
        #ADAS
        num_species = 2
        species_density = np.zeros((num_species, num_cells))

    species_density[0, :] = n0[:]  # neutral density D0
    species_density[1, :] = ni[:]  # ion density D+1
    edge2d_mesh = Edge2DMesh(rv, zv) #, rc, zc)

    print(species_list)

    sim = PESDTSimulation(edge2d_mesh, species_list ) #[['D0', 0], ['D+1', 1]])
    sim.electron_temperature = te
    sim.electron_density = ne
    sim.ion_temperature = ti
    sim.species_density = species_density

    if data_source in ["AMJUEL", "YACORA"]:
        sim.emission = emission

    return sim
