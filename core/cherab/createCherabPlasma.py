
import numpy as np
from raysect.core.math.function.float.function2d.interpolate import Discrete2DMesh

from cherab.edge2d.mesh_geometry import Edge2DMesh
from cherab.PESDT_addon import PESDTSimulation, PESDTElement, deuterium, EIRENEMesh

from ..utils import (read_amjuel_1d,
                     read_amjuel_2d,reactions, 
                     calc_cross_sections, 
                     calc_photon_rate,
                     A_coeff, 
                     H2_wavelength, 
                     wavelength,
                     calc_H2_band_emission, 
                     YACORA,
                     doppler_absorbance,
                     cen_absorbance,
                     ideal_absorbance)
import logging
logger = logging.getLogger(__name__)

BaseD = deuterium
D0 = PESDTElement("Deuterium", "D", 1.0, 2.0, BaseD)
D2 = PESDTElement("Deuterium2", "D2", 2.0, 4.0, BaseD)
D2vibr = PESDTElement("Deuterium2vibr", "D2", 2.0, 4.0, BaseD)
D3 = PESDTElement("Deuterium3+", "D3", 3.0, 6.0, BaseD)
M_D = 3.344e-27
def createCherabPlasma(PESDT, transitions: list, 
                       convert_denel_to_m3 = True, 
                       data_source = "AMJUEL", 
                       recalc_h2_pos = True,
                       mol_exc_bands = None,
                       opaque = False,
                       opaque_mode = 0,
                       opaque_bins = 21):
    '''
    Creates a cherab compatible PLASMA simulation object
    
    convert_denel_to_m3: When using adas data, you need to explicitly convert to the right units
    recalc_h2_pos: Recalculate the H2+ denisity according to AMJUEL H.12 2.0c. Should only be used, if
                   the molecular ion density is not available in the simualtion output
    
    '''

    ########################################################################
    # Start by loading in all the data from the PESDT object #

    num_cells = len(PESDT.cells)
    
    rv = np.zeros((num_cells, 4))
    zv = np.zeros((num_cells, 4))
    # Eirene uses triangles
    if PESDT.edge_code == "eirene":
        rv = np.zeros((num_cells, 3))
        zv = np.zeros((num_cells, 3))
    rc = np.zeros(num_cells)
    zc = np.zeros(num_cells)

    te = np.zeros(num_cells)
    ti = np.zeros(num_cells)
    t0 = np.zeros(num_cells)
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
        t0[ith_cell] = cell.te if cell.t0 is None else cell.t0
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
    if PESDT.edge_code in ["solps", "edge2d", "oedge"]:
        mesh = Edge2DMesh(rv, zv) #, rc, zc)
    elif PESDT.edge_code in ["eirene"]:
        mesh = EIRENEMesh(PESDT.data.vertices, PESDT.data.triangles)
    species_list = [(D0, 0), (D0, 1)]
    emission_keys = transitions
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
            
            em_n_exc, em_n_rec, em_mol, em_h2_pos, em_h3_pos, em_h_neg, tot = calc_photon_rate(transitions[i], te, ne, n0[:], n2[:], debug = True, mol_p_density = h2_pos_den[:],recalc_h2_pos = recalc_h2_pos)
            logger.info(f"Mean: {np.mean(tot)}")
            emission[0][transitions[i]] = em_n_exc
            emission[1][transitions[i]] = em_n_rec
            emission[2][transitions[i]] = em_mol
            emission[3][transitions[i]] = em_h2_pos
            emission[4][transitions[i]] = em_h3_pos
            emission[5][transitions[i]] = em_h_neg
        if mol_exc_bands is not None:
            logger.info("Precalculating molecular band emission")
            species_list.append((D2vibr, 0))
            emission_keys +=mol_exc_bands
            for band in mol_exc_bands:
                logger.info(f"   Band: {band}")
                em, den = calc_H2_band_emission(te, ne, n2[:], band=band)
                emission[6][band], species_density[6, :] = em, den
    elif data_source == "YACORA":
        yacora = YACORA(PESDT.YACORA_RATES_PATH)
        
        num_species = 6
        species_density = np.zeros((num_species, num_cells))
        
        species_list.append((D2, 0))
        # USE AMJUEL TO CALCULATE SPECIES DENSITY
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
        species_density[2,:] = n2[:]
        species_density[3, :] = h2_pos_den[:]
        species_density[4, :] = h3_pos_den[:]
        species_density[5, :] = h_neg_den[:]
        logger.info("Precalculating emission")    
        emission = [{} for _ in range(len(species_density))]
        for i in range(len(transitions)):
            logger.info(f"   Calculating emission for line: {transitions[i]}")
            h_emiss, h_rec_emiss, h2_emiss, h2_pos_emiss, h3_pos_emiss, hneg_emiss, tot = yacora.calc_photon_rate(transitions[i], te, ne, n0[:], n2[:], h2_pos_den, h3_pos_den, h_neg_den)
            logger.info(f"Mean: {np.mean(tot)}")
            emission[0][transitions[i]] = h_emiss
            emission[1][transitions[i]] = h_rec_emiss
            emission[2][transitions[i]] = h2_emiss
            emission[3][transitions[i]] = h2_pos_emiss
            emission[4][transitions[i]] = h3_pos_emiss
            emission[5][transitions[i]] = hneg_emiss
    else:
        #ADAS
        num_species = 2
        species_density = np.zeros((num_species, num_cells))
    
    if opaque:
        num_species +=1
        _species_density = np.zeros((num_species, num_cells))
        _species_density[:-1, :] = species_density
        species_density = _species_density # resize, no density for photons

        emission.append({})
        species_list.append((D0, 2))
        for tra in transitions:
            if PESDT.edge_code == "eirene":
                n0_N2 = PESDT.data.n0_ph2 # Contrib. of opacity to N=2
                n0_N3 = PESDT.data.n0_ph3 # Contrib. of opacity to N=3
            else:
                n0_N2 = np.zeros((num_cells,)) 
                n0_N3 = np.zeros((num_cells,)) 
            if tra[0] == 2:
                emission[7][tra] = n0_N2*A_coeff(tra)*1/(4.0*np.pi)
            elif tra[0] ==3:
                emission[7][tra] = n0_N3*A_coeff(tra)*1/(4.0*np.pi)
            else:
                emission[7][tra] = np.zeros((num_cells,))
        absorbance = emission.copy() # Same shape
        # Reset array
        

        if opaque_mode == 0:
            absorb_ = {}
            for tra in transitions:
                absorb_[tra] = ideal_absorbance(tra, None, species_density[2,:], M_D)
            for i in range(len(absorbance)):
                for key, _ in absorbance[i].keys():
                    absorbance[i][key] = absorb_[key] # absorbance is the same for each contribution
        elif opaque_mode == 1:
            try:
                t0 = PESDT.data.t0
            except:
                logger.warning("No Td available, using Ti")
                t0 = PESDT.data.ti
            absorb_ = {}
            for tra in transitions:

                absorb_[tra] = cen_absorbance(tra, t0, species_density[2,:], M_D)
            for i in range(len(absorbance)):
                for key, _ in absorbance[i].keys():
                    absorbance[i][key] = absorb_[key]
        elif opaque_mode == 2:
            
            try:
                t0 = PESDT.data.t0
            except:
                logger.warning("No Td available, using Ti")
                t0 = PESDT.data.ti
            absorb_ = {}
            
            for tra in transitions:
                
                absorb_[tra] = doppler_absorbance(tra, t0, species_density[2,:], M_D)
            for i in range(len(absorbance)):
                for key, _ in absorbance[i].keys():
                    absorbance[i][key] = absorb_[key]
        else:
            raise Exception("unknown opaque mode")
    else:
        absorbance = emission.copy() # Same shape
        for i in range(len(absorbance)):
            for key, values in absorbance[i].keys():
                absorbance[i][key] = np.zeros_like(values)
    species_density[0, :] = n0[:]  # neutral density D0
    species_density[1, :] = ni[:]  # ion density D+1
    
    neutral_temperature = np.zeros_like(species_density)
    neutral_temperature[0, :] = t0[:]
    neutral_temperature[1, :] = ti[:]

    for i in range(neutral_temperature.shape[0]-2):
        neutral_temperature[i+2, :] = t0[:] # Assing neutral atom temperature to all others

    print(species_list)

    sim = PESDTSimulation(mesh, species_list ) #[['D0', 0], ['D+1', 1]])
    sim.electron_temperature = te
    sim.electron_density = ne
    sim.ion_temperature = ti
    sim.neutral_temperature = neutral_temperature
    sim.species_density = species_density

    if data_source in ["AMJUEL", "YACORA"]:
        sim.emission = [emission_keys, emission]

    return sim
