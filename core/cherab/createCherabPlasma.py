
import numpy as np
from cherab.PESDT_addon import PESDTSimulation, PESDTElement, deuterium, EIRENEMesh, QuadMesh
from cherab.core.atomic.elements import helium, beryllium, carbon, nitrogen, neon,tungsten
from scipy.constants import h, c
from ..utils import (read_amjuel_1d,
                     read_amjuel_2d,reactions, 
                     calc_cross_sections, 
                     calc_photon_rate,
                     A_coeff,
                     continuo_read, 
                     H2_wavelength, 
                     wavelength,
                     calc_H2_band_emission, 
                     YACORA,
                     doppler_absorbance,
                     cen_absorbance,
                     ideal_absorbance,
                     ADF15,
                     ADF11,
                     continuov_)
from ..database import spectroscopic_lines_db
import logging
logger = logging.getLogger(__name__)

sdb = spectroscopic_lines_db()

BaseD = deuterium
D0 = PESDTElement("Deuterium", "D", 1.0, 2.0, BaseD)
D2 = PESDTElement("Deuterium2", "D2", 2.0, 4.0, BaseD)
D2vibr = PESDTElement("Deuterium2vibr", "D2", 2.0, 4.0, BaseD)
D3 = PESDTElement("Deuterium3+", "D3", 3.0, 6.0, BaseD)

He = PESDTElement("Helium", "He", 2.0, 4.0, helium)
Be = PESDTElement("Beryllium", "Be", 9.0, 9.0, beryllium)
C = PESDTElement("Carbon", "C", 12.0, 12.0, carbon)
N = PESDTElement("Nitrogen", "N", 7.0, 14.0, nitrogen)
N2 = PESDTElement("Nitrogen2", "N2", 14.0, 28.0, nitrogen)
Ne = PESDTElement("Neon", "Ne", 10, 20.18, neon)
W = PESDTElement("Tungsten", "W", 74.0, 183.84, tungsten)

ELEMENT_DICT = {"He": He, "Be": Be, "C": C, "N": N, "N2": N2, "Ne": Ne, "W": W}

M_D = 3.344e-27

def get_base_species_and_charge(species: str, plasma_species: list[str]):
    ps_ = None
    idx = 0
    for ps in plasma_species:
        if ps in species:
            ps_ = ps
            break
        idx +=1
    charge = int(species[len(ps_):])

    return ps_, charge, idx

def get_num_charge_states(species: str):
    return int(sdb.data_full[species]["ATOM_NUM"])

def create_cherab_mesh(PESDT):
    rv: np.ndarray = None
    zv: np.ndarray = None
    mesh = None

    if PESDT.edge_code in ["solps", "edge2d", "oedge"]:
        rv = np.transpose(PESDT.data.rv[:, 0:4])
        zv = np.transpose(PESDT.data.zv[:, 0:4])
        mesh = QuadMesh(rv, zv) 
    elif PESDT.edge_code in ["eirene"]:
        mesh = EIRENEMesh(PESDT.data.vertices, PESDT.data.triangles)

    return mesh

def createZCherabPlasma(PESDT, species_transitions: dict):
    '''
    Creates a cherab compatible PLASMA simulation object for any species using OpenADAS rates
    
    '''
    num_cells = len(PESDT.data.te)
    mesh = create_cherab_mesh(PESDT)
    plasma_species: list[str] = PESDT.species 
    te = PESDT.data.te #np.zeros(num_cells)
    ti = PESDT.data.ti #np.zeros(num_cells)
    
    ne = PESDT.data.ne  #np.zeros(num_cells)

    n_azs = PESDT.data.n_azs
    n_izs = PESDT.data.n_izs

    species_list = []
    emission_dict = {}
    
    for species, wl_transitions in species_transitions.items():
        adf = ADF15(species)
        sp, charge, idx = get_base_species_and_charge(species, plasma_species)
        exc_sp = (ELEMENT_DICT[sp], charge) # Excitation
        rec_sp = (ELEMENT_DICT[sp], charge+1) # Recombination

        if not exc_sp in emission_dict:
            emission_dict[exc_sp] = {}
        if not rec_sp in emission_dict:
            emission_dict[rec_sp] = {}

        # count the number of charge states of the elements before this to get the correct index
        bc_idx = 0
        ba_idx = idx
        for k in range(idx):
            bc_idx += get_num_charge_states(plasma_species[k])

        n_exc = None
        n_rec = None
        if charge == 0:
            n_exc = n_azs[ba_idx, :]
            n_rec = n_izs[bc_idx, :]
        else:
            n_exc = n_izs[bc_idx + charge -1, :]
            n_rec = n_izs[bc_idx + charge, :]

        for wl, transition in wl_transitions.items():
            tr = (int(transition[0]), int(transition[1]))
            emission[exc_sp][tr] = adf.interpolate(te, ne, "EXCIT", wl)*ne*n_exc*1/(4.0*np.pi)
            emission[rec_sp][tr] = adf.interpolate(te, ne, "RECOM", wl)*ne*n_rec*1/(4.0*np.pi)

    emission_keys = list(emission_dict.keys())
    emission = [values for _, values in emission_dict.items()]
    num_species = len(species_list)

    sim = PESDTSimulation(mesh, species_list) #[['D0', 0], ['D+1', 1]])
    sim.electron_temperature = te
    sim.electron_density = ne
    sim.ion_temperature = ti
    sim.species_density  = np.zeros((num_species, num_cells))
    sim.emission = [emission_keys, emission] # Emission is precalculated for all data sources
    return sim


def createZCherabPlasmaBolo(PESDT):
    '''
    Creates a cherab compatible PLASMA simulation object for any species using OpenADAS rates
    
    '''
    num_cells = len(PESDT.data.te)
    mesh = create_cherab_mesh(PESDT)
    plasma_species: list[str] = PESDT.species 
    te = PESDT.data.te #np.zeros(num_cells)
    ti = PESDT.data.ti #np.zeros(num_cells)
    
    ne = PESDT.data.ne  #np.zeros(num_cells)

    n_azs = PESDT.data.n_azs
    n_izs = PESDT.data.n_izs

    species_list = []
    emission = []
    emission_keys = ["plt", "prb", "fffb"]
    idx = 0
    for i, species in enumerate(plasma_species):
        adf = ADF11(species)

        max_charge = get_num_charge_states(species)
        for z in range(max_charge):
            sp = (ELEMENT_DICT[species], z)
            species_list.append(sp)
            em_dict = {}
            if z == 0:
                n_exc = n_azs[i, :]
                n_rec = n_izs[idx +z, :]
            else:
                n_exc = n_izs[idx +z-1, :]
                n_rec = n_izs[idx +z, :]
            
            wl = 10**np.arange(0, 4.01, 0.1)
            ff, fffb = continuov_(wl, te, max_charge, z+1)
            em_dict["plt"] = adf.interp_plt(te, ne, z)*ne*n_exc*1/(4.0*np.pi)
            em_dict["prb"] = adf.interp_prb(te, ne, z)*ne*n_rec*1/(4.0*np.pi)
            em_dict["fffb"] = np.trapezoid(fffb* h*c/(1e-10*wl[None, :]), wl, axis = 1)*ne*n_rec*1/(4.0*np.pi)
            emission.append(em_dict)
            idx +=1


    num_species = len(species_list)

    sim = PESDTSimulation(mesh, species_list) #[['D0', 0], ['D+1', 1]])
    sim.electron_temperature = te
    sim.electron_density = ne
    sim.ion_temperature = ti
    sim.species_density  = np.zeros((num_species, num_cells))
    sim.emission = [emission_keys, emission] # Emission is precalculated for all data sources
    return sim

def createHydrogenicCherabPlasma(PESDT, transitions: list,
                       data_source = "AMJUEL", 
                       recalc_h2_pos = True,
                       mol_exc_bands = None,
                       opaque = False,
                       opaque_mode = 0,
                       opaque_bins = 21):
    '''
    Creates a cherab compatible PLASMA simulation object for hydrogenic plasmas with user defined data source,
    e.g. AMJUEL to have molecular contributions
    
    convert_denel_to_m3: When using adas data, you need to explicitly convert to the right units
    recalc_h2_pos: Recalculate the H2+ denisity according to AMJUEL H.12 2.0c. Should only be used, if
                   the molecular ion density is not available in the simualtion output
    
    '''

    ########################################################################
    # Start by loading in all the data from the PESDT object #

    num_cells = len(PESDT.data.te)
    num_neut = 2 if data_source in ["AMJUEL", "YACORA"] else 1

    
    te = PESDT.data.te #np.zeros(num_cells)
    ti = PESDT.data.ti #np.zeros(num_cells)
    t0 = PESDT.data.t0 #np.zeros(num_cells)
    ne = PESDT.data.ne  #np.zeros(num_cells)
    ni = PESDT.data.ni #np.zeros(num_cells)
    n0 = PESDT.data.n0 #np.zeros(num_cells)
    n2 = PESDT.data.n2 #np.zeros(num_cells)
    n2p = PESDT.data.n2p #np.zeros(num_cells)
    mesh = create_cherab_mesh(PESDT)

    #####################################################
    # Now load the simulation object with plasma values #

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
            
            em_n_exc, em_n_rec, em_mol, em_h2_pos, em_h3_pos, em_h_neg, tot = calc_photon_rate(transitions[i], te, ne, n0[:], mol_n_density = n2[:], mol_p_density = h2_pos_den[:],recalc_h2_pos = recalc_h2_pos, debug = True)
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
            num_neut +=1
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
        adf = ADF15()
        tra_wl_dct = {trans: wl for wl, trans in sdb.data["H"].items()}
        #print(tra_wl_dct)
        emission = [{} for _ in range(len(species_density))]
        for i in range(len(transitions)):
            wl = tra_wl_dct[transitions[i]]
            emission[0][transitions[i]] = adf.interpolate(te, ne, "EXCIT", wl)*ne*n0*1/(4.0*np.pi)
            emission[1][transitions[i]] = adf.interpolate(te, ne, "RECOM", wl)*ne*ne*1/(4.0*np.pi)
    
    species_density[0, :] = n0[:]  # neutral density D0
    species_density[1, :] = ni[:]  # ion density D+1
    
    neutral_temperature = np.zeros((num_neut, num_cells))
    for i in range(num_neut):
        neutral_temperature[i, :] = t0[:] # just use T_n0 for now
    #neutral_temperature[-1, :] = t0[:]

    # Test with zero absorbance
    
    print(species_list)

    sim = PESDTSimulation(mesh, species_list, opaque = opaque) #[['D0', 0], ['D+1', 1]])
    sim.electron_temperature = te
    sim.electron_density = ne
    sim.ion_temperature = ti
    sim.neutral_temperature = neutral_temperature
    sim.species_density = species_density
    sim.emission = [emission_keys, emission] # Emission is precalculated for all data sources
    return sim

def createHydrogenicCherabPlasmaBolo(PESDT, data_source = "AMJUEL", **kwargs):
    num_cells = len(PESDT.data.te)

    te = PESDT.data.te #np.zeros(num_cells)
    ti = PESDT.data.ti #np.zeros(num_cells)
    t0 = PESDT.data.t0 #np.zeros(num_cells)
    ne = PESDT.data.ne  #np.zeros(num_cells)
    ni = PESDT.data.ni #np.zeros(num_cells)
    n0 = PESDT.data.n0 #np.zeros(num_cells)
    n2 = PESDT.data.n2 #np.zeros(num_cells)
    n2p = PESDT.data.n2p #np.zeros(num_cells)

    mesh = create_cherab_mesh(PESDT)

    emission_keys = [(2, 1)] # Use Lyman alpha as the wl
    if data_source == "ADAS":
        species_list = [(D0, 0), (D0, 1), (D0, 2), (D0, 3)]
        num_species = 4
        species_density = np.zeros((num_species, num_cells))
        adf = ADF11()
        

        emission = [{} for _ in range(num_species)]
        
        emission[0][(2,1)] = adf.interpolate_plt(te, ne, 0)*ne*n0*1/(4.0*np.pi)
        emission[1][(2,1)] = adf.interpolate_prb(te, ne, 0)*ne*ne*1/(4.0*np.pi)
        ff, fffb = continuov_(10**np.arange(0, 4.01, 0.1), te, 1, 1)
        wl = 10**np.arange(0, 4.01, 0.1)
        emission[2][(2, 1)] = np.trapezoid(ff * h*c/(1e-10*wl[None, :]), wl, axis = 1)*ne*ne*1/(4.0*np.pi)
        emission[3][(2, 1)] = np.trapezoid(fffb* h*c/(1e-10*wl[None, :]), wl, axis = 1)*ne*ne*1/(4.0*np.pi)
    elif data_source == "AMJUEL":
        species_list = [(D0, 0), (D0, 2), (D0, 3)]
        num_species = 3
        species_density = np.zeros((num_species, num_cells))
        emission = [{} for _ in range(num_species)]
        em_line = np.zeros_like(te)
        
        for i in range(1,5):
            for j in range(2, 7):
                if i >= j: continue
                transition = (j, i)
                wl = wavelength(transition)
                em_line += calc_photon_rate(transition, te, ne, n0, mol_n_density = n2, mol_p_density = n2p, h_neg = kwargs.get("h_neg", False), recalc_h2_pos = kwargs.get("recalc_h2_pos")) * h*c/(1e-9*wl)
        emission[0][(2, 1)] = em_line
        ff, fffb = continuov_(10**np.arange(0, 4.01, 0.1), te, 1, 1)
        wl = 10**np.arange(0, 4.01, 0.1)
        emission[1][(2, 1)] = np.trapezoid(ff * h*c/(1e-10*wl[None, :]), wl, axis = 1)*ne*ne*1/(4.0*np.pi)
        emission[2][(2, 1)] = np.trapezoid(fffb* h*c/(1e-10*wl[None, :]), wl, axis = 1)*ne*ne*1/(4.0*np.pi)
    else:
        # Assume Cell has total radiated power
        species_list = [(D0, 0)]
        num_species = 1
        species_density = np.zeros((num_species, num_cells))
        emission = [{} for _ in range(num_species)]
        rad = np.zeros((num_cells))
        for ith_cell, cell in enumerate(PESDT.cells):
            rad[ith_cell] = cell.tot_rad
        emission[0][(2,1)] = rad

    sim = PESDTSimulation(mesh, species_list) 
    sim.electron_temperature = te
    sim.electron_density = ne
    sim.ion_temperature = ti
    sim.species_density = species_density
    sim.emission = [emission_keys, emission]
    return sim