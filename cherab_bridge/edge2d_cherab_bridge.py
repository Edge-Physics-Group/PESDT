
import numpy as np
from raysect.core.math.function.float.function2d.interpolate import Discrete2DMesh
from cherab.PESDT_addon import Edge2DMesh, Edge2DSimulation
from PESDT.amread import read_amjuel_1d,read_amjuel_2d,reactions, calc_cross_sections, calc_photon_rate

def load_edge2d_from_PESDT(PESDT, convert_denel_to_m3 = True, load_mol_data = False, recalc_h2_pos = True, quick = True, **kwargs ):
    '''
    Creates a cherab compatible EDGE2D simulation object
    
    convert_denel_to_m3: When using adas data, you need to explicitly convert to the right units
    load_mol_data: Load and pass molecular data based on AMJUEL rates to the object
        recalc_h2_pos: Recalculate the H2+ denisity according to AMJUEL H.12 2.0c. Should only be used, if
                       the molecular ion density is not available in the simualtion output
    quick (BETA): Pre calculate emission, and pass directly to cherab
    **kwargs: used in passing data required for 'quick'
    
    '''

    ########################################################################
    # Start by loading in all the data from B Lowmanowski's PESDT object #

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

    # Master list of species, e.g. ['D0', 'D+1', 'C0', 'C+1', ...
    
    #sim._species_list = ['D0', 'D+1']
    species_list = [('D', 0), ('D', 1)]
    
    if load_mol_data:
        '''
        Calculate the H2+, H3+, and H- densities through AMJUEL rates, and add the molecular density 
        and derived densities to species

        '''


        print("Loading H2, H2+, H3+ and H-")
        num_species = 6
        species_density = np.zeros((num_species, num_cells))
        species_list.append(('D2', 0))
        
        reac = reactions("2") # The densities are independent of the hydrogenic excited state
        if recalc_h2_pos:
            MARc_h2_pos_den = read_amjuel_2d(reac["den_H2+"][0],reac["den_H2+"][1])
            h2_pos_den = calc_cross_sections(MARc_h2_pos_den, T = te, n = ne*1e-6)*n2
        else:
            h2_pos_den = n2p[:]
        species_list.append(('D2', 1))
        '''
        MARc_h3_pos_den = read_amjuel_1d(reac["den_H3+"][0],reac["den_H3+"][1])
        h3_pos_den = calc_cross_sections(MARc_h3_pos_den, T = te, n = ne*1e-6)*n2*h2_pos_den/ne
        species_list.append(('D3', 1))
        '''
        MARc_h_neg_den = read_amjuel_1d(reac["den_H-"][0],reac["den_H-"][1])
        h_neg_den = calc_cross_sections(MARc_h_neg_den, T = te, n = ne*1e-6)*n2
        species_list.append(('H', 0)) # Need a proxy, neg allowed in Cherab

        species_density[2, :] = n2[:]  # Mol. density D2
        species_density[3, :] = h2_pos_den[:]
        #species_density[4, :] = h3_pos_den[:]
        species_density[4, :] = h_neg_den[:]
    else:
        num_species = 2
        species_density = np.zeros((num_species, num_cells))

    species_density[0, :] = n0[:]  # neutral density D0
    species_density[1, :] = ni[:]  # ion density D+1
    edge2d_mesh = Edge2DMesh(rv, zv) #, rc, zc)

    print(species_list)

    sim = Edge2DSimulation(edge2d_mesh, species_list ) #[['D0', 0], ['D+1', 1]])
    sim.electron_temperature = te
    sim.electron_density = ne
    sim.ion_temperature = ti

    sim.species_density = species_density
    #sim._species_density = species_density
    #sim._ion_temperature = ti

    if quick:
        if load_mol_data:
            transitions = kwargs.get('transitions', None)
            excit_emiss = {}
            rec_emiss = {}
            mol_emiss = {}
            mol_ion_emiss = {}
            at_neg_emiss = {}
            for t in transitions:
                t_str = f'{t[0]},{t[1]}'
                E_excit, E_recom, E_mol, E_h2_pos, E_h3_pos, E_h_neg, E_tot = calc_photon_rate(t, te, ne, n0, mol_n_density=n2,molp_n_density=n2p,p_density=ni, h3=False, recalc_h2_pos=recalc_h2_pos, debug=True)
                excit_emiss[t_str] = E_excit
                rec_emiss[t_str] = E_recom
                mol_emiss[t_str] = E_mol
                mol_ion_emiss[t_str] = E_h2_pos
                at_neg_emiss[t_str] = E_h_neg
            sim.emission = [excit_emiss, rec_emiss, mol_emiss, mol_ion_emiss, at_neg_emiss]
    return sim
