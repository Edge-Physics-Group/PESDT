import numpy as np
import os
"""
AMREAD
Vesa-Pekka Rikala, 2024, Aalto University

The functions to read AMJUEL rate coefficients were adapted from an example script by Ray Chandra (Aalto)

The purpose of this script file is to pride functions for reading AMJUEL (EIRENE) rate coefficients, and to calculate the associated cross-sections (or ratios).
There are also functions to directly calculate the hydrogeninc photon emission rates.
"""

"""
Define constants
"""

Ry = 13.58 #eV, Rydberg energy (ionization potential) for hydrogen
h = 4.1357e-15 #(eV s), Planck's constant in eV
c = 299792458 # m/s, speed of light
k = 8.617e-5 # eV/K, Boltzmann constant in eV
amjuel_path = os.path.expanduser('~') +'/AMJUEL.tex'

"""
In this file the hydrogenic transition is typed as "n"->"m", i.e.
transition = ("n", "m") (n>m)

"""


def reactions(excitation_level):
    '''
    Dictionary of relevant AMJUEL(2020) reactions for each excitation level of hydrogen
    Currently in the AMJUEL (2020) rates go only to n=6, thus, many of the high balmer, paschen and mid n=4 m>n transitions cannot be evaluated using AMJUEL rates

    The rates for H3+ density were added to AMJUEL in 2017, i.e. if the available version is of earlier date, disable the H3+ contribution
    
    '''
    react_dict = {
        2 : {
                "atomic_exc": ("H.12", "2.1.5b"),
                "atomic_rec": ("H.12", "2.1.8b"),
                "H-": ("H.12", "7.2b"),
                "H2": ("H.12", "2.2.5b"),
                "H2+": ("H.12", "2.2.14b"),
                "H3+": ("H.12", "2.2.15b"),
                "den_H2+": ("H.12", "2.0c"),
                "den_H3+": ("H.11", "4.0a"),
                "den_H-": ("H.11", "7.0a"),
            },
        3 : {
                "atomic_exc": ("H.12", "2.1.5a"),
                "atomic_rec": ("H.12", "2.1.8a"),
                "H-": ("H.12", "7.2a"),
                "H2": ("H.12", "2.2.5a"),
                "H2+": ("H.12", "2.2.14a"),
                "H3+": ("H.12", "2.2.15a"),
                "den_H2+": ("H.12", "2.0c"),
                "den_H3+": ("H.11", "4.0a"),
                "den_H-": ("H.11", "7.0a"),
            },
        4 : {
                "atomic_exc": ("H.12", "2.1.5c"),
                "atomic_rec": ("H.12", "2.1.8c"),
                "H-": ("H.12", "7.2c"),
                "H2": ("H.12", "2.2.5c"),
                "H2+": ("H.12", "2.2.14c"),
                "H3+": ("H.12", "2.2.15c"),
                "den_H2+": ("H.12", "2.0c"),
                "den_H3+": ("H.11", "4.0a"),
                "den_H-": ("H.11", "7.0a"),
            },
        5 : {
                "atomic_exc": ("H.12", "2.1.5d"),
                "atomic_rec": ("H.12", "2.1.8d"),
                "H-": ("H.12", "7.2d"),
                "H2": ("H.12", "2.2.5d"),
                "H2+": ("H.12", "2.2.14d"),
                "H3+": ("H.12", "2.2.15d"),
                "den_H2+": ("H.12", "2.0c"),
                "den_H3+": ("H.11", "4.0a"),
                "den_H-": ("H.11", "7.0a"),
            },
        6 : {
                "atomic_exc": ("H.12", "2.1.5e"),
                "atomic_rec": ("H.12", "2.1.8e"),
                "H-": ("H.12", "7.2e"),
                "H2": ("H.12", "2.2.5e"),
                "H2+": ("H.12", "2.2.14e"),
                "H3+": ("H.12", "2.2.15e"),
                "den_H2+": ("H.12", "2.0c"),
                "den_H3+": ("H.11", "4.0a"),
                "den_H-": ("H.11", "7.0a"),
            },
    }
    return react_dict[excitation_level]

def wavelength(transition):
    '''
    Returns the wavelength for hydrogen line transition using the Rydberg formula in nm
    '''
    return (1.096677e7*(1/int(transition[1]**2) -1/int(transition[0])**2))**(-1)*1e9

def FF(l):
    v = c/l
    return (2*h*v**3)/c**2 

def pp(l, T):
    v=c/l
    return FF(l)* 1/(np.exp(h*v/(k*T))-1)

def A_coeff(transition):

    '''
    Returns the Einstein coefficient for transition "n"<-"m"

    transition ("str", "str")

    Coefficients from "Elementary Processes in Hydrogen-Helium Plasmas" (1987), by Janev, Appendix A.2.The original source of the table is Wiese et al. (1966)
    '''

    coeff_dict = {
        1: {2: 4.699e8, 3: 5.575e7, 4: 1.278e7, 5: 4.125e6, 6: 1.644e6}, 
        2: {3: 4.410e7, 4: 8.419e6, 5: 2.53e6, 6: 9.732e5, 7: 4.389e5},
        3: {4: 8.989e6, 5: 2.201e6, 6: 7.783e5, 7: 3.358e5, 8: 1.651e5},
        4: {5: 2.699e6, 6: 7.711e5, 7: 3.041e5, 8:1.424e5, 9: 7.459e4}
    }

    return coeff_dict[transition[1]][transition[0]]

def Bmn_coeff(transition):
    '''
    Calculates the stimulated emission coefficient
    '''
    return A_coeff(transition)*FF(wavelength(transition))

def calc_cross_sections(MARc, T = None, n = None, E = None):
    '''
    Returns the cross-sections for a given coefficient matrix MARc in T(,n or E)
    if given either n or E (2D MARc), the shape of T and n or E must be the same

    MARc: AMJUEL coefficient matrix, either 9 by 1, or 9 by 9
    T: Temperature vector (eV)
    n: electron density vector (cm^-3)
    E: energy of the particle (J)
    '''
    # Calc 2d fit
    cross_sections = np.empty(np.shape(MARc), dtype=np.ndarray)
    if np.shape(MARc) == (9,9):
        if (n is not None):
            nE = n/1e8
        else:
            nE = E
        for n in range(0,9):
            for m in range(0,9):
                cross_sections[n,m] = MARc[n,m]*np.power(np.log(T), n)*np.power(np.log(nE), m)
    else:
    # Calc 1d fit
        for n in range(0,9):
            cross_sections[n] =  MARc[n]*np.power(np.log(T), n)

    cross_sections = np.sum(cross_sections)
    return np.exp(cross_sections)        

def read_amjuel_1d(h_name, collisionName, **kwargs):
    '''
    Reads in 1d fits (either in E or in T)

    param file (str): path to the AMJUEL.tex

    
    '''
    coeff_letter_dict = {'H.0': 'p', 'H.1': 'a', 'H.2': 'b', 'H.5': 'e', 'H.8': 'h', 'H.11': 'k'}
    coeff_letter = coeff_letter_dict[h_name]
    file = kwargs.get( 'file', amjuel_path)
    collect = False
    MARc = np.zeros(9)
    f = open(file,'r')
    rate = False
    
    for line in f:
        if '\section{' +f'{h_name}' in line:
            rate = True
        if collect and rate:
            #print(line)
            if (f'{coeff_letter}0' in line):
                line = line.replace('D','E')
                columns = line.split()
                MARc[0] = float(columns[1])
                MARc[1] = float(columns[3])
                MARc[2] = float(columns[5])
            if (f'{coeff_letter}3' in line):
                line = line.replace('D','E')
                columns = line.split()
                MARc[3] = float(columns[1])
                MARc[4] = float(columns[3])
                MARc[5] = float(columns[5])
            if (f'{coeff_letter}6' in line):
                line = line.replace('D','E')
                columns = line.split()
                MARc[6] = float(columns[1])
                MARc[7] = float(columns[3])
                MARc[8] = float(columns[5])
                collect = False
        if line == '\end{document}':
            break
        elif ('Reaction '+ collisionName) in line:
            collect = True
    return MARc

def read_amjuel_2d(h_name, collisionName, **kwargs):
    '''
    function to read 2d (E,T) or (n,T)
    '''

    file = kwargs.get( 'file', amjuel_path)
    collect = False
    MARc = np.zeros((9,9))
    f = open(file,'r')
    rate = False
    start_read = False
    index0 = [0, 0, 0]
    collected = 0
    for line in f:
        if '\section{' + f'{h_name}'  in line:
            rate = True
        if rate:
            if line == '\end{document}':
                break
            elif ('Reaction '+ collisionName) in line:
                collect = True
        if collect and rate:
            #print(line)
            if 'E-Index' in line:
                columns = line.split()
                index0[0] = int(columns[1])
                index0[1] = int(columns[2])
                index0[2] = int(columns[3])
                continue
            if 'Max. rel. Error' in line:
                collect = False
            if 'T-Index' in line:
                # Empty line
                start_read = True
                continue
            elif  start_read:
                line = line.replace('D','E')
                columns = line.split()
                MARc[int(columns[0]), index0[0]] = float(columns[1] )
                MARc[int(columns[0]), index0[1]] = float(columns[2] )
                MARc[int(columns[0]), index0[2]] = float(columns[3] )
                if int(columns[0]) == 8:
                    start_read = False
                    collected += 1
                    if collected == 3:
                        break                    
        
    return MARc

def calc_photon_rate(transition, Temperature, el_density, n_density, mol_n_density = None, p_density = None, h3 = True, **kwargs):

    '''
    Calculates the photon rate per unit volume (ph/m^3) for the transition "n"<-"m" using tabulated Einstein emission coefficients, and AMJUEL rates for the H("n") population. 
    Also accounts for the molecular (H2, H2+, H3+) contribution to H("n"), and H- contribution to H("n"), if the molecular density is given. 
    Otherwise only the direct electron impact excitation and recombinations processes is used.

    

    TODO:
    Also count the stimulated emission
    '''
    # Make sure everyhting is in np.array format
    Temperature = np.array(Temperature); el_density = np.array(el_density); n_density = np.array(n_density)

    # Check if transition is typed in the expected format. If not reverse
    if (int(transition[1])>int(transition[0])):
        transition = (transition[0], transition[1])
    
    el_density = el_density*1e-6 # Convert to cm^-3
    n_density = n_density*1e-6 # Convert to cm^-3

    # Check if ion density is given. If not, use n_e = n_i
    if p_density is None:
        p_density = el_density
    else:
        p_density = np.array(p_density)*1e-6
    
    # Initialize the result matrices. If molecular effects are not accounted for,
    # the em_mol etc. will stay as zero.
    em_n_exc = np.zeros(np.shape(Temperature))
    em_n_rec = np.zeros(np.shape(Temperature))
    em_mol = np.zeros(np.shape(Temperature))
    em_h2_pos = np.zeros(np.shape(Temperature))
    em_h3_pos = np.zeros(np.shape(Temperature))
    em_h_neg = np.zeros(np.shape(Temperature))

    reac = reactions(transition[0])

    # Calculate the atomic contribution
    MARc_h_exc = read_amjuel_2d(reac["atomic_exc"][0], reac["atomic_exc"][1])
    MARc_h_rec = read_amjuel_2d(reac["atomic_rec"][0], reac["atomic_rec"][1])

    #print(A_coeff(transition))

    em_n_exc = A_coeff(transition)*calc_cross_sections(MARc_h_exc, T = Temperature, n = el_density)*n_density/(4*np.pi)
    em_n_rec = A_coeff(transition)*calc_cross_sections(MARc_h_rec, T = Temperature, n = el_density)*p_density/(4*np.pi)

    #print(mol_n_density)
    if mol_n_density is not None:
        #print("Calculating molecular contribution")
        mol_n_density = np.array(mol_n_density)
        # Include molecular contributions to H('n'), i.e. h2, h2+, h3+, h-
        mol_n_density = mol_n_density*1e-6 # Convert to cm^-3

        MARc_h2 = read_amjuel_2d(reac["H2"][0],reac["H2"][1])
        em_mol = A_coeff(transition)*calc_cross_sections(MARc_h2, T = Temperature, n = el_density)*mol_n_density/(4*np.pi)

        # H2+
        MARc_h2_pos_den = read_amjuel_2d(reac["den_H2+"][0],reac["den_H2+"][1])
        h2_pos_den = calc_cross_sections(MARc_h2_pos_den, T = Temperature, n = el_density)*mol_n_density

        MARc_h2_pos = read_amjuel_2d(reac["H2+"][0],reac["H2+"][1])
        em_h2_pos = A_coeff(transition)*calc_cross_sections(MARc_h2_pos, T = Temperature, n = el_density)*h2_pos_den/(4*np.pi)

        # H3+
        # H3+ rates may not be generally available, as they're supplied only in (relatively) recent editions of AMJUEL(>=2017)
        if h3:
            MARc_h3_pos_den = read_amjuel_1d(reac["den_H3+"][0],reac["den_H3+"][1])
            h3_pos_den = calc_cross_sections(MARc_h3_pos_den, T = Temperature)*mol_n_density*h2_pos_den/el_density

            MARc_h3_pos = read_amjuel_2d(reac["H3+"][0],reac["H3+"][1])
            em_h3_pos = A_coeff(transition)*calc_cross_sections(MARc_h3_pos, T = Temperature, n = el_density)*h3_pos_den/(4*np.pi)

        # H-
        MARc_h_neg_den = read_amjuel_1d(reac["den_H-"][0],reac["den_H-"][1])
        h_neg_den = calc_cross_sections(MARc_h_neg_den, T = Temperature)*mol_n_density

        MARc_h_neg = read_amjuel_2d(reac["H-"][0],reac["H-"][1])
        em_h_neg = A_coeff(transition)*calc_cross_sections(MARc_h_neg, T = Temperature, n = el_density)*h_neg_den/(4*np.pi)

    # Debug: output each contribution separately
    if (kwargs.get("debug", False)):
        return em_n_exc*1e6, em_n_rec*1e6, em_mol*1e6, em_h2_pos*1e6, em_h3_pos*1e6, em_h_neg*1e6, (em_n_exc + em_n_rec + em_mol + em_h2_pos + em_h3_pos + em_h_neg)*1e6
    
    return (em_n_exc + em_n_rec + em_mol + em_h2_pos + em_h3_pos + em_h_neg)*1e6

def photon_rate_coeffs(n):
    '''
    Returns the coefficient matrices for a "n" excited hydrogen

    param: n (int)
    '''

    reac = reactions(n)

    MARc_h_exc = read_amjuel_2d(reac["atomic_exc"][0], reac["atomic_exc"][1])
    MARc_h_rec = read_amjuel_2d(reac["atomic_rec"][0], reac["atomic_rec"][1])
    MARc_h2 = read_amjuel_2d(reac["H2"][0],reac["H2"][1])
    MARc_h2_pos = read_amjuel_2d(reac["H2+"][0],reac["H2+"][1])
    MARc_h3_pos = read_amjuel_2d(reac["H3+"][0],reac["H3+"][1])
    MARc_h_neg = read_amjuel_2d(reac["H3+"][0],reac["H3+"][1])

    return MARc_h_exc, MARc_h_rec, MARc_h2, MARc_h2_pos, MARc_h3_pos, MARc_h_neg
