import numpy as np
import os
from scipy.constants import Rydberg, h, c, k, m_e, e, epsilon_0
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


h = h/e #(eV s), Planck's constant in eV
k = k/e # eV/K, Boltzmann constant in eV
amjuel_dir =  os.environ.get( "AMJUEL_PATH",os.path.expanduser('~') )
amjuel_path  = os.path.join(amjuel_dir, 'AMJUEL.tex')

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

def H2_reactions(band: str = "fulcher"):
    """
    Dictionary of H2 excitation emission bands. Returns the rate names, and the associated average Einstein Coefficient.
    """
    dct: dict =  {
        "fulcher": ("H.12", "2.2.5fu", 2.43e7),
        "werner":  ("H.12", "2.2.5we", 1.04e9),
        "lyman":   ("H.12", "2.2.5ly", 0.84e9)
    }
    return dct[band.lower()]

def wavelength(transition):
    '''
    Returns the wavelength for hydrogen line transition using the Rydberg formula in nm
    '''
    return (1.096677e7*(1/int(transition[1]**2) -1/int(transition[0])**2))**(-1)*1e9

def H2_wavelength(band: str = "fulcher"):
    '''
    Returns the average wavelength of a H2 excitation band from a tabulated dictionary
    '''
    dct: dict = {
        "fulcher": 620.0,
        "werner":  130.0,
        "lyman":   150.0
        }
    return dct[band]

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

def oscillator_strength(transition):

    wl = wavelength(transition)
    gu_gl = (2.0*transition[0]+1)/(2.0*transition[1]+1)
    A = A_coeff(transition)

    return 8.0*np.pi*e**2*(wl*1e-9)**2/(m_e*c*epsilon_0)*gu_gl*A

def doppler_absorbance_wave(transition, wave, T_D, n_D, m_D):
    '''
    Absorbance per wavelength of a Doppler broadened line.
    
    :param tuple[int, int] transition:
    :param np.ndarray wave: [m], dim m
    :param np.ndarry T_d: [eV], dim n
    :param np.ndarry n_d: [m^-3], dim n
    :param float m_D: [kg]

    return np.ndarray dim (n, m)
    '''
    lambda_0 = wavelength(transition)*1e-9
    delta_lambda_D = lambda_0 * np.sqrt(2*e*T_D/(m_D*c**2))

    return np.sqrt(np.pi)*e**2/(m_e*c*delta_lambda_D[:, None]) *oscillator_strength(transition)*n_D[:, None]*np.exp(-((wave-lambda_0)/delta_lambda_D)**2)

def doppler_absorbance(transition, T_D, n_D, m_D):
    '''
    Absorbance per wavelength of a Doppler broadened line.
    
    :param tuple[int, int] transition:
    :param np.ndarry T_d: [eV], dim n
    :param np.ndarry n_d: [m^-3], dim n
    :param float m_D: [kg]

    return np.ndarray dim (n)

    Multiply by 1/DeltaLambda_D * exp(-((lambda0-wave)((delta_lambda_D))**2) to get absorbance per wl
    '''

    return np.sqrt(np.pi)*e**2/(m_e*c) *oscillator_strength(transition)*n_D


def ideal_absorbance(transition, T_D, n_D, m_D):
    '''
    Ideal absorbance of the emission line
    '''

    return np.pi*e**2/(m_e*c**2)*wavelength(transition)**2*oscillator_strength(transition)*n_D

def cen_absorbance(transition, T_D, n_D, m_D):
    '''
    Center absorbance of a Doppler broadened line
    
    '''
    return np.pi*e**2/(m_e*wavelength(transition)/c)*np.sqrt(m_D/(2*e*T_D))*oscillator_strength(transition)*n_D

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

def calc_H2_band_emission(Temperature, el_density, mol_n_density, band = "fulcher"):
    """
    calculates the H2 Fulcher/Werner/Lyman band emission, and the density of the excited population
    DENSITIES IN M^-3
    EMISSION IN PH/S/SR/M^3
    """
    el_density = el_density*1e-6
    mol_n_density = mol_n_density*1e-6
    h_name, collisionName, acoeff = H2_reactions(band=band)
    MARc = read_amjuel_2d(h_name, collisionName)
    cs = calc_cross_sections(MARc, Temperature, n = el_density)
    den = cs * mol_n_density
    em = acoeff*den/(4*np.pi)*1e6
    return em, cs

def calc_photon_rate(transition, Temperature, el_density, n_density, **kwargs):

    '''
    Calculates the photon rate per unit volume (ph/m^3) for the transition "n"<-"m" using tabulated Einstein emission coefficients, and AMJUEL rates for the H("n") population. 
    Also accounts for the molecular (H2, H2+, H3+) contribution to H("n"), and H- contribution to H("n"), if the molecular density is given. 
    Otherwise only the direct electron impact excitation and recombinations processes is used.

    

    TODO:
    Also count the stimulated emission
    '''
    
    mol_n_density = kwargs.get("mol_n_density", None)
    mol_p_density = kwargs.get("mol_p_density", None)
    p_density = kwargs.get("p_density", el_density)*1e-6
    h2_pos = kwargs.get("h2_pos", True)
    recalc_h2_pos = kwargs.get("recalc_h2_pos")
    h3 = kwargs.get("h3", True)
    h_neg = kwargs.get("h_neg", True)
    separate_contributions = kwargs.get("separate_contributions", kwargs.get("debug", False)) # Support old "debug" flag for separate contributions
    # Make sure everyhting is in np.array format
    Temperature = np.array(Temperature); el_density = np.array(el_density); n_density = np.array(n_density)

    # Check if transition is typed in the expected format. If not reverse
    if (int(transition[1])>int(transition[0])):
        transition = (transition[0], transition[1])
    
    el_density = el_density*1e-6 # Convert to cm^-3
    n_density = n_density*1e-6 # Convert to cm^-3

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
        
        em_h2_pos = np.zeros_like(mol_n_density)
        # H2+
        if h2_pos:
            if recalc_h2_pos or mol_p_density is None:
                MARc_h2_pos_den = read_amjuel_2d(reac["den_H2+"][0],reac["den_H2+"][1])
                h2_pos_den = calc_cross_sections(MARc_h2_pos_den, T = Temperature, n = el_density)*mol_n_density
            else:
                h2_pos_den = mol_p_density
            MARc_h2_pos = read_amjuel_2d(reac["H2+"][0],reac["H2+"][1])
            em_h2_pos = A_coeff(transition)*calc_cross_sections(MARc_h2_pos, T = Temperature, n = el_density)*h2_pos_den/(4*np.pi)

        # H3+
        # H3+ rates may not be generally available, as they're supplied only in (relatively) recent editions of AMJUEL(>=2017)
        em_h3_pos = np.zeros_like(mol_n_density)
        if h3:
            MARc_h3_pos_den = read_amjuel_1d(reac["den_H3+"][0],reac["den_H3+"][1])
            h3_pos_den = calc_cross_sections(MARc_h3_pos_den, T = Temperature)*mol_n_density*h2_pos_den/el_density

            MARc_h3_pos = read_amjuel_2d(reac["H3+"][0],reac["H3+"][1])
            em_h3_pos = A_coeff(transition)*calc_cross_sections(MARc_h3_pos, T = Temperature, n = el_density)*h3_pos_den/(4*np.pi)

        # H-
        em_h_neg = np.zeros_like(mol_n_density)
        if h_neg:
            MARc_h_neg_den = read_amjuel_1d(reac["den_H-"][0],reac["den_H-"][1])
            h_neg_den = calc_cross_sections(MARc_h_neg_den, T = Temperature)*mol_n_density

            MARc_h_neg = read_amjuel_2d(reac["H-"][0],reac["H-"][1])
            em_h_neg = A_coeff(transition)*calc_cross_sections(MARc_h_neg, T = Temperature, n = el_density)*h_neg_den/(4*np.pi)

    # Debug: output each contribution separately
    if separate_contributions:
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
