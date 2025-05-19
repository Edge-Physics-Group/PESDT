import numpy as np
from collections import OrderedDict
from PESDT.plot_old import Plot
import struct

from PESDT.pyADASutils.source_sink_utils import calc_Siz_synth, calc_Srec_synth, calc_Siz_exp, calc_Srec_exp

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def floatToBits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>l', s)[0]

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


def get_var_by_ring(workdir, case, firstSOLring=12, nSOLrings=5):
    
    out_dict = {}
#    for casekey, case in cases.items():
        
    o = Plot(workdir, case['case'])
    print('get_H_rad_power_by_ring ', case['case'])
    
    H_rad = []
    H_rad_perm3 = []
    for iring in range(nSOLrings):
        ring = iring+firstSOLring
        _row = []
        _H_rad = []
        _H_rad_perm3 = []
        for cell in o.data2d.cells:
            if cell.ring == ring:
                _row.append(cell.row)
                _H_rad.append(cell.H_radpwr)
                _H_rad_perm3.append(cell.H_radpwr_perm3)

        sorted_idx = np.argsort(_row)
        H_rad.append(np.asarray(_H_rad)[sorted_idx])
        H_rad_perm3.append(np.asarray(_H_rad_perm3)[sorted_idx])
        
    out_dict['H_rad'] = np.asarray(H_rad)
    out_dict['H_rad_perm3'] = np.asarray(H_rad_perm3)

    return out_dict



def get_region_Sion_Srec(workdir, cases, regions=['lfs_div']):
    """
    JET Regions:
        hfs_sol
        lfs_sol
        hfs_div
        lfs_div
        xpt_conreg
        hfs_lower
        lfs_lower
        rhon_09_10
    """

    Sion = np.zeros((len(cases)))
    Srec = np.zeros((len(cases)))
    icnt = 0
    for casekey, case in cases.items():

        o = Plot(workdir, case['case'])
        print('get_region_Sion_Srec ', case['case'])
        for in_region in regions:

            for name, region in o.data2d.regions.items():
                if name == in_region:
                    Sion[icnt] += region.Sion
                    Srec[icnt] += region.Srec
        icnt += 1

    return Sion, Srec


def get_Idiv(workdir, cases):

    hfs_Idiv = np.zeros((len(cases)))
    lfs_Idiv = np.zeros((len(cases)))
    hfs_Idiv_prof = []
    hfs_Idiv_prof_x = []
    lfs_Idiv_prof = []
    lfs_Idiv_prof_x = []
    hfs_Idiv_prof_sqrtpsi = []
    lfs_Idiv_prof_sqrtpsi = []

    outdict = {}
    
    icnt = 0
    for casekey, case in cases.items():

        o = Plot(workdir, case['case_pyproc'])
        print('get_Idiv ', case['case_pyproc'])

        hfs_Idiv_prof.append(o.data2d.pflxd_IT['ydata'][:o.data2d.pflxd_IT['npts']])
        lfs_Idiv_prof.append(o.data2d.pflxd_OT['ydata'][:o.data2d.pflxd_OT['npts']])
        
        lfs_Idiv_prof_x.append(o.data2d.pflxd_OT['xdata'][:o.data2d.pflxd_OT['npts']])
        hfs_Idiv_prof_x.append(o.data2d.pflxd_IT['xdata'][:o.data2d.pflxd_IT['npts']])
        
        lfs_Idiv_prof_sqrtpsi.append(np.sqrt(o.data2d.psi_OT['ydata'][:o.data2d.psi_OT['npts']]))
        hfs_Idiv_prof_sqrtpsi.append(np.sqrt(o.data2d.psi_IT['ydata'][:o.data2d.psi_IT['npts']]))
        
        hfs_Idiv[icnt] = np.sum(o.data2d.pflxd_IT['ydata'][:o.data2d.pflxd_IT['npts']])
        lfs_Idiv[icnt] = np.sum(o.data2d.pflxd_OT['ydata'][:o.data2d.pflxd_OT['npts']])
        icnt += 1

    outdict['hfs_Idiv_prof']=hfs_Idiv_prof
    outdict['lfs_Idiv_prof']=lfs_Idiv_prof
    outdict['lfs_Idiv_prof_x']=lfs_Idiv_prof_x
    outdict['hfs_Idiv_prof_x']=hfs_Idiv_prof_x
    outdict['lfs_Idiv_prof_sqrtpsi']=lfs_Idiv_prof_sqrtpsi
    outdict['hfs_Idiv_prof_sqrtpsi']=hfs_Idiv_prof_sqrtpsi
    
    outdict['hfs_Idiv']=hfs_Idiv
    outdict['lfs_Idiv']=lfs_Idiv
    
    return outdict


def get_omp(workdir, cases):

    ne_prof = []
    ni_prof = []
    te_prof = []
    ti_prof = []
    da_prof = []
    dm_prof = []
    psi_prof = []
    x_prof = []

    outdict = {}
    
    for casekey, case in cases.items():

        o = Plot(workdir, case['case'])
        print('get_omp', case['case'])

        ne_prof.append(o.data2d.ne_OMP['ydata'][:o.data2d.ne_OMP['npts']])
        ni_prof.append(o.data2d.ni_OMP['ydata'][:o.data2d.ni_OMP['npts']])
        te_prof.append(o.data2d.te_OMP['ydata'][:o.data2d.te_OMP['npts']])
        ti_prof.append(o.data2d.ti_OMP['ydata'][:o.data2d.ti_OMP['npts']])
        da_prof.append(o.data2d.da_OMP['ydata'][:o.data2d.da_OMP['npts']])
        dm_prof.append(o.data2d.dm_OMP['ydata'][:o.data2d.dm_OMP['npts']])
        psi_prof.append(o.data2d.psi_OMP['ydata'][:o.data2d.psi_OMP['npts']])
        x_prof.append(o.data2d.ne_OMP['xdata'][:o.data2d.ne_OMP['npts']])

    outdict['ne_omp']=np.asarray(ne_prof)
    outdict['ni_omp']=np.asarray(ni_prof)
    outdict['te_omp']=np.asarray(te_prof)
    outdict['ti_omp']=np.asarray(ti_prof)
    outdict['da_omp']=np.asarray(da_prof)
    outdict['dm_omp']=np.asarray(dm_prof)
    outdict['rho_pol_omp']=np.sqrt(np.asarray(psi_prof))
    outdict['x_omp']=np.asarray(x_prof)

    return outdict



def get_spec_Te_ne(workdir, cases, spec='KT3', cherab_bridge=False):
    
    outdict = {}
    
    Te_lo = []
    Te_hi = []
    ne = []

    denel_OT = []
    teve_OT = []
    x_teve_OT = []
    denel_IT = []
    teve_IT = []
    x_teve_IT = []
    
    psi_OT = []
    psi_IT = []

#    spec_line_dict = OrderedDict([
#        ('1', {'1': OrderedDict([('1215.2', ['2', '1'])])}),
#    ])
#    if cherab_bridge:
#        plot_dict = {
#            'cherab_abs_factor': {'1215.2': 115., '3969.5': 347.},
#            'spec_line_dict':spec_line_dict,
#            'cherab_bridge_results': True,
#            'cherab_reflections': True,
#        }
#    else:
#        plot_dict = {
#            'spec_line_dict':spec_line_dict,
#            'cherab_bridge_results': False,
#            'cherab_reflections': False,
#        }

    for casekey, case in cases.items():
        o = Plot(workdir, case['case_pyproc'], plot_dict=None)
        
        print('get_spec_Te_ne ', case['case_pyproc'])

        p2 = o.get_line_int_sorted_data_by_chord_id(spec, ['chord', 'p2'])
        R = p2[:, 0]
        Te_hi.append(o.get_line_int_sorted_data_by_chord_id(spec, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_400_500']))
        Te_lo.append(o.get_line_int_sorted_data_by_chord_id(spec, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_300_360']))
        ne.append(o.get_line_int_sorted_data_by_chord_id(spec, ['los_int', 'stark', 'fit', 'ne']))

        # Also return the actual OSP Te and ne

        _denel_OT = o.data2d.denel_OT['ydata'][:o.data2d.denel_OT['npts']]
        denel_OT.append(_denel_OT)

        x_teve_OT.append(o.data2d.teve_OT['xdata'][:o.data2d.teve_OT['npts']])
        _teve_OT = o.data2d.teve_OT['ydata'][:o.data2d.teve_OT['npts']]
        teve_OT.append(_teve_OT)
        
        psi_OT.append(np.sqrt(o.data2d.psi_OT['ydata'][:o.data2d.psi_OT['npts']]))
        
        _denel_IT = o.data2d.denel_IT['ydata'][:o.data2d.denel_IT['npts']]
        denel_IT.append(_denel_IT)

        x_teve_IT.append(o.data2d.teve_IT['xdata'][:o.data2d.teve_IT['npts']])
        _teve_IT = o.data2d.teve_IT['ydata'][:o.data2d.teve_IT['npts']]
        teve_IT.append(_teve_IT)
        
        psi_IT.append(np.sqrt(o.data2d.psi_IT['ydata'][:o.data2d.psi_IT['npts']]))

    outdict['spec_x'] = R
    outdict['spec_Te_lo'] = np.asarray(Te_lo)
    outdict['spec_Te_hi'] = np.asarray(Te_hi)
    outdict['OT_x'] = np.asarray(x_teve_OT)
    outdict['OT_Te'] = np.asarray(teve_OT)
    outdict['OT_ne'] = np.asarray(denel_OT)
    outdict['OT_rho_pol'] = np.sqrt(np.asarray(psi_OT))
    outdict['IT_x'] = np.asarray(x_teve_IT)
    outdict['IT_Te'] = np.asarray(teve_IT)
    outdict['IT_ne'] = np.asarray(denel_IT)
    outdict['IT_rho_pol'] = np.sqrt(np.asarray(psi_IT))
    
    return outdict


def get_nH_ISP_OSP(workdir, cases):

    hfs_nH = np.zeros((len(cases)))
    lfs_nH = np.zeros((len(cases)))

    icnt = 0
    for casekey, case in cases.items():

        o = Plot(workdir, case['case'])

        # OSP nH
        x = o.data2d.da_IT['xdata'][:o.data2d.da_IT['npts']]
        idx, xval = find_nearest(x, 0.0)
        hfs_nH[icnt] = o.data2d.da_IT['ydata'][idx+1]
        lfs_nH[icnt] = o.data2d.da_OT['ydata'][idx+1]

        # hfs_nH[icnt] = np.max(o.data2d.da_IT['ydata'][:o.data2d.da_IT['npts']])
        # lfs_nH[icnt] = np.max(o.data2d.da_OT['ydata'][:o.data2d.da_OT['npts']])
        icnt += 1

    return hfs_nH, lfs_nH


def get_nH_IT_OT(workdir, cases):

    hfs_nH = []
    lfs_nH = []

    for casekey, case in cases.items():

        o = Plot(workdir, case['case'])

        # OSP nH
        x_IT = o.data2d.da_IT['xdata'][:o.data2d.da_IT['npts']]
        x_OT = o.data2d.da_OT['xdata'][:o.data2d.da_OT['npts']]
        hfs_nH.append(o.data2d.da_IT['ydata'][:o.data2d.da_IT['npts']])
        lfs_nH.append(o.data2d.da_OT['ydata'][:o.data2d.da_OT['npts']])

    return np.asarray(x_IT), np.asarray(x_OT), np.asarray(hfs_nH), np.asarray(lfs_nH)



def get_nHdelL_OT(workdir, cases, trans='H21', config='VT5C',
                  cherab_bridge=False, opt_thin=False):
    """

    Get outer target nHdelL estiamte from Ly alpha/Dalpha excitation component emission. Take the
    average of the two chords at the outer strike point as represenative of the outer target nHdelL estimate.

    """
    nHdelL = []

    if opt_thin:
        n0delL_key = 'n0delL_fit_thin'
    else:
        n0delL_key = 'n0delL_fit'

    spec_line_dict = OrderedDict([
        ('1', {'1': OrderedDict([('1215.2', ['2', '1'])])}),
    ])
    if cherab_bridge:
        plot_dict = {
            'cherab_abs_factor': {'1215.2': 115., '3969.5': 347.},
            'spec_line_dict': spec_line_dict,
            'cherab_bridge_results': True,
            'cherab_reflections': True,
        }
    else:
        plot_dict = {
            'spec_line_dict': spec_line_dict,
            'cherab_bridge_results': False,
            'cherab_reflections': False,
        }

    for casekey, case in cases.items():
        o = Plot(workdir, case['case'], plot_dict=plot_dict)
        p2 = o.get_line_int_sorted_data_by_chord_id('KT3', ['chord', 'p2'])
        R = p2[:, 0]
        nHdelL.append(
            o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', n0delL_key, trans, 'n0delL']))

    nHdelL = np.asarray(nHdelL)

    # OSP value at R=2.75, index 11
    abs_factor = 1.0

    if cherab_bridge:
        abs_factor = plot_dict['cherab_abs_factor']['1215.2']

    # nHdelL_OT = abs_factor * np.average(nHdelL[:,11:13], axis=1)
    if config == 'VT5C':
        nHdelL_OT = abs_factor * nHdelL[:,12]
    elif config == 'VV':
        nHdelL_OT = abs_factor * nHdelL[:,17] # VV last chord
    elif config=='VT6':
        nHdelL_OT = abs_factor * nHdelL[:,18] # VT6 last chord
#    nHdelL_OT = abs_factor * nHdelL[:,18] # VV chord on tile 7

    return nHdelL_OT



def get_lfs_adf11_Sion_Srec(workdir, cases, cherab_bridge=False, get_opt_thin_results=False,
                            use_Lyb_esc_fac_prof=False, refl_scal=1.0):
    """Extract ionization and recombination radial profiles from pyproc (or cherab_bridge) results.

    Parameters
    ----------
    cherab_bridge : bool
        Use this keyword for recovering the results from cherab_bridge instead of pyproc.

    get_opt_thin_results : bool
        Use this keyword for recovering the optically thin results from a pyproc case that includes Ly trapping.
        NOTE: Recombination from d72 only available from uncorrected adas data!

    use_Lyb_esc_fac_prof : bool
        Use Ly-beta escape factor radial profile. In this case Calculate Sion and Srec given a Ly-beta escape
        factor profile (e.g., derived from synth Ly-b/D-a ratio calculated with corrected adas data)

    refl_scal : float (default 1.0)
        Approximates the reflections impact on Siz, Srec. Since S/prpoto intensity*sxb(or axb), the scaling factor
        can be applied either to the line intensities or directly to the resulting Siz, Srec.

    Returns
    -------


    """

    Arate_1_3 = 0.5575e08  # s^-1
    Arate_2_3 = 0.4410e08  # s^-1

    Hlines_dict = OrderedDict([
        ('1215.2', ['2', '1']),
        ('6561.9', ['3', '2']),
        # ('4339.9', ['5', '2']),
        # ('4101.2', ['6', '2']),
        ('3969.5', ['7', '2'])
    ])

    spec_line_dict = OrderedDict([
        ('1', {'1': Hlines_dict})
    ])

    if cherab_bridge:
        plot_dict = {
            'cherab_abs_factor': {'1215.2': 115., '6561.9': 615., '3969.5': 347.},
            'spec_line_dict':spec_line_dict,
            'cherab_bridge_results': True,
            'cherab_reflections': True,
        }
        Sion_scal_H21 = 115. # correction factor for Ly-alpha cherab absolute intensity
        Sion_scal_H32 = 615. # correction factor for Ly-alpha cherab absolute intensity
        Srec_scal_H72 = 347. # correction factor for D-epsilon cherab absolute intensity
    else:
        plot_dict = {
            'spec_line_dict':spec_line_dict,
            'cherab_bridge_results': False,
            'cherab_reflections': False,
        }
        Sion_scal_H21=1.0
        Sion_scal_H32=1.0
        Srec_scal_H72=1.0

    Sion_adf11_H21 = np.zeros((len(cases)))
    Sion_adf11_H32 = np.zeros((len(cases)))
    Srec_adf11_H72 = np.zeros((len(cases)))

    return_items = {}

    icnt = 0
    for casekey, case in cases.items():

        o = Plot(workdir, case['case'], plot_dict=plot_dict)
        print('get_lfs_adf11_Sion_Srec ', case['case'])

        p2 = o.get_line_int_sorted_data_by_chord_id('KT3', ['chord', 'p2'])
        R = p2[:, 0]

        if use_Lyb_esc_fac_prof:

            # get Te, ne and Ly-alpha, D-alpha profiles
            ne = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'stark', 'fit', 'ne'])
            Te_hi = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_400_500'])
            Te_lo = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_300_360'])

            # Line intensities using the opacity corrected adas data (up to n=5)
            excit_Da = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '6564.57', 'excit'])
            recom_Da = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '6564.57', 'recom'])
            Da = refl_scal*(excit_Da + recom_Da)

            excit_Lyb = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '1025.72', 'excit'])
            recom_Lyb = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '1025.72', 'recom'])
            Lyb = refl_scal*(excit_Lyb + recom_Lyb)
            
            excit_Lya = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '1215.67', 'excit'])
            recom_Lya = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '1215.67', 'recom'])
            Lya = refl_scal*(excit_Lya + recom_Lya)

            # Line intensity using standard adas data
            excit_D72 = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '3969.5', 'excit'])
            recom_D72 = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '3969.5', 'recom'])
            D72 = refl_scal*(excit_D72 + recom_D72)

            Ly_beta_esc_fac = (Arate_2_3 / Arate_1_3) * (Lyb / Da)

            Siz_21, Siz_21_tot, sxb, nHdelL_21 = calc_Siz_synth(R, Lya, [2,1], Te_hi, ne, opacity=True,
                                                     Ly_beta_esc_fac_prof=Ly_beta_esc_fac)
            Sion_adf11_H21[icnt] = Siz_21_tot
            # print(R)
            # print(Siz)
            Siz_32, Siz_32_tot, sxb, nHdelL_32 = calc_Siz_synth(R, Da, [3,2], Te_hi, ne, opacity=True, Ly_beta_esc_fac_prof=Ly_beta_esc_fac)
            Sion_adf11_H32[icnt] = Siz_32_tot
            Srec_72, Srec_72_tot, axb = calc_Srec_synth(R, D72, [7,2], Te_lo, ne, opacity=True, Ly_beta_esc_fac_prof=Ly_beta_esc_fac)
            Srec_adf11_H72[icnt] = Srec_72_tot

            if not return_items:
                return_items['case'] = [case['case']]
                return_items['R'] = R
                return_items['Siz_21'] = [Siz_21.tolist()]
                return_items['Siz_32'] = [Siz_32.tolist()]
                return_items['Srec_72'] = [Srec_72.tolist()]
                return_items['Ly_beta_esc_fac'] = [Ly_beta_esc_fac.tolist()]
                return_items['Lya'] = [Lya.tolist()]
                return_items['Da'] = [Da.tolist()]
                return_items['Te_hi'] = [Te_hi.tolist()]
                return_items['ne'] = [ne.tolist()]
                return_items['nHdelL_21'] = [nHdelL_21.tolist()]
                return_items['nHdelL_32'] = [nHdelL_32.tolist()]
            else:
                return_items['case'].append(case['case'])
                return_items['Siz_21'].append(Siz_21.tolist())
                return_items['Siz_32'].append(Siz_32.tolist())
                return_items['Srec_72'].append(Srec_72.tolist())
                return_items['Ly_beta_esc_fac'].append(Ly_beta_esc_fac.tolist())
                return_items['Lya'].append(Lya.tolist())
                return_items['Da'].append(Da.tolist())
                return_items['Te_hi'].append(Te_hi.tolist())
                return_items['ne'].append(ne.tolist())
                return_items['nHdelL_21'].append(nHdelL_21.tolist())
                return_items['nHdelL_32'].append(nHdelL_32.tolist())

        else:
            # Using opt thick Ly-beta, but without any opacity corrections (corresponds to case in experiment with opt thin analysis)
            if get_opt_thin_results:
                Siz_21 = refl_scal*(Sion_scal_H21*o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'adf11_fit_opt_thin', 'H21', 'Sion']))
                Sion_adf11_H21[icnt] = refl_scal*(Sion_scal_H21*np.sum(o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'adf11_fit_opt_thin', 'H21', 'Sion'])))
                Siz_32 = refl_scal*(Sion_scal_H32*o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'adf11_fit_opt_thin', 'H32', 'Sion']))
                Sion_adf11_H32[icnt] = refl_scal*(Sion_scal_H32*np.sum(o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'adf11_fit_opt_thin', 'H32', 'Sion'])))
                Srec_72 = refl_scal*(Srec_scal_H72*o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'adf11_fit_opt_thin', 'H72', 'Srec']))
                Srec_adf11_H72[icnt] = refl_scal*(Srec_scal_H72*np.sum(o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'adf11_fit_opt_thin', 'H72', 'Srec'])))

                # Optically thick Lya and Da
                Lya_excit = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '1215.67', 'excit'])
                Lya_recom = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '1215.67', 'recom'])
                Lya = refl_scal * (Lya_excit + Lya_recom)

                Da_excit = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '6564.57', 'excit'])
                Da_recom = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '6564.57', 'recom'])
                Da = refl_scal * (Da_excit + Da_recom)


            else:
                Siz_21 = refl_scal*(Sion_scal_H21*o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'adf11_fit', 'H21', 'Sion']))
                Sion_adf11_H21[icnt] = refl_scal*(Sion_scal_H21*np.sum(o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'adf11_fit', 'H21', 'Sion'])))
                Siz_32 = refl_scal*(Sion_scal_H32*o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'adf11_fit', 'H32', 'Sion']))
                Sion_adf11_H32[icnt] = refl_scal*(Sion_scal_H32*np.sum(o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'adf11_fit', 'H32', 'Sion'])))
                Srec_72 = refl_scal*(Srec_scal_H72*o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'adf11_fit', 'H72', 'Srec']))
                Srec_adf11_H72[icnt] = refl_scal*(Srec_scal_H72*np.sum(o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'adf11_fit', 'H72', 'Srec'])))

                # Optically thin Lya and Da
                Lya_excit = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '1215.2', 'excit'])
                Lya_recom = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '1215.2', 'recom'])
                Lya = refl_scal * (Lya_excit + Lya_recom)

                Da_excit = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '6561.9', 'excit'])
                Da_recom = o.get_line_int_sorted_data_by_chord_id('KT3', ['los_int', 'H_emiss', '6561.9', 'recom'])
                Da = refl_scal * (Da_excit + Da_recom)

            if not return_items:
                return_items['case'] = [case['case']]
                return_items['R'] = R
                return_items['Siz_21'] = [Siz_21.tolist()]
                return_items['Siz_32'] = [Siz_32.tolist()]
                return_items['Srec_72'] = [Srec_72.tolist()]
                return_items['Lya'] = [Lya.tolist()]
                return_items['Da'] = [Da.tolist()]
            else:
                return_items['case'].append(case['case'])
                return_items['Siz_21'].append(Siz_21.tolist())
                return_items['Siz_32'].append(Siz_32.tolist())
                return_items['Srec_72'].append(Srec_72.tolist())
                return_items['Lya'].append(Lya.tolist())
                return_items['Da'].append(Da.tolist())

        icnt += 1

    if not use_Lyb_esc_fac_prof:
        return_items['nHdelL_21'] = get_nHdelL_OT(workdir, cases, trans='H21', cherab_bridge=False, opt_thin=True)
        return_items['nHdelL_32'] = get_nHdelL_OT(workdir, cases, trans='H32', cherab_bridge=False, opt_thin=True)

    return Sion_adf11_H21, Sion_adf11_H32, Srec_adf11_H72, return_items

if __name__ == "__main__":

    workdir = '/work/bloman/pyproc/'
    case = {}
    case['case'] = 'bloman_cmg_catalog_edge2d_jet_81472_mar1920_seq#4'
    get_var_by_ring(workdir, case, nSOLrings=5)
