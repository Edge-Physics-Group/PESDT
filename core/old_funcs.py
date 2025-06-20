# contains old versions of some functions, kept here for reference, if something in the new functions is found broken
# not usable as is

 @staticmethod
    def recover_line_int_ff_fb_Te_old(res_dict):
        """
            RECOVER LINE-AVERAGED ELECTRON TEMPERATURE FROM FF-FB CONTINUUM SPECTRA
            Balmer edge ratio estimate: 360 nm and 400 nm
            Balmer ff-fb below edge ratio: 300 nm and 400 nm
            Balmer ff-fb above edge ratio: 400 nm and 500 nm
        """
        cont_ratio_360_400 = continuo_read.get_fffb_intensity_ratio_fn_T(360.0, 400.0, 1.0, save_output=True, restore=False)
        cont_ratio_300_360 = continuo_read.get_fffb_intensity_ratio_fn_T(300.0, 360.0, 1.0, save_output=True, restore=False)
        cont_ratio_400_500 = continuo_read.get_fffb_intensity_ratio_fn_T(400.0, 500.0, 1.0, save_output=True, restore=False)

        for diag_key in res_dict.keys():
            for chord_key in res_dict[diag_key].keys():

                logger.info(f"Fitting ff+fb continuum spectra, LOS id= : {diag_key}, {chord_key}")

                wave_fffb = np.asarray(res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['wave'])
                synth_data_fffb = np.asarray(
                    res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['intensity'])
                idx_300, val = AnalyseSynthDiag.find_nearest(wave_fffb, 300.0)
                idx_360, val = AnalyseSynthDiag.find_nearest(wave_fffb, 360.0)
                idx_400, val = AnalyseSynthDiag.find_nearest(wave_fffb, 400.0)
                idx_500, val = AnalyseSynthDiag.find_nearest(wave_fffb, 500.0)
                ratio_360_400 = synth_data_fffb[idx_360] / synth_data_fffb[idx_400]
                ratio_300_360 = synth_data_fffb[idx_300] / synth_data_fffb[idx_360]
                ratio_400_500 = synth_data_fffb[idx_400] / synth_data_fffb[idx_500]
                icont_ratio, vcont_ratio = AnalyseSynthDiag.find_nearest(cont_ratio_360_400[:, 1], ratio_360_400)
                fit_te_360_400 = cont_ratio_360_400[icont_ratio, 0]
                icont_ratio, vcont_ratio = AnalyseSynthDiag.find_nearest(cont_ratio_300_360[:, 1], ratio_300_360)
                fit_te_300_360 = cont_ratio_300_360[icont_ratio, 0]
                icont_ratio, vcont_ratio = AnalyseSynthDiag.find_nearest(cont_ratio_400_500[:, 1], ratio_400_500)
                fit_te_400_500 = cont_ratio_400_500[icont_ratio, 0]

                ##### Add fit Te result to dictionary
                res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum'] = {
                    'fit': {'fit_te_360_400': fit_te_360_400, 'fit_te_300_360': fit_te_300_360,
                            'fit_te_400_500': fit_te_400_500, 'units': 'eV'}}

                ###############################################################
                # CALCULATE EFFECTIVE DEL_L USING LINE-INT ne AND Te VALUES AND CONTINUUM
                ###############################################################
                try:
                    cond= 'ne' in res_dict[diag_key][chord_key]['los_int']['stark']['fit']
                except:
                    cond = False
                if cond:
                    params = Parameters()
                    params.add('delL', value=0.5, min=0.0001, max=10.0)
                    params.add('te_360_400', value=fit_te_360_400)
                    fit_ne = res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne']
                    params.add('ne', value=fit_ne)
                    params['te_360_400'].vary = False
                    params['ne'].vary = False

                    fit_result = minimize(AnalyseSynthDiag.residual_continuo, params, args=(wave_fffb, synth_data_fffb),
                                          method='leastsq')
                    data_fit_ff_fb = AnalyseSynthDiag.residual_continuo(fit_result.params, wave_fffb, None, None)

                    fit_report(fit_result)

                    vals = fit_result.params.valuesdict()

                    ##### Add fit delL result to dictionary
                    res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['delL_360_400'] = \
                        vals['delL']
    @staticmethod
    def recover_line_int_Stark_ne_old(res_dict, data_source = "AMJUEL"):
        """
            RECOVER LINE-AVERAGED ELECTRON DENSITY FROM H6-2 STARK BROADENED SPECTRA
        """
        mmm_coeff = {'6t2': {'C': 3.954E-16, 'a': 0.7149, 'b': 0.028}}

        for diag_key in res_dict.keys():
            for chord_key in res_dict[diag_key].keys():

                logger.info(f"Fitting Stark broadened H6-2 spectra, LOS id= : {diag_key}, {chord_key}")

                for H_line_key in res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'].keys():

                    if res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][0] == '6' and \
                                    res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][1] == '2':

                        wave_stark = np.asarray(res_dict[diag_key][chord_key]['los_int']['stark']['wave'])
                        synth_data_stark = np.asarray(res_dict[diag_key][chord_key]['los_int']['stark']['intensity'])

                        # print(wave_stark)
                        # print(synth_data_stark)

                        params = Parameters()
                        params.add('cwl', value=float(H_line_key) / 10.0)
                        if data_source == "AMJUEL":
                            params.add('area', value=float(
                                res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit'] +
                                res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['recom'] +
                                res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['h2'] +
                                res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['h2+'] +
                                res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['h-']))
                        else:
                            params.add('area', value=float(
                                res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit'] +
                                res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['recom']))
                        params.add('stark_fwhm', value=0.15, min=0.0001, max=10.0)

                        params['cwl'].vary = True
                        params['area'].vary = True

                        fit_result = minimize(AnalyseSynthDiag.residual_lorentz_52, params,
                                              args=(wave_stark,), kws={'data':synth_data_stark}, method='leastsq')
                        data_fit_ne = AnalyseSynthDiag.residual_lorentz_52(fit_result.params, wave_stark)

                        fit_report(fit_result)

                        vals = fit_result.params.valuesdict()

                        # Assume Te = 1 eV
                        fit_ne = np.power((vals['stark_fwhm'] / mmm_coeff['6t2']['C']),
                                          1. / mmm_coeff['6t2']['a'])

                        ##### Add fit ne result to dictionary
                        res_dict[diag_key][chord_key]['los_int']['stark']['fit'] = {'ne': fit_ne, 'units': 'm^-3'}

    def recover_line_int_particle_bal_old(self, res_dict, sion_H_transition=[[2,1], [3, 2]],
                                      srec_H_transition=[[7,2]], ne_scal=1.0,
                                      cherab_ne_Te_KT3_resfile=None):
        """
            ESTIMATE RECOMBINATION/IONISATION RATES USING ADF11 ACD, SCD COEFF

            cherab_ne_Te_KT3_resfile: replace pyproc ne, Te fits with the cherab results to account for reflections. This flag
            is used when ADAS Lyman opacity data is used since the Lyman trapping option is not available directly in
            cherab. Cherab processed file must already exist.
        """

        # Use ADAS adf15,11 data taking into account Ly-series trapping and hence a modification
        # to the ionization/recombination rates.
        if self.ADAS_dict_lytrap:
            ADAS_dict_local = self.ADAS_dict_lytrap
            spec_line_dict_key = 'spec_line_dict_lytrap'
        else:
            ADAS_dict_local = self.ADAS_dict
            spec_line_dict_key = 'spec_line_dict'

        for diag_key in res_dict.keys():
            for chord_key in res_dict[diag_key].keys():

                if (res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne'] and
                    res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']):

                    if cherab_ne_Te_KT3_resfile and diag_key=='KT3':
                        try:
                            with open(cherab_ne_Te_KT3_resfile, 'r') as f:
                                cherab_res_dict = json.load(f)
                        except IOError as e:
                            raise
                        if (cherab_res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne'] and
                                cherab_res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']):
                            fit_ne = ne_scal * cherab_res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne']
                            fit_Te = cherab_res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit'][
                                'fit_te_360_400']
                    else:
                        fit_ne = ne_scal*res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne']
                        fit_Te = res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']
                        # Use highest Te estimate from continuum (usually not available from experiment)
                        # fit_Te = res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_400_500']

                    logger.info(f"Ionization/recombination, LOS id= :{diag_key}, {chord_key}")

                    # area_cm2 = 2*pi*R*dW
                    w2unmod = res_dict[diag_key][chord_key]['chord']['w2']
                    # area_cm2 = 1.0e04 * 2.*np.pi*res_dict[diag_key][chord_key]['chord']['d2unmod']*res_dict[diag_key][chord_key]['coord']['v2'][0]
                    area_cm2 = 1.0e04 * 2. * np.pi * w2unmod * \
                               res_dict[diag_key][chord_key]['chord']['p2'][0]
                    idxTe, Te_val = self.find_nearest(self.ADAS_dict['adf11']['1'].Te_arr, fit_Te)
                    idxne, ne_val = self.find_nearest(self.ADAS_dict['adf11']['1'].ne_arr, fit_ne * 1.0E-06)

                    # RECOMBINATION:
                    # NOTE: D7-2 line must be read from standard ADAS adf15 data as it is above the
                    # max transition available in the Ly-trapped adf15 files
                    # But still use modified adf11 data, if Ly-trapping case
                    for itran, tran in enumerate(srec_H_transition):
                        for H_line_key in res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'].keys():
                            if res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][0] == str(srec_H_transition[itran][0]) and \
                                            res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][1] == str(srec_H_transition[itran][1]):
                                hij = res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit'] + \
                                      res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['recom']
                                srec = 1.0E-04 * area_cm2 * hij * 4. * np.pi * \
                                       ADAS_dict_local['adf11']['1'].acd[idxTe, idxne] / \
                                       self.ADAS_dict['adf15']['1']['1'][H_line_key + 'recom'].pec[idxTe, idxne]

                                # Add to results dict
                                tran_str = 'H' + str(srec_H_transition[itran][0]) + str(srec_H_transition[itran][1])
                                if 'adf11_fit' in res_dict[diag_key][chord_key]['los_int']:
                                    res_dict[diag_key][chord_key]['los_int']['adf11_fit'][tran_str] = {'Srec': srec, 'units': 's^-1'}
                                else:
                                    res_dict[diag_key][chord_key]['los_int']['adf11_fit'] = {tran_str:{'Srec': srec, 'units': 's^-1'}}

                                if self.ADAS_dict_lytrap:
                                    # In case of Ly-trapping, also calculate with standard adf11 data for comparison
                                    _srec = 1.0E-04 * area_cm2 * hij * 4. * np.pi * \
                                           self.ADAS_dict['adf11']['1'].acd[idxTe, idxne] / \
                                           self.ADAS_dict['adf15']['1']['1'][H_line_key + 'recom'].pec[idxTe, idxne]

                                    # Add to results dict
                                    tran_str = 'H' + str(srec_H_transition[itran][0]) + str(srec_H_transition[itran][1])
                                    if 'adf11_fit_opt_thin' in res_dict[diag_key][chord_key]['los_int']:
                                        res_dict[diag_key][chord_key]['los_int']['adf11_fit_opt_thin'][tran_str] = {'Srec': _srec, 'units': 's^-1'}
                                    else:
                                        res_dict[diag_key][chord_key]['los_int']['adf11_fit_opt_thin'] = {tran_str:{'Srec': _srec, 'units': 's^-1'}}

                    # IONIZATION:
                    # Use Ly-trapping adf15,11 data if available (ADAS_dict_local at this point already contains adf11 opacity data, if selected in the input json file)
                    for itran, tran in enumerate(sion_H_transition):
                        for H_line_key in res_dict[diag_key][chord_key][spec_line_dict_key]['1']['1'].keys():
                            if res_dict[diag_key][chord_key][spec_line_dict_key]['1']['1'][H_line_key][0] == str(sion_H_transition[itran][0]) and \
                                            res_dict[diag_key][chord_key][spec_line_dict_key]['1']['1'][H_line_key][1] == str(sion_H_transition[itran][1]):
                                h_intensity = (res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit']+
                                               res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['recom'])
                                sion = 1.0E-04 * area_cm2 * h_intensity * 4. * np.pi * \
                                       ADAS_dict_local['adf11']['1'].scd[idxTe, idxne] / \
                                       ADAS_dict_local['adf15']['1']['1'][H_line_key + 'excit'].pec[idxTe, idxne]

                                # Add to results dict
                                tran_str = 'H' + str(sion_H_transition[itran][0]) + str(sion_H_transition[itran][1])
                                if tran_str in res_dict[diag_key][chord_key]['los_int']['adf11_fit']:
                                    res_dict[diag_key][chord_key]['los_int']['adf11_fit'][tran_str].update({'Sion': sion, 'units': 's^-1'})
                                else:
                                    res_dict[diag_key][chord_key]['los_int']['adf11_fit'][tran_str] = {'Sion': sion, 'units': 's^-1'}

                                # Use optically thick Ly-alpha intensity, but opt. thin SXB (adf11 and adf15) coeffs.
                                # This is analogous of real
                                # situation where the Ly-alpha measurement is from optically thick plasma,
                                # but interpretation assumes optically thin plasma and uncorrected adas data is used.
                                if self.ADAS_dict_lytrap:
                                    for H_line_key in res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'].keys():
                                        if res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][0] == str(sion_H_transition[itran][0]) and \
                                                res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][1] == str(sion_H_transition[itran][1]):
                                            _sion = 1.0E-04 * area_cm2 * h_intensity * 4. * np.pi * \
                                                   self.ADAS_dict['adf11']['1'].scd[idxTe, idxne] / \
                                                   self.ADAS_dict['adf15']['1']['1'][H_line_key + 'excit'].pec[idxTe, idxne]
                                            # Add to results dict
                                            tran_str = 'H' + str(sion_H_transition[itran][0]) + str(sion_H_transition[itran][1])
                                            if 'adf11_fit_opt_thin' in res_dict[diag_key][chord_key]['los_int']:
                                                res_dict[diag_key][chord_key]['los_int']['adf11_fit_opt_thin'][tran_str] = {'Sion': _sion, 'units': 's^-1'}
                                            else:
                                                res_dict[diag_key][chord_key]['los_int']['adf11_fit_opt_thin'] = {tran_str:{'Sion': _sion, 'units': 's^-1'}}
    
    def recover_delL_atomden_product_old(self, res_dict, sion_H_transition=[[2,1], [3, 2]], excit_only=True):
        """
            ESTIMATE DEL_L * ATOMIC DENSITY PRODUCT FROM LY-ALPHA ASSUMING EXCITATION DOMINATED

            excit_only flag added to isolate the Ly-alpha/D-alpha component to allow apples-to-apples comparison of
            nH*delL with experiment, since in experiment the
            recombination component of Ly-alpha is smaller outboard of the OSP on the horizontal target than in
            modelling. Otherwise,
            the larger recombinaiont component in EDGE2D modelling overestimates nH*delL, such that a comparison to
            experiment values is not valid.
            NOTE: including the Ly-alpha recombination contr. has little impact on S_iz_tot estimates, but large (~50%) impact
            on the max nH*delL on the outer target.
        """

        # Use ADAS adf15,11 data taking into account Ly-series trapping
        if self.ADAS_dict_lytrap:
            ADAS_dict_local = self.ADAS_dict_lytrap
            spec_line_dict_key = 'spec_line_dict_lytrap'
        else:
            ADAS_dict_local = self.ADAS_dict
            spec_line_dict_key = 'spec_line_dict'

        for diag_key in res_dict.keys():
            for chord_key in res_dict[diag_key].keys():

                if (res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne'] and
                        res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']):

                    fit_ne = res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne']
                    fit_Te = res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']

                    for itran, tran in enumerate(sion_H_transition):
                        logger.info(f"delL * n0 from transition {str(tran)} LOS id= : {diag_key}, {chord_key}")

                        for H_line_key in res_dict[diag_key][chord_key][spec_line_dict_key]['1']['1'].keys():
                            if res_dict[diag_key][chord_key][spec_line_dict_key]['1']['1'][H_line_key][0] == str(
                                    sion_H_transition[itran][0]) and \
                                    res_dict[diag_key][chord_key][spec_line_dict_key]['1']['1'][H_line_key][1] == str(
                                sion_H_transition[itran][1]):
                                if excit_only:
                                    h_ij = res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit']
                                else:
                                    h_ij = res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit'] + \
                                           res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['recom']
                                idxTe, Te_val = self.find_nearest(ADAS_dict_local['adf15']['1']['1'][H_line_key + 'recom'].Te_arr, fit_Te)
                                idxne, ne_val = self.find_nearest(ADAS_dict_local['adf15']['1']['1'][H_line_key + 'recom'].ne_arr, fit_ne * 1.0E-06)
                                n0delL_Hij_tmp = 4. * np.pi * 1.0e-04 * h_ij / (ADAS_dict_local['adf15']['1']['1'][H_line_key + 'excit'].pec[idxTe, idxne] * ne_val)
                                n0delL_Hij_tmp = n0delL_Hij_tmp * 1.0e06 * 1.0e-02  # convert to m^-2
                                ##### Add fit n0*delL result to dictionary
                                tran_str = 'H' + str(sion_H_transition[itran][0]) + str(sion_H_transition[itran][1])
                                if 'n0delL_fit' in res_dict[diag_key][chord_key]['los_int']:
                                    res_dict[diag_key][chord_key]['los_int']['n0delL_fit'][tran_str]={'n0delL': n0delL_Hij_tmp, 'units': 'm^-2'}
                                else:
                                    res_dict[diag_key][chord_key]['los_int']['n0delL_fit']={tran_str:{'n0delL': n0delL_Hij_tmp, 'units': 'm^-2'}}

                                # Use optically thick Ly-alpha/D-alpha intensity, but opt. thin PEC coeffs.
                                # This is analogous to real
                                # situation where the Ly-alpha measurement is from optically thick plasma,
                                # but interpretation assumes optically thin plasma and uncorrected adas data is used.
                                if self.ADAS_dict_lytrap:
                                    for H_line_key in res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'].keys():
                                        if res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][0] == str(sion_H_transition[itran][0]) and \
                                                res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][1] == str(sion_H_transition[itran][1]):
                                            idxTe, Te_val = self.find_nearest(
                                                self.ADAS_dict['adf15']['1']['1'][H_line_key + 'recom'].Te_arr, fit_Te)
                                            idxne, ne_val = self.find_nearest(
                                                self.ADAS_dict['adf15']['1']['1'][H_line_key + 'recom'].ne_arr,
                                                fit_ne * 1.0E-06)
                                            _n0delL_Hij_tmp = 4. * np.pi * 1.0e-04 * h_ij / (
                                                    self.ADAS_dict['adf15']['1']['1'][H_line_key + 'excit'].pec[
                                                            idxTe, idxne] * ne_val)
                                            _n0delL_Hij_tmp = _n0delL_Hij_tmp * 1.0e06 * 1.0e-02  # convert to m^-2
                                            # Add to results dict
                                            tran_str = 'H' + str(sion_H_transition[itran][0]) + str(sion_H_transition[itran][1])
                                            if 'n0delL_fit_thin' in res_dict[diag_key][chord_key]['los_int']:
                                                res_dict[diag_key][chord_key]['los_int']['n0delL_fit_thin'][tran_str] = {
                                                    'n0delL': _n0delL_Hij_tmp, 'units': 'm^-2'}
                                            else:
                                                res_dict[diag_key][chord_key]['los_int']['n0delL_fit_thin'] = {
                                                    tran_str: {'n0delL': _n0delL_Hij_tmp, 'units': 'm^-2'}}
