import numpy as np
import os

class YACORA():

    def __init__(self, data_path: str):
        self.data_path = data_path
        # Create data paths
        h_data_path: str = os.path.join(str(data_path), "PopKoeff_n=3_from_H.txt")
        h_rec_data_path: str = os.path.join(str(data_path), "PopKoeff_n=3_from_H+.txt")
        h2_data_path: str = os.path.join(str(data_path), "PopKoeff_n=3_from_H2.txt")
        h2_pos_data_path: str = os.path.join(str(data_path), "PopKoeff_n=3_from_H2+.txt")
        h3_pos_data_path: str = os.path.join(str(data_path), "PopKoeff_n=3_from_H3+.txt")
        hneg1_data_path: str = os.path.join(str(data_path), "PopKoeff_n=3_from_H-_with_H2+.txt")
        hneg2_data_path: str = os.path.join(str(data_path), "PopKoeff_n=3_from_H-_with_H+.txt")
        # Read data
        h_data, _ = self.read_yacora_rate(h_data_path)
        h_rec_data,_ = self.read_yacora_rate(h_rec_data_path)
        h2_data, _ = self.read_yacora_rate(h2_data_path)
        h2_pos_data,_ = self.read_yacora_rate(h2_pos_data_path)
        h3_pos_data, _ = self.read_yacora_rate(h3_pos_data_path)
        hneg1_data,_ = self.read_yacora_rate(hneg1_data_path)
        hneg2_data,_ = self.read_yacora_rate(hneg2_data_path)
        # Assing to dicts for calculating rates
        self.h_rates = {3: h_data}
        self.h_rec_rates = {3: h_rec_data}
        self.h2_rates = {3: h2_data}
        self.h2_pos_rates = {3: h2_pos_data}
        self.h3_pos_rates = {3: h3_pos_data}
        self.hneg1_rates = {3: hneg1_data}
        self.hneg2_rates = {3: hneg2_data}
    
    @staticmethod
    def A_coeff(transition):
        '''
        Returns the Einstein coefficient for transition "m"->"n"

        transition (m: int, n: int)

        Coefficients from "Elementary Processes in Hydrogen-Helium Plasmas" (1987), by Janev, Appendix A.2.The original source of the table is Wiese et al. (1966)
        '''

        coeff_dict = {
            1: {2: 4.699e8, 3: 5.575e7, 4: 1.278e7, 5: 4.125e6, 6: 1.644e6}, 
            2: {3: 4.410e7, 4: 8.419e6, 5: 2.530e6, 6: 9.732e5, 7: 4.389e5},
            3: {4: 8.989e6, 5: 2.201e6, 6: 7.783e5, 7: 3.358e5, 8: 1.651e5},
            4: {5: 2.699e6, 6: 7.711e5, 7: 3.041e5, 8: 1.424e5, 9: 7.459e4}
        }

        return coeff_dict[transition[1]][transition[0]]
    
    
    @staticmethod
    def read_yacora_rate(filename: str, header_size: int = 27):
        """

        """
        raw_data = np.loadtxt(filename, skiprows=header_size, dtype=np.float64)
        a_vals = raw_data[:, 0]
        b_c_vals = raw_data[:, 1:]  # (b, c) pairs

        # Get sorted unique 'a' values
        unique_a = np.unique(a_vals)

        # Group (b, c) by each unique 'a'
        grouped_data = np.array([b_c_vals[a_vals == a] for a in unique_a])

        data = {}
        i = 0
        for a in unique_a:
            data[a] = grouped_data[i]
            i +=1
        return  data, raw_data
    
    @staticmethod
    def interpolate_idw(x, y, points, power=2):
        """
        Interpolates z at (x, y) using inverse distance weighting from four points.

        points: list of tuples (xi, yi, zi)
        """
        num = 0.0
        denom = 0.0
        for xi, yi, zi in points:
            dx = x - xi
            dy = y - yi
            dist_sq = dx*dx + dy*dy
            if dist_sq == 0:
                return zi  # Exact match
            weight = 1 / (dist_sq ** (power / 2))
            num += weight * zi
            denom += weight
        return num / denom

    def calc_photon_rate(self, 
                         transition: tuple, 
                         te: np.ndarray, 
                         ne: np.ndarray, 
                         nh: np.ndarray, 
                         nh2: np.ndarray, 
                         nh2_pos: np.ndarray, 
                         nh3_pos: np.ndarray, 
                         nhneg: np.ndarray):
        # Make sure that the transition is in integers
        transition = (int(transition[0]), int(transition[1]))
        h_rate = self.h_rates.get(transition[0], None)
        h_rec_rate = self.h_rec_rates.get(transition[0], None)
        h2_rate = self.h2_rates.get(transition[0], None)
        h2_pos_rate = self.h2_pos_rates.get(transition[0], None)
        h3_pos_rate = self.h3_pos_rates.get(transition[0], None)
        hneg1_rate = self.hneg1_rates.get(transition[0], None)
        hneg2_rate = self.hneg2_rates.get(transition[0], None)

        if h2_rate is None or h_rate is None:
            raise Exception(f"Transition {transition} not found in YACORA data. Check your transitions and YACORA database. \nYACORA data path: {self.data_path}")
        acoeff = self.A_coeff(transition)

        h_rate_interp = self.interpolate_yacora_rate_arr(te, ne, h_rate)
        h_rec_rate_interp = self.interpolate_yacora_rate_arr(te, ne, h_rec_rate)
        h2_rate_interp = self.interpolate_yacora_rate_arr(te, ne, h2_rate)
        h2_pos_rate_interp = self.interpolate_yacora_rate_arr(te, ne, h2_pos_rate)
        h3_pos_rate_interp = self.interpolate_yacora_rate_arr(te, ne, h3_pos_rate)
        hneg1_rate_interp = self.interpolate_yacora_rate_arr(te, ne, hneg1_rate)
        hneg2_rate_interp = self.interpolate_yacora_rate_arr(te, ne, hneg2_rate)
        hneg_tot = hneg1_rate_interp + hneg2_rate_interp

        h_emiss = acoeff*h_rate_interp * ne * nh /(4.0*np.pi)
        h_rec_emiss = acoeff*h_rec_rate_interp * ne * ne /(4.0*np.pi)
        h2_emiss = acoeff*h2_rate_interp * ne * nh2 /(4.0*np.pi)
        h2_pos_emiss = acoeff*h2_pos_rate_interp * ne * nh2_pos /(4.0*np.pi)
        h3_pos_emiss = acoeff*h3_pos_rate_interp * ne * nh3_pos /(4.0*np.pi)
        hneg_emiss = acoeff*hneg_tot * ne * nhneg /(4.0*np.pi)

        tot = h_emiss + h_rec_emiss + h2_emiss + h2_pos_emiss + h3_pos_emiss + hneg_emiss


        return h_emiss, h_rec_emiss, h2_emiss, h2_pos_emiss, h3_pos_emiss, hneg_emiss, tot

    def interpolate_yacora_rate_arr(self, te_arr_in, ne_arr_in, rate):
        ret = np.zeros(te_arr_in.shape)
        for i in range(len(te_arr_in)):
            ret[i] = self.interpolate_yacora_rate(te_arr_in[i], ne_arr_in[i], rate)
        return ret

    def interpolate_yacora_rate(self, te: np.float64, ne: np.float64, rate: dict):
        """
        
        """
        te_arr = np.fromiter(rate.keys(), dtype=float)

        te_idx = np.searchsorted(te_arr, te)
        te_idx = np.clip(te_idx, 0, len(te_arr) - 2)

        te_lo = te_arr[te_idx]
        te_hi = te_arr[te_idx+1]

        ne_arr_lo = rate[te_lo][:, 0]
        ne_arr_hi = rate[te_hi][:, 0]

        pop_arr_lo = rate[te_lo][:, 1]
        pop_arr_hi = rate[te_hi][:, 1]


        ne_idx_lo = np.searchsorted(ne_arr_lo, ne)
        ne_idx_lo = np.clip(ne_idx_lo, 0, len(ne_arr_lo) - 2)
        ne_idx_hi = np.searchsorted(ne_arr_hi, ne)
        ne_idx_hi = np.clip(ne_idx_hi, 0, len(ne_arr_lo) - 2)

        ne_lo_lo = ne_arr_lo[ne_idx_lo]
        ne_lo_hi = ne_arr_lo[ne_idx_lo+1]
        
        ne_hi_lo = ne_arr_hi[ne_idx_hi]
        ne_hi_hi = ne_arr_hi[ne_idx_hi+1]

        pop_lo_lo = pop_arr_lo[ne_idx_lo]
        pop_lo_hi = pop_arr_lo[ne_idx_lo+1]
        
        pop_hi_lo = pop_arr_hi[ne_idx_hi]
        pop_hi_hi = pop_arr_hi[ne_idx_hi+1]

        points = [(te_lo, ne_lo_lo, pop_lo_lo), (te_lo, ne_lo_hi, pop_lo_hi), (te_hi, ne_hi_lo, pop_hi_lo), (te_hi, ne_hi_hi, pop_hi_hi)]
        return self.interpolate_idw(te, ne, points)

    