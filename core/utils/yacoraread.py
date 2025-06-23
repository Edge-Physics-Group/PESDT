import numpy as np
import os

class YACORA():

    def __init__(self, data_path: str):
        self.data_path = data_path
        
        h_data_path: str = os.path.join(str(data_path), "PopKoeff_n=3_from_H.txt")
        h2_data_path: str = os.path.join(str(data_path), "PopKoeff_n=3_from_H2.txt")
        h_data, _ = self.read_yacora_rate(h_data_path)
        h2_data,_ = self.read_yacora_rate(h2_data_path)
        self.h_rates = {3: h_data}
        self.h2_rates = {3: h2_data}
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

    def calc_photon_rate(self, transition: tuple, te: np.ndarray, ne: np.ndarray, nh: np.ndarray, nh2: np.ndarray):
        # Make sure that the transition is in integers
        transition = (int(transition[0]), int(transition[1]))
        h_rate = self.h_rates.get(transition[0], None)
        h2_rate = self.h2_rates.get(transition[0], None)
        if h2_rate is None or h_rate is None:
            raise Exception(f"Transition {transition} not found in YACORA data. Check your transitions and YACORA database. \nYACORA data path: {self.data_path}")
        acoeff = self.A_coeff(transition)
        h_rate_interp = self.interpolate_yacora_rate_arr(te, ne, h_rate)
        h2_rate_interp = self.interpolate_yacora_rate_arr(te, ne, h2_rate)
        h_emiss = acoeff*h_rate_interp * ne * nh 
        h2_emiss = acoeff*h2_rate_interp * ne * nh2 

        return h_emiss, h2_emiss
    @staticmethod
    def interpolate_idw_vectorized(x, y, points, power=2):
        # NOT FUNCTIONAL
        """
        Vectorized inverse distance weighting interpolation for arrays x, y.

        points: shape (n_points, 3), with columns [xi, yi, zi]
        x, y: arrays of shape (N,)
        Returns: array of shape (N,)
        """
        xi = points[:, 0][:, None]  # shape (4, 1)
        yi = points[:, 1][:, None]  # shape (4, 1)
        zi = points[:, 2][:, None]  # shape (4, 1)

        dx = x[None, :] - xi        # shape (4, N)
        dy = y[None, :] - yi        # shape (4, N)
        dist_sq = dx**2 + dy**2     # shape (4, N)

        # Avoid division by zero
        eps = 1e-12
        dist_sq = np.where(dist_sq == 0, eps, dist_sq)

        weights = 1 / dist_sq**(power / 2)  # shape (4, N)
        weighted_vals = weights * zi       # shape (4, N)

        return np.sum(weighted_vals, axis=0) / np.sum(weights, axis=0)  # shape (N,)

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

        te_lo = te_arr[te_idx]
        te_hi = te_arr[te_idx+1]

        ne_arr_lo = rate[te_lo][:, 0]
        ne_arr_hi = rate[te_hi][:, 0]

        pop_arr_lo = rate[te_lo][:, 1]
        pop_arr_hi = rate[te_hi][:, 1]


        ne_idx_lo = np.searchsorted(ne_arr_lo, ne)
        ne_idx_hi = np.searchsorted(ne_arr_hi, ne)

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

    def interpolate_yacora_rate_vectorized(self, te_arr_in, ne_arr_in, rate):
        te_grid = np.array(sorted(rate.keys()))  # sorted Te values
        te_arr_in = np.asarray(te_arr_in)
        ne_arr_in = np.asarray(ne_arr_in)

        idx_te = np.searchsorted(te_grid, te_arr_in, side='right') - 1
        idx_te = np.clip(idx_te, 0, len(te_grid) - 2)  # ensure idx and idx+1 are valid

        te_lo = te_grid[idx_te]
        te_hi = te_grid[idx_te + 1]

        N = len(te_arr_in)
        pop_lo_lo = np.empty(N)
        pop_lo_hi = np.empty(N)
        pop_hi_lo = np.empty(N)
        pop_hi_hi = np.empty(N)
        ne_lo_lo = np.empty(N)
        ne_lo_hi = np.empty(N)
        ne_hi_lo = np.empty(N)
        ne_hi_hi = np.empty(N)

        for i in range(N):
            te0 = te_lo[i]
            te1 = te_hi[i]

            ne_vals0 = rate[te0][:, 0]
            pop_vals0 = rate[te0][:, 1]
            ne_vals1 = rate[te1][:, 0]
            pop_vals1 = rate[te1][:, 1]

            ne_idx0 = np.searchsorted(ne_vals0, ne_arr_in[i], side='right') - 1
            ne_idx1 = np.searchsorted(ne_vals1, ne_arr_in[i], side='right') - 1

            ne_idx0 = np.clip(ne_idx0, 0, len(ne_vals0) - 2)
            ne_idx1 = np.clip(ne_idx1, 0, len(ne_vals1) - 2)

            ne_lo_lo[i] = ne_vals0[ne_idx0]
            ne_lo_hi[i] = ne_vals0[ne_idx0 + 1]
            pop_lo_lo[i] = pop_vals0[ne_idx0]
            pop_lo_hi[i] = pop_vals0[ne_idx0 + 1]

            ne_hi_lo[i] = ne_vals1[ne_idx1]
            ne_hi_hi[i] = ne_vals1[ne_idx1 + 1]
            pop_hi_lo[i] = pop_vals1[ne_idx1]
            pop_hi_hi[i] = pop_vals1[ne_idx1 + 1]

        # Pack the 4 surrounding points
        points = np.stack([
            np.stack([te_lo, ne_lo_lo, pop_lo_lo], axis=1),
            np.stack([te_lo, ne_lo_hi, pop_lo_hi], axis=1),
            np.stack([te_hi, ne_hi_lo, pop_hi_lo], axis=1),
            np.stack([te_hi, ne_hi_hi, pop_hi_hi], axis=1),
        ])  # shape (4, N, 3)

        # Reshape for interpolation function: (4, N, 3) → (4, 3) x N cases
        interpolated = self.interpolate_idw_vectorized(te_arr_in, ne_arr_in, points.reshape(4, -1, 3))
        return interpolated
