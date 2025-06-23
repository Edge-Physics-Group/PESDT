import numpy as np

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

def interpolate_yacora_rate(te: np.float64, ne: np.float64, rate: dict):
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
    return interpolate_idw(te, ne, points)
