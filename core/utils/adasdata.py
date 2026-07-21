import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

TRANSITION_TABLE = {(3, 2): "6561.9",
                    (4, 2): "4860.6",
                    (5, 2): "4339.9",
                    (6, 2): "4101.2",
                    (7, 2): "3969.5",
                    (8, 2): "3888.5",
                    (9, 2): "3834.9",
                    (10, 2):"3797.4"}

class ADF:
    inv4pi = 1/(4*np.pi)

    def __init__(self, atomic_transitions_list, path: str = "adas/pec12#h_pju#h0.dat", wavelength: str = '6561.9', discard_low_N = True):
        if discard_low_N:
            trans_to_keep = []
            for i, trans in enumerate(atomic_transitions_list):
                if trans[0] < 6:
                    print(f"discarding line {trans} (Use AMJUEL for N<=6)")
                    continue
                else:
                    trans_to_keep.append(trans)
            self.atomic_transitions_list = trans_to_keep
        else:
            self.atomic_transitions_list = atomic_transitions_list
        self.M = len(self.atomic_transitions_list)

        self.path = path
        self.raw_lines = self._read_file()
        self.blocks = self._extract_blocks()
        self.data = self._parse_all_blocks()
        self.current_wl = list(self.data.keys())[0]
        self.current_pec = "EXCIT" 
        self.te = self.data[self.current_wl][self.current_pec]["te"]
        self.ne = self.data[self.current_wl][self.current_pec]["ne"]
        self.interp = self.data[self.current_wl][self.current_pec]["interp"]
        self.set_wavelength(wavelength)

    def set_wavelength(self, new_wavelength: str):
        if new_wavelength in self.data:
            self.current_wl = new_wavelength
            self.te = self.data[self.current_wl][self.current_pec]["te"]
            self.ne = self.data[self.current_wl][self.current_pec]["ne"]
            self.interp = self.data[self.current_wl][self.current_pec]["interp"]
        else:
            print(f"Wavelength {new_wavelength} not found in data\nCurrent wavelength is {self.current_wl}")
            
        
    # -------------------------
    # File reading / cleaning
    # -------------------------
    def _read_file(self):
        """Read file, remove comments and empty lines, normalize formatting."""
        clean = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("C"):
                    continue
                clean.append(line.replace("D", "E"))
        return clean

    # -------------------------
    # Block extraction
    # -------------------------
    def _extract_blocks(self):
        """
        Extract blocks keyed by wavelength.
        Each block starts with a line containing 'A'.
        """
        blocks = {}
        current_key = None
        current_block = []

        for line in self.raw_lines[1:]:
            if "A" in line:  # header line
                if current_key is not None:
                    if current_key not in blocks.keys():
                        blocks[current_key] = [current_block]
                    else:
                        blocks[current_key].append(current_block)

                # wavelength is first token (strip trailing 'A')
                wl = line.split()[0].replace("A", "")
                current_key = wl
                current_block = [line]
            else:
                current_block.append(line)

        # last block
        if current_key not in blocks.keys():
            blocks[current_key] = [current_block]
        else:
            blocks[current_key].append(current_block)

        return blocks

    # -------------------------
    # Parsing
    # -------------------------
    def _parse_block(self, block):
        """Parse a single wavelength block into (ne, te, em)."""

        header = block[0].split()
        num_ne = int(header[1])
        num_te = int(header[2])
        pec_type = header[8]
        # Flatten numeric data
        values = []
        for line in block[1:]:
            values.extend([float(x) for x in line.split()])

        values = np.array(values)

        # Extract sections
        idx = 0

        ne = values[idx:idx + num_ne] * 1e6  # cm^-3 → m^-3
        idx += num_ne

        te = values[idx:idx + num_te]
        idx += num_te

        em = values[idx:].reshape((num_ne, num_te)).T

        return pec_type, ne, te, em

    def _parse_all_blocks(self):
        """Parse all wavelength blocks into structured dict."""
        parsed = {}
        
        for wl, blocks in self.blocks.items():
            for block in blocks:
                pec_type, ne, te, em = self._parse_block(block)
                interp = RegularGridInterpolator(
                    (te, ne), em, bounds_error=False, fill_value=None
                )
                if wl not in parsed:
                    parsed[wl] = { pec_type: {
                        "ne": ne,
                        "te": te,
                        "em": em,
                        "interp": interp
                        }
                    }
                else:
                    parsed[wl][pec_type] = {
                        "ne": ne,
                        "te": te,
                        "em": em,
                        "interp": interp
                        }
            
        return parsed

    # -------------------------
    # Public API
    # -------------------------
    def interpolate(self, te, ne, pec_type, wl):
        """Evaluate interpolator for given wavelength."""
        return self.data[wl][pec_type]["interp"]((te, ne))
    
    def f(self, te, ne):
        return self.interpolate(self.current_wl, te, ne)

    def calc_photon_rates(self, te, ne, nh):
        rows = []
        
        for i in range(self.M):
            wl = TRANSITION_TABLE[self.atomic_transitions_list[i]] 
            row = (
                  self.interpolate(te, ne, "EXCIT", wl) * nh*ne
                + self.interpolate(te, ne, "RECOM", wl) * ne*ne
                    
                )* self.inv4pi*1e-6
            rows.append(row)

        return np.vstack(rows)
    
if __name__ == "__main__":
    adf = ADF()