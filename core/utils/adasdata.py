import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import os, json
import urllib.parse
import urllib.request

ADAS_DB_PATH = os.path.join(os.environ.get('PESDT_HOME', os.path.expanduser('~') + "/PESDT/"), "adas_db/")

with open(os.path.join(ADAS_DB_PATH, "adf_dict.json"), "r") as f:
    ADF_DICT = json.load(f)


def populate_adas_db(website = 'http://open.adas.ac.uk/download/'):
    #ADF15:
    for species, adaspath in ADF_DICT["ADF15"].items():
        target = os.path.join(ADAS_DB_PATH, adaspath)
        if not os.path.isfile(target):
            directory = os.path.dirname(target)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            url = urllib.parse.urljoin(website, adaspath.replace('#', '][').lstrip('/'))
            urllib.request.urlretrieve(url, target)

    for species, adaspath in ADF_DICT["ADF11"]["plt"].items():
        target = os.path.join(ADAS_DB_PATH, adaspath)
        if not os.path.isfile(target):
            directory = os.path.dirname(target)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            url = urllib.parse.urljoin(website, adaspath.replace('#', '][').lstrip('/'))
            urllib.request.urlretrieve(url, target)
    for species, adaspath in ADF_DICT["ADF11"]["prb"].items():
        target = os.path.join(ADAS_DB_PATH, adaspath)
        if not os.path.isfile(target):
            directory = os.path.dirname(target)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            url = urllib.parse.urljoin(website, adaspath.replace('#', '][').lstrip('/'))
            urllib.request.urlretrieve(url, target)
    return

class ADF11():
    inv4pi = 1/(4*np.pi)

    def __init__(self, species: str = "H", **kwargs):

        self.species = species

        self.path_plt  = kwargs.get( "path_plt",os.path.join(ADAS_DB_PATH, ADF_DICT["ADF11"]["plt"][species]))
        self.path_prb  = kwargs.get( "path_prb",os.path.join(ADAS_DB_PATH, ADF_DICT["ADF11"]["prb"][species]))
        self._read_files()

    def _read_files(self):

        with open(self.path_plt, "r") as f:
            lines = f.readlines()
            header = lines.pop(0)
            num_ne, num_te = header.split()[1:3]
            num_ne = int(num_ne); num_te= int(num_te)
            num_data = num_ne*num_te

            lines.pop(0) # remove separator ----

            num_data_per_line = len(lines[0].split())
            num_ne_te_lines = int(np.ceil((num_ne+ num_te)/num_data_per_line))
            
            ne_te_data = np.concatenate([ np.array([float(x) for x in line.split()]) for line in lines[:num_ne_te_lines]])

            self.ne_plt = ne_te_data[:num_ne]
            self.te_plt = ne_te_data[num_ne:]

            data_lines = lines[num_ne_te_lines+1:]
            num_data_lines = int(np.ceil(num_data/num_data_per_line))

            self.data_plt = np.concatenate([[np.array([np.float64(x) for x in line.split()]) for line in data_lines[:num_data_lines]]]).reshape((num_te, num_ne))
            self.interp_plt = RegularGridInterpolator((self.te_plt, self.ne_plt), self.data_plt, bounds_error=False, fill_value=None)

        with open(self.path_prb, "r") as f:
            lines = f.readlines()
            header = lines.pop(0)
            num_ne, num_te = header.split()[1:3]
            num_ne = int(num_ne); num_te= int(num_te)
            num_data = num_ne*num_te

            lines.pop(0) # remove separator ----

            num_data_per_line = len(lines[0].split())
            num_ne_te_lines = int(np.ceil((num_ne+ num_te)/num_data_per_line))
            num_data_lines = int(np.ceil(num_data/num_data_per_line))
            ne_te_data = np.concatenate([ np.array([float(x) for x in line.split()]) for line in lines[:num_ne_te_lines]])

            self.ne_prb = ne_te_data[:num_ne]
            self.te_prb = ne_te_data[num_ne:]

            data_lines = lines[num_ne_te_lines+1:]

            self.data_prb = np.concatenate([[np.array([np.float64(x) for x in line.split()]) for line in data_lines[:num_data_lines]]]).reshape((num_te, num_ne))
            self.interp_prb = RegularGridInterpolator((self.te_prb, self.ne_prb), self.data_prb, bounds_error=False, fill_value=None)
        

    # -------------------------
    # Public API
    # -------------------------
    def interpolate_plt(self, te, ne):

        ne_ = np.log10(ne*1e-6)
        te_ = np.log10(te)
        return 1e-6*10**self.interp_plt((te_, ne_))

    def interpolate_prb(self, te, ne):
    
        ne_ = np.log10(ne*1e-6)
        te_ = np.log10(te)
        return 1e-6*10**self.interp_prb((te_, ne_))

class ADF15():
    inv4pi = 1/(4*np.pi)

    def __init__(self, species: str = "H"):

        self.species = species

        self.path  = os.path.join(ADAS_DB_PATH, ADF_DICT["ADF15"][species])
        self.raw_lines = self._read_file()
        self.blocks = self._extract_blocks()
        self.data = self._parse_all_blocks()
        
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
        return self.data[wl][pec_type]["interp"]((te, ne))*1e-6
    

    
if __name__ == "__main__":
    adf = ADF15()