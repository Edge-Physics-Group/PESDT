

class Region:

    def __init__(self, name, Rmin, Rmax, Zmin, Zmax, include_confined=True, include_SOL=True, include_PFR=True):
        self.name = name
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.include_confined = include_confined
        self.include_SOL = include_SOL
        self.include_PFR= include_PFR

        # container of cell objects belonging to the region
        self.cells = []

        # Total radiated power - main ions
        self.Prad_H = 0.0
        self.Prad_H_Lytrap = 0.0
        self.Prad_units = 'W'

        # Main ion ionization and recombination [units: s^-1]
        self.Sion = 0.0
        self.Srec = 0.0

    def cell_in_region(self, cell, shply_sep_poly):

        if cell.Z >= self.Zmin and cell.Z <= self.Zmax and cell.R >= self.Rmin and cell.R <= self.Rmax:
            if self.include_confined and shply_sep_poly.contains(cell.poly):
                return True
            if self.include_SOL and not shply_sep_poly.contains(cell.poly):
                return True
            if self.include_PFR and shply_sep_poly.contains(cell.poly):
                return True

        return False