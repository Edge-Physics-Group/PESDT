
from cherab.core.atomic.elements import hydrogen
from cherab.core.atomic cimport Element
import sys

_molecule_index = {}

"""
Implemnts molecule class for passing molecules into cherab

"""

cdef class Molecule(Element):
    cdef dict __dict__

    """
    Class representing a Molecule.
    
    """

    def __init__(self, name, symbol, element, mass_number, atomic_weight):

        super().__init__(name, symbol, element.atomic_number, atomic_weight)
        self.mass_number = mass_number
        self.element = element

    def __repr__(self):
        return '<Molecule: {}>'.format(self.name)

    def __hash__(self):
        return hash((self.name, self.symbol, self.atomic_number, self.atomic_weight, self.mass_number))

    def __richcmp__(self, other, op):

        if not isinstance(other, Molecule):
            return NotImplemented

        cdef Molecule e
        e = other
        if op == 2:     # __eq__()
            return (self.name == e.name and self.symbol == e.symbol and
                    self.atomic_number == e.atomic_number and self.atomic_weight == e.atomic_weight and
                    self.element == e.element and self.mass_number == e.mass_number)
        elif op == 3:   # __ne__()
            return (self.name != e.name or self.symbol != e.symbol or
                    self.atomic_number != e.atomic_number or self.atomic_weight != e.atomic_weight or
                    self.element != e.element or self.mass_number != e.mass_number)
        else:
            return NotImplemented

def _build_molecule_index():
    """
    Populates an isotope search dictionary.

    Populates the isotope index so users can search for isotopes by name or
    symbol.
    """

    module = sys.modules[__name__]
    for name in dir(module):
        obj = getattr(module, name)
        if type(obj) is Molecule:
            # lookup by name or symbol including variations e.g. D and H2 refer to deuterium)
            _molecule_index[obj.symbol.lower()] = obj
            _molecule_index[obj.name.lower()] = obj
            _molecule_index[obj.element.symbol.lower() + str(obj.mass_number)] = obj
            _molecule_index[obj.element.name.lower() + str(obj.mass_number)] = obj

def lookup_molecule(v, number = None):
    """
    Finds a molecule by name, symbol or number.

    Molecules are uniquely determined by the element type and mass number. These
    can be specified as a single string or a combination of element and mass number.

    :param v: Search string, integer or element.
    :param number: Integer mass number
    :return: Element object.
    """

    if type(v) is Molecule:
        return v

    # full information contained in string
    key = str(v).lower()

    try:
        return _molecule_index[key]
    except KeyError:
        if number:
            raise ValueError('Could not find an molecule object for the element \'{}\' and number \'{}\'.'.format(v, number))
        else:
            raise ValueError('Could not find an molecule object for the key \'{}\'.'.format(v))

# List of Molecules
Deuterium2 = Molecule('Deuterium2', 'D2', hydrogen, 4, 4)
Deuterium3 = Molecule('Deuterium3', 'D3', hydrogen, 6, 6)
Tritium2 = Molecule('Tritium2', 'T2', hydrogen, 6, 6)
Tritium3 = Molecule('Tritium3', 'T3', hydrogen, 9, 9)

_build_molecule_index()