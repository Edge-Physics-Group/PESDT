# cython: language_level=3
from cherab.core.atomic cimport Element, Isotope
from cherab.core.distribution cimport DistributionFunction
from cherab.core cimport Species


# immutable, so the plasma doesn't have to track changes
cdef class PESDTSpecies(Species):
    """
    A class representing a given plasma species.

    A plasma in Cherab will be composed of 1 or more Species objects. A species
    can be uniquely identified through its element and charge state.

    When instantiating a Species object a 6D distribution function (3 space, 3 velocity)
    must be defined. The DistributionFunction object provides the base interface for
    defining a distribution function, it could be a reduced analytic representation
    (such as a Maxwellian for example) or a fully numerically interpolated 6D function.

    :param Element element: The element object of this species.
    :param int charge: The charge state of the species.
    :param DistributionFunction distribution: A distribution function for this species.

    """


    def __init__(self, Element element, int charge, DistributionFunction distribution):
        # Allow any distribution and any charge, as molecules and negative ions do exist
        self.element = element
        self.charge = charge
        self.distribution = distribution

    def __repr__(self):
        return '<Species: element={}, charge={}>'.format(self.element.name, self.charge)


cdef class PESDTElement(Element):
    """
    A wrapper for Cherab Element, to keep note that we are going to break every single convention
    of Cherab, with the addition of "base element". The purpose of the base element is have a truly
    valid element, which can be passed to Cherab functions, if needed. For example, D_2 could be
    created like
        PESDTElement("Deuterium2", "D2", 2, 4.0, Deuterium)
    The atomic number and weight could be confused with helium, but instead we may pass the base
    element, which is a "valid" element.
    """
    cdef Element _base_element
    def __init__(self, str name, str symbol, int atomic_number, double atomic_weight, Element base_element):

        super().__init__(name, symbol, atomic_number, atomic_weight)
        self._base_element = base_element


    cpdef Element base_element(self):
        return self._base_element

hydrogen = Element('hydrogen', 'H', 1, (1.00784 + 1.00811) / 2)
deuterium = Isotope('deuterium', 'D', hydrogen, 2, 2.0141017778)