# cython: language_level=3

from cherab.core.atomic.elements cimport Element
from cherab.core cimport Line

cdef class PESDTLine(Line):
    """
    A class fully specifies an observed spectroscopic emission line.

    Note that wavelengths are not arguments to this class. This is because in
    principle the transition has already been fully specified with the other three
    arguments. The wavelength is looked up in the wavelength database of the
    atomic data provider.

    :param Element element: The atomic element/isotope to which this emission line belongs.
    :param int charge: The charge state of the element/isotope that emits this line.
    :param tuple transition: A two element tuple that defines the upper and lower electron
      configuration states of the transition. For hydrogen-like ions it may be enough to
      specify the n-levels with integers (e.g. (3,2)). For all other ions the full spectroscopic
      configuration string should be specified for both states. It is up to the atomic data
      provider package to define the exact notation.

    """
    cdef object _arb_transition
    def __init__(self, Element element, int charge, object transition):
        self.element = element
        self.charge = charge
        self.arb_transition = transition

    def __repr__(self):
        return '<Line: {}, {}, {}>'.format(self.element.name, self.charge, self.arb_transition)

    def __hash__(self):
        return hash((self.element, self.charge, self.arb_transition))

    def __richcmp__(self, object other, int op):

        cdef Line line

        if not isinstance(other, Line):
            return NotImplemented

        line = <Line> other
        if op == 2:     # __eq__()
            return self.element == line.element and self.charge == line.charge and self.arb_transition == line.arb_transition
        elif op == 3:   # __ne__()
            return self.element != line.element or self.charge != line.charge or self.arb_transition != line.arb_transition
        else:
            return NotImplemented
    @property
    def arb_transition(self):
        return self._arb_transition

    @arb_transition.setter
    def arb_transition(self, val):
        self._arb_transition = val
cdef class PESDTLineMol(Line):
    """
    A class fully specifies an observed spectroscopic emission line.

    Note that wavelengths are not arguments to this class. This is because in
    principle the transition has already been fully specified with the other three
    arguments. The wavelength is looked up in the wavelength database of the
    atomic data provider.

    :param Element element: The atomic element/isotope to which this emission line belongs.
    :param int charge: The charge state of the element/isotope that emits this line.
    :param str transition: the name of the emission band, e.g "fulcher"
    """
    cdef object _mol_transition

    def __init__(self, Element element, int charge, object transition):
        self.element = element
        self.charge = charge
        self.mol_transition = transition

    def __repr__(self):
        return '<Line: {}, {}, {}>'.format(self.element.name, self.charge, self.mol_transition)

    def __hash__(self):
        return hash((self.element, self.charge, self.mol_transition))

    def __richcmp__(self, object other, int op):

        cdef Line line

        if not isinstance(other, Line):
            return NotImplemented

        line = <Line> other
        if op == 2:     # __eq__()
            return self.element == line.element and self.charge == line.charge and self.mol_transition == line.mol_transition
        elif op == 3:   # __ne__()
            return self.element != line.element or self.charge != line.charge or self.mol_transition != line.mol_transition
        else:
            return NotImplemented

    @property
    def mol_transition(self):
        return self._mol_transition

    @mol_transition.setter
    def mol_transition(self, val):
        self._mol_transition = val
