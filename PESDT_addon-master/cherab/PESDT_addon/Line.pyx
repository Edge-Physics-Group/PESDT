# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.


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
    cdef :
        Element element
        int charge
        tuple transition
    def __init__(self, Element element, int charge, tuple transition):
        self.element = element
        self.charge = charge
        self.transition = transition

    def __repr__(self):
        return '<Line: {}, {}, {}>'.format(self.element.name, self.charge, self.transition)

    def __hash__(self):
        return hash((self.element, self.charge, self.transition))

    def __richcmp__(self, object other, int op):

        cdef Line line

        if not isinstance(other, Line):
            return NotImplemented

        line = <Line> other
        if op == 2:     # __eq__()
            return self.element == line.element and self.charge == line.charge and self.transition == line.transition
        elif op == 3:   # __ne__()
            return self.element != line.element or self.charge != line.charge or self.transition != line.transition
        else:
            return NotImplemented