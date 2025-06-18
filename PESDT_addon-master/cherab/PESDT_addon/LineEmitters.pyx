# cython: language_level=3
from cherab.core.atomic cimport AtomicData
from cherab.core.plasma cimport PlasmaModel
from cherab.core.utility import PerCm3ToPerM3, PhotonToJ, Cm3ToM3


from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core cimport Line, Species, Plasma, Beam
from cherab.core.model.lineshape cimport GaussianLine, LineShapeModel
from cherab.core.utility.constants cimport RECIP_4_PI

import numpy as np


'''
Cherab AMJUEL plasma models

    Cherab itsel implements only ADAS plasma models and reading
    In the future these could be included as a cherab module, and Cythonized

'''
cdef class DirectEmission(PlasmaModel):
    cdef dict __dict__
    cdef Line _line
    cdef object _lineshape_class
    cdef object _lineshape_args
    cdef object _lineshape_kwargs
    
    def __init__(self, Line line, Plasma plasma=None, AtomicData atomic_data=None, object lineshape=None,
                 object lineshape_args=None, object lineshape_kwargs=None):
        

        super().__init__(plasma, atomic_data)

        self._line = line

        self._lineshape_class = lineshape or GaussianLine
        if not issubclass(self._lineshape_class, LineShapeModel):
            raise TypeError("The attribute lineshape must be a subclass of LineShapeModel.")

        if lineshape_args:
            self._lineshape_args = lineshape_args
        else:
            self._lineshape_args = []
        if lineshape_kwargs:
            self._lineshape_kwargs = lineshape_kwargs
        else:
            self._lineshape_kwargs = {}

        # ensure that cache is initialised
        self._change()

    def __repr__(self):
        return '<ExcitationLine: element={}, charge={}, transition={}>'.format(self._line.element.name, self._line.charge, self._line.transition)

    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):
        cdef double radiance
        # cache data on first run
        if self._target_species is None:
            self._populate_cache()

        radiance = self._target_species.distribution.emission(point.x, point.y, point.z)
        if radiance <= 0.0:
            return spectrum

        # add emission line to spectrum
        return self._lineshape.add_line(radiance, point, direction, spectrum)

    cdef int _populate_cache(self) except -1:

        # sanity checks
        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")
        if self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to an atomic data source.")

        if self._line is None:
            raise RuntimeError("The emission line has not been set.")

        # locate target species
        # set the emission to the current transition
        try:
            self._target_species = self._plasma.composition.get(self._line.element, self._line.charge)
            self._target_species.distribution.update_emission(self._line.transition)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified line "
                               "(element={}, ionisation={}).".format(self._line.element.symbol, self._line.charge))
        # identify wavelength
        self._wavelength = self._atomic_data.wavelength(self._line.element, self._line.charge, self._line.transition)

        # instance line shape renderer
        self._lineshape = self._lineshape_class(self._line, self._wavelength, self._target_species.base_element(), self._plasma,
                                                *self._lineshape_args, **self._lineshape_kwargs)
        

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._rates = None
        self._lineshape = None


cdef class LineExcitation_AM(PlasmaModel):
    cdef dict __dict__ 

    def __init__(self, Line line, Plasma plasma=None, AtomicData atomic_data=None, object lineshape=None,
                 object lineshape_args=None, object lineshape_kwargs=None):
        

        super().__init__(plasma, atomic_data)

        self._line = line

        self._lineshape_class = lineshape or GaussianLine
        if not issubclass(self._lineshape_class, LineShapeModel):
            raise TypeError("The attribute lineshape must be a subclass of LineShapeModel.")

        if lineshape_args:
            self._lineshape_args = lineshape_args
        else:
            self._lineshape_args = []
        if lineshape_kwargs:
            self._lineshape_kwargs = lineshape_kwargs
        else:
            self._lineshape_kwargs = {}

        # ensure that cache is initialised
        self._change()

    def __repr__(self):
        return '<ExcitationLine: element={}, charge={}, transition={}>'.format(self._line.element.name, self._line.charge, self._line.transition)

    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):
        cdef double ne, n, te, radiance
        # cache data on first run
        if self._target_species is None:
            self._populate_cache()

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        n = self._target_species.distribution.density(point.x, point.y, point.z)
        if n <= 0.0:
            return spectrum

        # add emission line to spectrum
        radiance = RECIP_4_PI * self._rates.evaluate(ne, te) * n
        return self._lineshape.add_line(radiance, point, direction, spectrum)

    cdef int _populate_cache(self) except -1:

        # sanity checks
        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")
        if self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to an atomic data source.")

        if self._line is None:
            raise RuntimeError("The emission line has not been set.")

        # locate target species
        try:
            self._target_species = self._plasma.composition.get(self._line.element, self._line.charge)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified line "
                               "(element={}, ionisation={}).".format(self._line.element.symbol, self._line.charge))

        #print('T1')
        # obtain rate function
        self._rates = self._atomic_data.H_excit(self._line.transition)

        # identify wavelength
        self._wavelength = self._atomic_data.wavelength(self._line.element, self._line.charge, self._line.transition)

        # instance line shape renderer
        self._lineshape = self._lineshape_class(self._line, self._wavelength, self._target_species, self._plasma,
                                                *self._lineshape_args, **self._lineshape_kwargs)
        

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._rates = None
        self._lineshape = None


cdef class LineRecombination_AM(PlasmaModel):
    cdef dict __dict__ 

    def __init__(self, Line line, Plasma plasma=None, AtomicData atomic_data=None, object lineshape=None,
                 object lineshape_args=None, object lineshape_kwargs=None):

        super().__init__(plasma, atomic_data)

        self._line = line

        self._lineshape_class = lineshape or GaussianLine
        if not issubclass(self._lineshape_class, LineShapeModel):
            raise TypeError("The attribute lineshape must be a subclass of LineShapeModel.")

        if lineshape_args:
            self._lineshape_args = lineshape_args
        else:
            self._lineshape_args = []
        if lineshape_kwargs:
            self._lineshape_kwargs = lineshape_kwargs
        else:
            self._lineshape_kwargs = {}

        # ensure that cache is initialised
        self._change()

    def __repr__(self):
        return '<RecombinationLine: element={}, charge={}, transition={}>'.format(self._line.element.name, self._line.charge+1, self._line.transition)

    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):
        cdef double ne, ni, te, radiance
        # cache data on first run
        if self._target_species is None:
            self._populate_cache()

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        ni = self._target_species.distribution.density(point.x, point.y, point.z)
        if ni <= 0.0:
            return spectrum

        # add emission line to spectrum
        radiance = RECIP_4_PI * self._rates.evaluate(ne, te) * ni
        return self._lineshape.add_line(radiance, point, direction, spectrum)

    cdef int _populate_cache(self) except -1:

        # sanity checks
        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")
        if self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to an atomic data source.")

        if self._line is None:
            raise RuntimeError("The emission line has not been set.")

        # locate target species, increment charge by 1 to get the main D ion density
        try:
            self._target_species = self._plasma.composition.get(self._line.element, self._line.charge +1)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified line "
                               "(element={}, ionisation={}).".format(self._line.element.symbol, self._line.charge))

        # obtain rate function
        self._rates = self._atomic_data.H_rec(self._line.transition)

        # identify wavelength
        self._wavelength = self._atomic_data.wavelength(self._line.element, self._line.charge, self._line.transition)

        # instance line shape renderer
        self._lineshape = self._lineshape_class(self._line, self._wavelength, self._target_species, self._plasma,
                                                *self._lineshape_args, **self._lineshape_kwargs)

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._rates = None
        self._lineshape = None

cdef class LineH2_AM(PlasmaModel):
    cdef dict __dict__ 

    def __init__(self, Line line, Plasma plasma=None, AtomicData atomic_data=None, object lineshape=None,
                 object lineshape_args=None, object lineshape_kwargs=None):

        super().__init__(plasma, atomic_data)

        self._plasma = plasma
        self._atomic_data = atomic_data

        self._line = line

        self._lineshape_class = lineshape or GaussianLine
        if not issubclass(self._lineshape_class, LineShapeModel):
            raise TypeError("The attribute lineshape must be a subclass of LineShapeModel.")

        if lineshape_args:
            self._lineshape_args = lineshape_args
        else:
            self._lineshape_args = []
        if lineshape_kwargs:
            self._lineshape_kwargs = lineshape_kwargs
        else:
            self._lineshape_kwargs = {}

        # ensure that cache is initialised
        self._change()

    def __repr__(self):
        return '<ExcitationLine: element={}, charge={}, transition={}>'.format(self._line.element.name, self._line.charge, self._line.transition)

    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):
        cdef double ne, ni, te, radiance
        # cache data on first run
        if self._target_species is None:
            self._populate_cache()

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        ni = self._target_species.distribution.density(point.x, point.y, point.z)
        if ni <= 0.0:
            return spectrum

        # add emission line to spectrum
        radiance = RECIP_4_PI * self._rates.evaluate(ne, te) * ni
        return self._lineshape.add_line(radiance, point, direction, spectrum)

    cdef int _populate_cache(self) except -1:

        # sanity checks
        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")
        if self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to an atomic data source.")

        if self._line is None:
            raise RuntimeError("The emission line has not been set.")

        # locate target species
        try:
            self._target_species = self._plasma.composition.get(self._line.element, self._line.charge)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified line "
                               "(element={}, ionisation={}).".format(self._line.element.symbol, self._line.charge))

        # obtain rate function
        self._rates = self._atomic_data.H2_diss( self._line.transition)

        # identify wavelength
        self._wavelength = self._atomic_data.wavelength(self._line.element, self._line.charge, self._line.transition)

        # instance line shape renderer
        self._lineshape = self._lineshape_class(self._line, self._wavelength, self._target_species, self._plasma,
                                                *self._lineshape_args, **self._lineshape_kwargs)

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._rates = None
        self._lineshape = None

cdef class LineH2_pos_AM(PlasmaModel):
    cdef dict __dict__ 

    def __init__(self, Line line, Plasma plasma=None, AtomicData atomic_data=None, object lineshape=None,
                 object lineshape_args=None, object lineshape_kwargs=None):

        super().__init__(plasma, atomic_data)

        self._line = line

        self._lineshape_class = lineshape or GaussianLine
        if not issubclass(self._lineshape_class, LineShapeModel):
            raise TypeError("The attribute lineshape must be a subclass of LineShapeModel.")

        if lineshape_args:
            self._lineshape_args = lineshape_args
        else:
            self._lineshape_args = []
        if lineshape_kwargs:
            self._lineshape_kwargs = lineshape_kwargs
        else:
            self._lineshape_kwargs = {}

        # ensure that cache is initialised
        self._change()

    def __repr__(self):
        return '<ExcitationLine: element={}, charge={}, transition={}>'.format(self._line.element.name, self._line.charge, self._line.transition)

    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):
        cdef double ne, ni, te, radiance

        # cache data on first run
        if self._target_species is None:
            self._populate_cache()

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        ni = self._target_species.distribution.density(point.x, point.y, point.z)
        if ni <= 0.0:
            return spectrum

        # add emission line to spectrum
        radiance = RECIP_4_PI * self._rates.evaluate(ne, te) * ni
        return self._lineshape.add_line(radiance, point, direction, spectrum)

    cdef int _populate_cache(self) except -1:

        # sanity checks
        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")
        if self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to an atomic data source.")

        if self._line is None:
            raise RuntimeError("The emission line has not been set.")

        # locate target species
        try:
            self._target_species = self._plasma.composition.get(self._line.element, self._line.charge +1)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified line "
                               "(element={}, ionisation={}).".format(self._line.element.symbol, self._line.charge))

        # obtain rate function
        self._rates = self._atomic_data.H2_pos_diss(self._line.transition)

        # identify wavelength
        self._wavelength = self._atomic_data.wavelength(self._line.element, self._line.charge, self._line.transition)

        # instance line shape renderer
        self._lineshape = self._lineshape_class(self._line, self._wavelength, self._target_species, self._plasma,
                                                *self._lineshape_args, **self._lineshape_kwargs)

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._rates = None
        self._lineshape = None

cdef class LineH_neg_AM(PlasmaModel):
    cdef dict __dict__ 

    def __init__(self, Line line, Plasma plasma=None, AtomicData atomic_data=None, object lineshape=None,
                 object lineshape_args=None, object lineshape_kwargs=None):

        super().__init__(plasma, atomic_data)

        self._plasma = plasma
        self._atomic_data = atomic_data


        self._line = line

        self._lineshape_class = lineshape or GaussianLine
        if not issubclass(self._lineshape_class, LineShapeModel):
            raise TypeError("The attribute lineshape must be a subclass of LineShapeModel.")

        if lineshape_args:
            self._lineshape_args = lineshape_args
        else:
            self._lineshape_args = []
        if lineshape_kwargs:
            self._lineshape_kwargs = lineshape_kwargs
        else:
            self._lineshape_kwargs = {}

        # ensure that cache is initialised
        self._change()

    def __repr__(self):
        return '<ExcitationLine: element={}, charge={}, transition={}>'.format(self._line.element.name, self._line.charge, self._line.transition)

    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):
        cdef double ne, ni, te, radiance

        # cache data on first run
        if self._target_species is None:
            self._populate_cache()

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        ni = self._target_species.distribution.density(point.x, point.y, point.z)
        if ni <= 0.0:
            return spectrum

        # add emission line to spectrum
        radiance = RECIP_4_PI * self._rates.evaluate(ne, te) * ni
        return self._lineshape.add_line(radiance, point, direction, spectrum)

    cdef int _populate_cache(self) except -1:

        # sanity checks
        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")
        if self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to an atomic data source.")

        if self._line is None:
            raise RuntimeError("The emission line has not been set.")

        # locate target species
        try:
            self._target_species = self._plasma.composition.get(self._line.element, self._line.charge)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified line "
                               "(element={}, ionisation={}).".format(self._line.element.symbol, self._line.charge))

        # obtain rate function
        self._rates = self._atomic_data.H_neg(self._line.transition)

        # identify wavelength
        self._wavelength = self._atomic_data.wavelength(self._line.element, self._line.charge, self._line.transition)

        # instance line shape renderer
        self._lineshape = self._lineshape_class(self._line, self._wavelength, self._target_species, self._plasma,
                                                *self._lineshape_args, **self._lineshape_kwargs)

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._rates = None
        self._lineshape = None

cdef class LineH3_pos_AM(PlasmaModel):
    cdef dict __dict__ 

    def __init__(self, Line line, Plasma plasma=None, AtomicData atomic_data=None, object lineshape=None,
                 object lineshape_args=None, object lineshape_kwargs=None):

        super().__init__(plasma, atomic_data)

        self._line = line

        self._lineshape_class = lineshape or GaussianLine
        if not issubclass(self._lineshape_class, LineShapeModel):
            raise TypeError("The attribute lineshape must be a subclass of LineShapeModel.")

        if lineshape_args:
            self._lineshape_args = lineshape_args
        else:
            self._lineshape_args = []
        if lineshape_kwargs:
            self._lineshape_kwargs = lineshape_kwargs
        else:
            self._lineshape_kwargs = {}

        # ensure that cache is initialised
        self._change()

    def __repr__(self):
        return '<ExcitationLine: element={}, charge={}, transition={}>'.format(self._line.element.name, self._line.charge, self._line.transition)

    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef double ne, ni, te, radiance

        # cache data on first run
        if self._target_species is None:
            self._populate_cache()

        ne = self._plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self._plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        ni = self._target_species.distribution.density(point.x, point.y, point.z)
        if ni <= 0.0:
            return spectrum

        # add emission line to spectrum
        radiance = RECIP_4_PI * self._rates.evaluate(ne, te)  * ni
        return self._lineshape.add_line(radiance, point, direction, spectrum)

    cdef int _populate_cache(self) except -1:

        # sanity checks
        if self._plasma is None:
            raise RuntimeError("The emission model is not connected to a plasma object.")
        if self._atomic_data is None:
            raise RuntimeError("The emission model is not connected to an atomic data source.")

        if self._line is None:
            raise RuntimeError("The emission line has not been set.")

        # locate target species
        try:
            self._target_species = self._plasma.composition.get(self._line.element, self._line.charge)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified line "
                               "(element={}, ionisation={}).".format(self._line.element.symbol, self._line.charge))

        # obtain rate function
        self._rates = self._atomic_data.H3_poss_diss( self._line.transition)

        # identify wavelength
        self._wavelength = self._atomic_data.wavelength( self._line.element, self._line.charge, self._line.transition)

        # instance line shape renderer
        self._lineshape = self._lineshape_class(self._line, self._wavelength, self._target_species, self._plasma,
                                                *self._lineshape_args, **self._lineshape_kwargs)

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._rates = None
        self._lineshape = None



