# cython: language_level=3
from cherab.core.atomic cimport AtomicData
from cherab.core.plasma cimport PlasmaModel

from .LineShapes cimport DeltaLine
from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core cimport Line, Species, Plasma, Beam
from cherab.core.model.lineshape cimport GaussianLine, LineShapeModel
from cherab.core.utility.constants cimport RECIP_4_PI, ATOMIC_MASS, ELEMENTARY_CHARGE, SPEED_OF_LIGHT
from libc.math cimport sqrt, erf, M_SQRT2, floor, ceil, fabs, expl
from cherab.core.model.lineshape.doppler cimport doppler_shift, thermal_broadening
import numpy as np
cimport numpy as np
from numpy cimport PyArray_SimpleNew, PyArray_FILLWBYTE, NPY_FLOAT64, npy_intp, import_array
import_array()

DEF GAUSSIAN_CUTOFF_SIGMA = 10.0
"""
    PESDT DirectEmission model
    reads from a precalculated array, agnostic of the datasource
    As long as you pass emission on to the plasma (i.e. use PESDTSimulation),
    you can use any data source you want.
"""
cdef class DirectEmission(PlasmaModel):
    
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
        self._lineshape = self._lineshape_class(self._line, self._wavelength, self._target_species, self._plasma, self._atomic_data,
                                                *self._lineshape_args, **self._lineshape_kwargs)
        

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._lineshape = None

cdef class DirectEmissionMol(PlasmaModel):

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
        return '<ExcitationLine: element={}, charge={}, transition={}>'.format(self._line.element.name, self._line.charge, self._line.mol_transition)
    
    cdef double H2_wavelength(self, str band = "fulcher"):
        '''
        Returns the average wavelength of a H2 excitation band from a tabulated dictionary
        '''
        dct: dict = {
            "fulcher": 620.0,
            "werner":  130.0,
            "lyman":   150.0
            }
        return dct[band]

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
            self._target_species.distribution.update_emission(self._line.mol_transition)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified line "
                               "(element={}, ionisation={}).".format(self._line.element.symbol, self._line.charge))
        # identify wavelength
        self._wavelength = self.H2_wavelength(self._line.mol_transition)

        # instance line shape renderer
        self._lineshape = self._lineshape_class(self._line, self._wavelength, self._target_species, self._plasma, self._atomic_data,
                                                *self._lineshape_args, **self._lineshape_kwargs)
        

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._lineshape = None

cdef class OpaqueDeltaDirectEmission(PlasmaModel):

    def __init__(self, Line line, Plasma plasma=None, AtomicData atomic_data=None, object lineshape=None,
                 object lineshape_args=None, object lineshape_kwargs=None):
        

        super().__init__(plasma, atomic_data)

        self._line = line
        self.prev_init = False
        self._lineshape_class = GaussianLine
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
        cdef: 
            double radiance
            double absorbance
            double ds = 0.0
            npy_intp size
        # cache data on first run
        if self._target_species is None:
            self._populate_cache()
            size = spectrum.bins
            self.absorbances = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
            PyArray_FILLWBYTE(self.absorbances, 0)
            self.absorbances_mv = self.absorbances

        radiance = self._target_species.distribution.emission(point.x, point.y, point.z)
        if radiance <= 0.0:
            if self.prev_init:
                ds = point.distance_to(self.prev_point)
                self.prev_point = point.copy()
                if ds > 0.1: # New ray, jump to origin. Not sure if needed
                    PyArray_FILLWBYTE(self.absorbances, 0)
                    self.absorbances_mv = self.absorbances
            else:
                self.prev_point = point.copy()
                self.prev_init = True
                ds = 0.0
            return spectrum # Keep track of previous point
            
        absorbance = self._target_species.distribution.absorbance(point.x, point.y, point.z)
        # add emission line to spectrum
        if self.prev_init:
            ds = point.distance_to(self.prev_point)
            self.prev_point = point.copy()
            if ds > 0.1: # New ray, jump to origin. Not sure if needed
                PyArray_FILLWBYTE(self.absorbances, 0)
                self.absorbances_mv = self.absorbances
                ds = 0.0
        else:
            self.prev_point = point.copy()
            self.prev_init = True
            ds = 0.0
        # deposit all radiance into one bin
        self.absorbances_mv[0] += absorbance*ds
        spectrum.samples_mv[0] += radiance*expl(-self.absorbances_mv[0])


        return spectrum

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
            self._target_species.distribution.update_absorbance(self._line.transition)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified line "
                               "(element={}, ionisation={}).".format(self._line.element.symbol, self._line.charge))
        # identify wavelength
        self._wavelength = self._atomic_data.wavelength(self._line.element, self._line.charge, self._line.transition)

        # instance line shape renderer
        self._lineshape = self._lineshape_class(self._line, self._wavelength, self._target_species, self._plasma, self._atomic_data,
                                                *self._lineshape_args, **self._lineshape_kwargs)
        

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._lineshape = None
        self.prev_init = False

cdef class OpaqueGaussianDirectEmission(PlasmaModel):

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
        cdef: 
            double radiance
            double absorbance
            double ts, sigma, shifted_wavelength, Td
            Vector3D ion_velocity
            double ds = 0.0
            npy_intp size
        
        # cache data on first run
        if self._target_species is None:
            self._populate_cache()
            size = spectrum.bins
            self.absorbances = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
            PyArray_FILLWBYTE(self.absorbances, 0)
            self.absorbances_mv = self.absorbances


        radiance = self._target_species.distribution.emission(point.x, point.y, point.z)
        if radiance <= 0.0:
            if self.prev_init:
                ds = point.distance_to(self.prev_point)
                self.prev_point = point.copy()
                if ds > 0.1: # New ray, jump to origin. Not sure if needed
                    PyArray_FILLWBYTE(self.absorbances, 0)
                    self.absorbances_mv = self.absorbances
                    ds = 0.0
            else:
                self.prev_point = point.copy()
                self.prev_init = True
                ds = 0.0
            return spectrum # Keep track of previous point
            
        absorbance = self._target_species.distribution.absorbance(point.x, point.y, point.z)
        # add emission line to spectrum

                
        if self.prev_init:
            ds = point.distance_to(self.prev_point)
            self.prev_point = point.copy()
            if ds > 0.1: # New ray, jump to origin. Not sure if needed
                PyArray_FILLWBYTE(self.absorbances, 0)
                self.absorbances_mv = self.absorbances
        else:
            self.prev_point = point.copy()
            self.prev_init = True
            ds = 0.0
        
        ts = self.target_species.distribution.effective_temperature(point.x, point.y, point.z)
        if ts <= 0.0:
            return spectrum

        ion_velocity = self.target_species.distribution.bulk_velocity(point.x, point.y, point.z)
        
        # calculate emission line central wavelength, doppler shifted along observation direction
        shifted_wavelength = doppler_shift(self._wavelength, direction, ion_velocity)
        Td = self.target_species.distribution.neutral_temperature(point.x, point.y, point.z)
        # calculate the line width
        sigma = thermal_broadening(self._wavelength, ts, self._line.element.atomic_weight)

        return self.add_opaque_gaussian_line(radiance, absorbance, Td, ds, shifted_wavelength, sigma, spectrum)

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
            self._target_species.distribution.update_absorbance(self._line.transition)
        except ValueError:
            raise RuntimeError("The plasma object does not contain the ion species for the specified line "
                               "(element={}, ionisation={}).".format(self._line.element.symbol, self._line.charge))
        # identify wavelength
        self._wavelength = self._atomic_data.wavelength(self._line.element, self._line.charge, self._line.transition)

        # instance line shape renderer
        self._lineshape = self._lineshape_class(self._line, self._wavelength, self._target_species, self._plasma, self._atomic_data,
                                                *self._lineshape_args, **self._lineshape_kwargs)
        

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._lineshape = None

    cpdef Spectrum add_opaque_gaussian_line(self, double radiance, double absorbance, double Td, double ds, double wavelength, double sigma, Spectrum spectrum):
        r"""
        Adds a Gaussian line to the given spectrum and returns the new spectrum.

        The formula used is based on the following definite integral:
        :math:`\frac{1}{\sigma \sqrt{2 \pi}} \int_{\lambda_0}^{\lambda_1} \exp(-\frac{(x-\mu)^2}{2\sigma^2}) dx = \frac{1}{2} \left[ -Erf(\frac{a-\mu}{\sqrt{2}\sigma}) +Erf(\frac{b-\mu}{\sqrt{2}\sigma}) \right]`

        :param float radiance: Intensity of the line in radiance.
        :param float wavelength: central wavelength of the line in nm.
        :param float sigma: width of the line in nm.
        :param Spectrum spectrum: the current spectrum to which the gaussian line is added.
        :return:
        """

        cdef double temp
        cdef double cutoff_lower_wavelength, cutoff_upper_wavelength
        cdef double lower_wavelength, upper_wavelength
        cdef double lower_integral, upper_integral
        cdef int start, end, i
        cdef double delta_lambda_D = wavelength * sqrt(2*ELEMENTARY_CHARGE*Td/(3.344e-27*SPEED_OF_LIGHT**2))

        if sigma <= 0:
            return spectrum

        # calculate and check end of limits
        cutoff_lower_wavelength = wavelength - GAUSSIAN_CUTOFF_SIGMA * sigma
        if spectrum.max_wavelength < cutoff_lower_wavelength:
            return spectrum

        cutoff_upper_wavelength = wavelength + GAUSSIAN_CUTOFF_SIGMA * sigma
        if spectrum.min_wavelength > cutoff_upper_wavelength:
            return spectrum

        # locate range of bins where there is significant contribution from the gaussian (plus a health margin)
        start = max(0, <int> floor((cutoff_lower_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))
        end = min(spectrum.bins, <int> ceil((cutoff_upper_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))

        # add line to spectrum
        temp = 1 / (M_SQRT2 * sigma)
        lower_wavelength = spectrum.min_wavelength + start * spectrum.delta_wavelength
        lower_integral = erf((lower_wavelength - wavelength) * temp)
        for i in range(start, end):

            upper_wavelength = spectrum.min_wavelength + spectrum.delta_wavelength * (i + 1)
            upper_integral = erf((upper_wavelength - wavelength) * temp)
            self.absorbances_mv[i] += absorbance*expl(-((upper_wavelength-wavelength)/(delta_lambda_D))**2)*ds/(delta_lambda_D*1e-9)
            spectrum.samples_mv[i] += radiance * 0.5 * (upper_integral - lower_integral) / spectrum.delta_wavelength #*expl(-self.absorbances_mv[i])

            lower_wavelength = upper_wavelength
            lower_integral = upper_integral

        return spectrum
'''
Cherab AMJUEL plasma models

    Cherab itsel implements only ADAS plasma models and reading
    In the future these could be included as a cherab module, and Cythonized

'''


cdef class LineExcitation_AM(PlasmaModel):

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



