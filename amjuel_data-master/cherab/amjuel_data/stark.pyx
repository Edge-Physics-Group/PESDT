# cython: language_level=3

# Slightly modified StarkBroadenedLine to add support for molecular contributions


import numpy as np
from scipy.special import hyp2f1

cimport numpy as np
from libc.math cimport sqrt, erf, M_SQRT2, floor, ceil, fabs
from scipy.special import hyp2f1
from raysect.optical.spectrum cimport new_spectrum
from raysect.optical cimport Spectrum, Point3D, Vector3D
from raysect.core.math.function.float cimport Function1D

from cherab.core.atomic.elements import hydrogen, deuterium, tritium, helium, helium3, beryllium, boron, carbon, nitrogen, oxygen, neon
from cherab.core.math.function cimport autowrap_function1d, autowrap_function2d
from cherab.core cimport Line, Species, Plasma, Beam
from cherab.core.math.integrators cimport GaussianQuadrature
from cherab.core.utility.constants cimport ATOMIC_MASS, ELEMENTARY_CHARGE, SPEED_OF_LIGHT
from cherab.core.model.lineshape cimport GaussianLine, LineShapeModel
from .molecules import Deuterium2, Deuterium3, Tritium2, Tritium3

cimport cython

# required by numpy c-api
np.import_array()

DEF LORENZIAN_CUTOFF_GAMMA = 50.0

cdef class StarkFunction(Function1D):
    """
    Normalised Stark function for the StarkBroadenedLine line shape.
    """
    #cdef dict __dict__
    cdef double _a, _x0, _norm

    STARK_NORM_COEFFICIENT = 4 * LORENZIAN_CUTOFF_GAMMA * hyp2f1(0.4, 1, 1.4, -(2 * LORENZIAN_CUTOFF_GAMMA)**2.5)

    def __init__(self, double wavelength, double lambda_1_2):

        if wavelength <= 0:
            raise ValueError("Argument 'wavelength' must be positive.")

        if lambda_1_2 <= 0:
            raise ValueError("Argument 'lambda_1_2' must be positive.")

        self._x0 = wavelength
        self._a = (0.5 * lambda_1_2)**2.5
        # normalise, so the integral over x is equal to 1 in the limits
        # (_x0 - LORENZIAN_CUTOFF_GAMMA * lambda_1_2, _x0 + LORENZIAN_CUTOFF_GAMMA * lambda_1_2)
        self._norm = (0.5 * lambda_1_2)**1.5 / <double> self.STARK_NORM_COEFFICIENT

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:

        return self._norm / ((fabs(x - self._x0))**2.5 + self._a)


cdef class StarkBroadenedLine(LineShapeModel):
    #cdef dict __dict__
    """
    Parametrised Stark broadened line shape based on the Model Microfield Method (MMM).
    Contains embedded atomic data in the form of fits to MMM.
    Only Balmer and Paschen series are supported by default.
    See B. Lomanowski, et al. "Inferring divertor plasma properties from hydrogen Balmer
    and Paschen series spectroscopy in JET-ILW." Nuclear Fusion 55.12 (2015)
    `123028 <https://doi.org/10.1088/0029-5515/55/12/123028>`_.

    Call `show_supported_transitions()` to see the list of supported transitions and
    default model coefficients.

    :param Line line: The emission line object for this line shape.
    :param float wavelength: The rest wavelength for this emission line.
    :param Species target_species: The target plasma species that is emitting.
    :param Plasma plasma: The emitting plasma object.
    :param dict stark_model_coefficients: Alternative model coefficients in the form
                                          {line_ij: (c_ij, a_ij, b_ij), ...}.
                                          If None, the default model parameters will be used.
    :param Integrator1D integrator: Integrator1D instance to integrate the line shape
        over the spectral bin. Default is `GaussianQuadrature()`.

    """

    STARK_MODEL_COEFFICIENTS_DEFAULT = {
        Line(hydrogen, 0, (3, 2)): (3.71e-18, 0.7665, 0.064),
        Line(hydrogen, 0, (4, 2)): (8.425e-18, 0.7803, 0.050),
        Line(hydrogen, 0, (5, 2)): (1.31e-15, 0.6796, 0.030),
        Line(hydrogen, 0, (6, 2)): (3.954e-16, 0.7149, 0.028),
        Line(hydrogen, 0, (7, 2)): (6.258e-16, 0.712, 0.029),
        Line(hydrogen, 0, (8, 2)): (7.378e-16, 0.7159, 0.032),
        Line(hydrogen, 0, (9, 2)): (8.947e-16, 0.7177, 0.033),
        Line(hydrogen, 0, (4, 3)): (1.330e-16, 0.7449, 0.045),
        Line(hydrogen, 0, (5, 3)): (6.64e-16, 0.7356, 0.044),
        Line(hydrogen, 0, (6, 3)): (2.481e-15, 0.7118, 0.016),
        Line(hydrogen, 0, (7, 3)): (3.270e-15, 0.7137, 0.029),
        Line(hydrogen, 0, (8, 3)): (4.343e-15, 0.7133, 0.032),
        Line(hydrogen, 0, (9, 3)): (5.588e-15, 0.7165, 0.033),
        Line(deuterium, 0, (3, 2)): (3.71e-18, 0.7665, 0.064),
        Line(deuterium, 0, (4, 2)): (8.425e-18, 0.7803, 0.050),
        Line(deuterium, 0, (5, 2)): (1.31e-15, 0.6796, 0.030),
        Line(deuterium, 0, (6, 2)): (3.954e-16, 0.7149, 0.028),
        Line(deuterium, 0, (7, 2)): (6.258e-16, 0.712, 0.029),
        Line(deuterium, 0, (8, 2)): (7.378e-16, 0.7159, 0.032),
        Line(deuterium, 0, (9, 2)): (8.947e-16, 0.7177, 0.033),
        Line(deuterium, 0, (4, 3)): (1.330e-16, 0.7449, 0.045),
        Line(deuterium, 0, (5, 3)): (6.64e-16, 0.7356, 0.044),
        Line(deuterium, 0, (6, 3)): (2.481e-15, 0.7118, 0.016),
        Line(deuterium, 0, (7, 3)): (3.270e-15, 0.7137, 0.029),
        Line(deuterium, 0, (8, 3)): (4.343e-15, 0.7133, 0.032),
        Line(deuterium, 0, (9, 3)): (5.588e-15, 0.7165, 0.033),
        Line(tritium, 0, (3, 2)): (3.71e-18, 0.7665, 0.064),
        Line(tritium, 0, (4, 2)): (8.425e-18, 0.7803, 0.050),
        Line(tritium, 0, (5, 2)): (1.31e-15, 0.6796, 0.030),
        Line(tritium, 0, (6, 2)): (3.954e-16, 0.7149, 0.028),
        Line(tritium, 0, (7, 2)): (6.258e-16, 0.712, 0.029),
        Line(tritium, 0, (8, 2)): (7.378e-16, 0.7159, 0.032),
        Line(tritium, 0, (9, 2)): (8.947e-16, 0.7177, 0.033),
        Line(tritium, 0, (4, 3)): (1.330e-16, 0.7449, 0.045),
        Line(tritium, 0, (5, 3)): (6.64e-16, 0.7356, 0.044),
        Line(tritium, 0, (6, 3)): (2.481e-15, 0.7118, 0.016),
        Line(tritium, 0, (7, 3)): (3.270e-15, 0.7137, 0.029),
        Line(tritium, 0, (8, 3)): (4.343e-15, 0.7133, 0.032),
        Line(tritium, 0, (9, 3)): (5.588e-15, 0.7165, 0.033),
        # Add Molecules -> Use the same coeff as for parent atom, as the "molecule line" is just the contribution of the
        # Molecule to the excited atomic polulation
        Line(Deuterium2, 0, (3, 2)): (3.71e-18, 0.7665, 0.064),
        Line(Deuterium2, 0, (4, 2)): (8.425e-18, 0.7803, 0.050),
        Line(Deuterium2, 0, (5, 2)): (1.31e-15, 0.6796, 0.030),
        Line(Deuterium2, 0, (6, 2)): (3.954e-16, 0.7149, 0.028),
        Line(Deuterium2, 0, (7, 2)): (6.258e-16, 0.712, 0.029),
        Line(Deuterium2, 0, (8, 2)): (7.378e-16, 0.7159, 0.032),
        Line(Deuterium2, 0, (9, 2)): (8.947e-16, 0.7177, 0.033),
        Line(Deuterium2, 0, (4, 3)): (1.330e-16, 0.7449, 0.045),
        Line(Deuterium2, 0, (5, 3)): (6.64e-16, 0.7356, 0.044),
        Line(Deuterium2, 0, (6, 3)): (2.481e-15, 0.7118, 0.016),
        Line(Deuterium2, 0, (7, 3)): (3.270e-15, 0.7137, 0.029),
        Line(Deuterium2, 0, (8, 3)): (4.343e-15, 0.7133, 0.032),
        Line(Deuterium2, 0, (9, 3)): (5.588e-15, 0.7165, 0.033),
        Line(Tritium2, 0, (3, 2)): (3.71e-18, 0.7665, 0.064),
        Line(Tritium2, 0, (4, 2)): (8.425e-18, 0.7803, 0.050),
        Line(Tritium2, 0, (5, 2)): (1.31e-15, 0.6796, 0.030),
        Line(Tritium2, 0, (6, 2)): (3.954e-16, 0.7149, 0.028),
        Line(Tritium2, 0, (7, 2)): (6.258e-16, 0.712, 0.029),
        Line(Tritium2, 0, (8, 2)): (7.378e-16, 0.7159, 0.032),
        Line(Tritium2, 0, (9, 2)): (8.947e-16, 0.7177, 0.033),
        Line(Tritium2, 0, (4, 3)): (1.330e-16, 0.7449, 0.045),
        Line(Tritium2, 0, (5, 3)): (6.64e-16, 0.7356, 0.044),
        Line(Tritium2, 0, (6, 3)): (2.481e-15, 0.7118, 0.016),
        Line(Tritium2, 0, (7, 3)): (3.270e-15, 0.7137, 0.029),
        Line(Tritium2, 0, (8, 3)): (4.343e-15, 0.7133, 0.032),
        Line(Tritium2, 0, (9, 3)): (5.588e-15, 0.7165, 0.033),

    }
    #cdef double self._aij, self._cij, self._bij
    cdef dict __dict__

    def __init__(self, Line line, double wavelength, Species target_species, Plasma plasma,
                 dict stark_model_coefficients=None, integrator=GaussianQuadrature()):

        stark_model_coefficients = stark_model_coefficients or self.STARK_MODEL_COEFFICIENTS_DEFAULT

        try:
            # Fitted Stark Constants
            cij, aij, bij = stark_model_coefficients[line]
            if cij <= 0:
                raise ValueError('Coefficient c_ij must be positive.')
            if aij <= 0:
                raise ValueError('Coefficient a_ij must be positive.')
            if bij <= 0:
                raise ValueError('Coefficient b_ij must be positive.')
            self._aij = aij
            self._bij = bij
            self._cij = cij
        except IndexError:
            raise ValueError('Stark broadening coefficients for {} is not currently available.'.format(line))

        super().__init__(line, wavelength, target_species, plasma, integrator)

    def show_supported_transitions(self):
        """ Prints all supported transitions."""
        for line, coeff in self.STARK_MODEL_COEFFICIENTS_DEFAULT.items():
            print('{}: c_ij={}, a_ij={}, b_ij={}'.format(line, coeff[0], coeff[1], coeff[2]))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef Spectrum add_line(self, double radiance, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef:
            double ne, te, lambda_1_2, lambda_5_2, wvl
            double cutoff_lower_wavelength, cutoff_upper_wavelength
            double lower_wavelength, upper_wavelength
            double bin_integral
            int start, end #, i
            Spectrum raw_lineshape

        ne = self.plasma.get_electron_distribution().density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self.plasma.get_electron_distribution().effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        lambda_1_2 = self._cij * ne**self._aij / (te**self._bij)

        self.integrator.function = StarkFunction(self.wavelength, lambda_1_2)

        # calculate and check end of limits
        cutoff_lower_wavelength = self.wavelength - LORENZIAN_CUTOFF_GAMMA * lambda_1_2
        if spectrum.max_wavelength < cutoff_lower_wavelength:
            return spectrum

        cutoff_upper_wavelength = self.wavelength + LORENZIAN_CUTOFF_GAMMA * lambda_1_2
        if spectrum.min_wavelength > cutoff_upper_wavelength:
            return spectrum

        # locate range of bins where there is significant contribution from the gaussian (plus a health margin)
        start = max(0, <int> floor((cutoff_lower_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))
        end = min(spectrum.bins, <int> ceil((cutoff_upper_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength))

        # add line to spectrum
        lower_wavelength = spectrum.min_wavelength + start * spectrum.delta_wavelength

        # Def memory view
        cdef double[:] spectrum_samples_mem_view = spectrum.samples_mv
        cdef double value_at_i
        cdef Py_ssize_t i
        for i in range(start, end):
            upper_wavelength = spectrum.min_wavelength + spectrum.delta_wavelength * (i + 1)

            bin_integral = self.integrator.evaluate(lower_wavelength, upper_wavelength)
            value_at_i = spectrum_samples_mem_view[i]
            spectrum_samples_mem_view[i] = value_at_i + radiance * bin_integral / spectrum.delta_wavelength

            lower_wavelength = upper_wavelength

        return spectrum