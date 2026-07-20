# cython: language_level=3
from cherab.core.atomic cimport AtomicData
from cherab.core.plasma cimport PlasmaModel

from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core cimport Line, Species, Plasma, Beam
from cherab.core.model.lineshape cimport GaussianLine, LineShapeModel
from cherab.core.utility.constants cimport RECIP_4_PI

cimport cython

import numpy as np
cimport numpy as np
from continuo cimport ContinuumRadiation, continuo_

def continuo(double wavelength_A,
             double Te_eV,
             int atomic_number,
             int ion_charge):
    """
    Calculate continuum radiation.

    Parameters
    ----------
    wavelength_A : float
        Wavelength in Angstrom.
    Te_eV : float
        Electron temperature in eV.
    atomic_number : int
        Nuclear charge Z.
    ion_charge : int
        Ion charge.

    Returns
    -------
    (free_free, free_bound)
    """

    cdef ContinuumRadiation result

    result = continuo_(
        wavelength_A,
        Te_eV,
        atomic_number,
        ion_charge
    )

    return result.free_free, result.free_bound




cdef class Continuo(PlasmaModel):
    """
    Emitter that calculates bremsstrahlung emission from a plasma object using the ADAS
    adaslib/continuo.f function.

    """
    
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
        return '<PlasmaModel - adaslib/continuo Bremsstrahlung>'

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef Spectrum _emission(self, Point3D point, Vector3D direction, Spectrum spectrum):

        cdef double ne, n, te, radiance, lower_sample, lower_wavelength, upper_sample, upper_wavelength, z_effective

        ne = self.plasma.electron_distribution.density(point.x, point.y, point.z)
        if ne <= 0.0:
            return spectrum

        te = self.plasma.electron_distribution.effective_temperature(point.x, point.y, point.z)
        if te <= 0.0:
            return spectrum

        z_effective = self.plasma.z_effective(point.x, point.y, point.z)

        # numerically integrate using trapezium rule
        # todo: add sub-sampling to increase numerical accuracy

        lower_wavelength = spectrum.min_wavelength
        lower_sample = self._continuo(lower_wavelength, te, ne, zeff=z_effective)

        # Define mem_view
        cdef double[:] spectrum_samples_mem_view = spectrum.samples

        cdef Py_ssize_t i
        cdef double value_at_i
        for i in range(spectrum.bins):

            upper_wavelength = spectrum.min_wavelength + spectrum.delta_wavelength * i

            upper_sample = self._continuo(upper_wavelength, te, ne, zeff=z_effective)
            value_at_i = spectrum_samples_mem_view[i]
            spectrum_samples_mem_view[i] = 0.5 * (lower_sample + upper_sample) + value_at_i #+ spectrum_samples_mem_view[i] 

            lower_wavelength = upper_wavelength
            lower_sample = upper_sample

        return spectrum
    
    cpdef Spectrum emission(self, Point3D point, Vector3D direction, Spectrum spectrum):
       return self._emission(point, direction, spectrum)


    cdef double _continuo(self, double wvl, double te, double ne, double zeff = 1.0):
        cdef int iz0, iz1
        cdef double wvl_A, contff, contin, radiance
        cdef ContinuumRadiation contrad
        wvl_A = wvl * 10.
        iz0=1
        iz1=1
        contrad = continuo_(wvl_A , te , iz0 , iz1 )
        tot = contrad.free_bound + contrad.free_free
        return RECIP_4_PI*contin*(1e-6)*ne*ne*10

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._lineshape = None



