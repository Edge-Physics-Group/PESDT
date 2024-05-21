import os

from cherab.core.atomic cimport AtomicData
from cherab.core.plasma cimport PlasmaModel
from cherab.core.utility import PerCm3ToPerM3, PhotonToJ, Cm3ToM3
from amread import photon_rate_coeffs, A_coeff
from amread import wavelength as wl

from raysect.optical cimport Spectrum, Point3D, Vector3D
from cherab.core cimport Line, Species, Plasma, Beam
from cherab.core.model.lineshape cimport GaussianLine, LineShapeModel
from cherab.core.utility.constants cimport RECIP_4_PI
#from adaslib cimport continuo_

cimport cython

#cdef extern from "/home/adas/include/adaslib.h":
#    cdef void continuo_(double *wave   , double *tev    , int *iz0,
#                      int *iz1       , double *contff , double *contin )

from adaslib.atomic.continuo_if import continuo_if

import numpy as np


cdef class Continuo(PlasmaModel):
    """
    Emitter that calculates bremsstrahlung emission from a plasma object using the ADAS
    adaslib/continuo.f function.

    """
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

        wvl_A = wvl * 10.
        iz0=1
        iz1=1
        contin, confff = continuo_if(wvl_A , te , iz0 , iz1 )

        return RECIP_4_PI*contin*(1e-6)*ne*ne*10

    '''
    cdef double _continuo(self, double wvl, double te, double ne, double zeff = 1.0):
        # TODO: implement zeff using adaslib/continuo function. This is a weighted sum of the
        # main ion and impurity ion densities
        cdef double wvl_A, iz0, iz1, contff, contin, pi, radiance
        """
        adaslib/continuo wrapper

        :param wvl: in nm
        :param te: in eV
        :param ne: in m^-3
        :param zeff: a.u.
        :return:

        /home/adas/python/adaslib/atomic/continuo.py doc

          PURPOSE    : calculates continuum emission at a requested wavelength
                       and temperature for an element ionisation stage.

          contff, contin = continuo(wave, tev, iz0, iz1)

                        NAME         TYPE     DETAILS
          REQUIRED   :  wave()       float    wavelength required (A)
                        tev()        float    electron temperature (eV)
                        iz0          int      atomic number
                        iz1          int      ion stage + 1

          RETURNS       contff(,)    float    free-free emissivity (ph cm3 s-1 A-1)
                        contin(,)    float    total continuum emissivity
                                              (free-free + free-bound) (ph cm3 s-1 A-1)
                                                  dimensions: wave, te (dropped if just 1).

          MODIFIED   :
               1.1     Martin O'Mullane
                         - First version

          VERSION    :
                1.1    16-06-2016


        """
        wvl_A = wvl * 10.
        iz0=1
        iz1=1
        contff, contin = continuo(wvl_A, te, iz0, iz1)
        pi = 3.141592

        # Convert to ph/s/m^3/str/nm
        contin = (1. / (4 * pi)) * contin * ne * ne * (1.0e-06) * 10.0 # ph/s/m^3/str/nm

        # radiance =  PhotonToJ.to(contin, wvl) # W/m^3/str/nm

        return radiance
    '''

    def _change(self):

        # clear cache to force regeneration on first use
        self._target_species = None
        self._wavelength = 0.0
        self._rates = None
        self._lineshape = None



